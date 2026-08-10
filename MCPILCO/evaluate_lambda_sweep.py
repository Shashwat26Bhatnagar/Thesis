#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_lambda_sweep.py   (repo root)

Select the L2 action-regularisation weight lambda WITHOUT touching the simulator.

THE CRITERION (per-hour counting, not averaging)
    Train one REFERENCE policy with lambda = 0 (chance constraints only), and one
    policy per candidate lambda. Then, hour by hour over the expert's horizon:

        good hour  <=>  W2_lam(h) <= W2_ref(h) * (1 + tol)      imitation not hurt
                   AND  ||a||_lam(h) <  ||a||_ref(h)             actions smaller

    Choose the lambda with the MOST good hours.

WHY COUNT HOURS RATHER THAN AVERAGE
    Averaging action values across policies hides multimodality. The classic
    illustration is ALVINN at a fork in the road: the network's steering density is
    bimodal (go left OR go right), and the MEAN of those modes is "straight" -- which
    belongs to neither mode and drives into the tree. The same failure appeared here
    concretely: two policies with near-identical mean flows produced completely
    different physics, because large opposing flows can average like small ones.
    A COUNT of hours cannot average two modes into a third that neither achieves.

EVALUATION IS DETERMINISTIC AND SHARED
    Every policy is evaluated on IDENTICAL start states (one fixed seed) with
    p_dropout = 0 and mean propagation (particle_pred=False). Without this the
    comparison would be dominated by sampling noise rather than by lambda.

NOTE ON THE HORIZON
    The expert (capped_traj.npz) covers 0..150 h, so the per-hour comparison exists
    for 150 hours, not the PenSim episode's 230. Hours beyond 150 have no expert to
    compare against.

    python evaluate_lambda_sweep.py -model results_pensim/rbf_model_bnd_rbf_iter0.pt \\
        -ref results_pensim/cdil_policy_lam0.pt \\
        -cand results_pensim/cdil_policy_lam*.pt
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
for _c in (os.path.expanduser("~/Thesis/penicillin-dcfba"),
           os.path.expanduser("~/penicillin-dcfba")):
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c); break

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
from policy_learning.gp_particle_rollout import gp_rollout, sample_initial_particles
from policy_learning.policy_variants import rebuild_policy
from policy_learning.wasserstein_loss import w2_cross_dim_torch
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")

ap = argparse.ArgumentParser()
ap.add_argument("-model", default=None, help="single frozen world model")
ap.add_argument("-phase_prefix", default=None,
                help="use THREE phase models <prefix>_phase{0,1,2}.pt, selected per "
                     "hour by expert time -- must match how the policies were trained")
ap.add_argument("-ref", required=True, help="reference policy (lambda = 0)")
ap.add_argument("-cand", nargs="+", required=True, help="candidate policies")
ap.add_argument("-tol", type=float, default=0.05,
                help="how much W2 may degrade vs the reference and still count as "
                     "'imitation not hurt' (0.05 = 5%%)")
ap.add_argument("-num_states", type=int, default=100)
ap.add_argument("-k_actions", type=int, default=5)
ap.add_argument("-seed", type=int, default=0)
ap.add_argument("-out", type=str, default="results_pensim/lambda_sweep.json")
ap.add_argument("-select_out", type=str, default=None,
                help="copy the winning policy here. The explore stage is submitted "
                     "BEFORE this evaluation runs, so it cannot know which lambda "
                     "wins -- it reads this fixed path instead.")
args = ap.parse_args()

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = STATE_DIM + INPUT_DIM
STEPS_PER_HOUR = 5
EXPERT_TIMES = np.arange(1.0, 150.0 + 1e-9, 1.0)


# ------------------------------------------------------------------ model ----
def _load(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    init = dict(active_dims=np.arange(0, GP_IN), lengthscales_init=np.ones(GP_IN),
                flg_train_lengthscales=True, lambda_init=np.ones(1),
                flg_train_lambda=True, sigma_n_init=1e-2 * np.ones(1),
                sigma_n_num=1e-4, flg_train_sigma_n=True, dtype=dtype, device=device)
    m = ML.Model_learning_RBF(num_gp=STATE_DIM,
                              init_dict_list=[dict(init) for _ in range(STATE_DIM)],
                              approximation_mode=None, dtype=dtype, device=device,
                              flg_norm=False)
    m.load_state_dict(ck["state_dict"])
    for k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
              "K_X_inv_list", "gp_inputs_tr_list"):
        setattr(m, k, ck[k])
    m.num_samples = ck["gp_inputs"].shape[0]
    m.dim_state, m.dim_input = STATE_DIM, INPUT_DIM
    m.norm_list = [1.0] * STATE_DIM
    m.set_eval_mode()
    return m


if not args.model and not args.phase_prefix:
    raise SystemExit("give -model or -phase_prefix")

# The evaluation must use the SAME model arrangement the policies were trained
# against, or the W2 values are not comparable to what the optimiser saw.
if args.phase_prefix:
    MODELS = {p: _load(f"{args.phase_prefix}_phase{p}.pt") for p in (0, 1, 2)}
    print("phase models:")
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        hi_s = "inf" if hi > 1e8 else f"{hi:g}"
        print(f"  phase {p}: [{lo:g},{hi_s}) h  train pts={MODELS[p].gp_inputs.shape[0]}")
    def model_at(t):
        for p in (0, 1, 2):
            lo, hi = pdata.PHASES[p]
            if lo <= t < hi:
                return MODELS[p]
        return MODELS[2]
    POOL = torch.cat([MODELS[p].gp_inputs[:, :STATE_DIM] for p in (0, 1, 2)], dim=0)
else:
    _M = _load(args.model)
    def model_at(t):
        return _M
    POOL = _M.gp_inputs[:, :STATE_DIM]
    print(f"world model: {os.path.basename(args.model)}  train pts={POOL.shape[0]}")

# ----------------------------------------------------------------- expert ----
q = PFQuery(verbose=False)
EIG = {round(float(t), 6): torch.linalg.eigvalsh(
           torch.tensor(np.asarray(q.next_state_distribution(t=float(t), source="traj")
                                   ["cov_n"]).tolist(), dtype=dtype))
       for t in EXPERT_TIMES}
print(f"expert: {len(EIG)} hourly distributions (1..150 h)")

# ------------------------------------------------- shared evaluation states ----
# ONE fixed draw, reused for every policy: otherwise the comparison measures sampling
# noise rather than lambda.
rng = np.random.default_rng(args.seed)
S0 = {}
for t in EXPERT_TIMES:
    st = sample_initial_particles(POOL, args.num_states, generator=rng,
                                  dtype=dtype, device=device)
    S0[round(float(t), 6)] = st.repeat_interleave(args.k_actions, dim=0)


def evaluate(policy_path):
    """Per-hour W2 and mean ||a||^2 for one policy, on the shared states."""
    pk = torch.load(policy_path, map_location=device, weights_only=False)
    pol = rebuild_policy(pk["policy_meta"], dtype=dtype, device=device)
    pol.load_state_dict(pk["policy_state_dict"])
    pol.eval()

    w2_h, a2_h = [], []
    for t in EXPERT_TIMES:
        key = round(float(t), 6)
        acc_var, acc_a = None, []

        def _loss(t, s, a, mu, cov, s_next):
            nonlocal acc_var
            acc_var = cov if acc_var is None else acc_var + cov
            acc_a.append(a)
            return torch.zeros((), dtype=cov.dtype, device=cov.device)

        with torch.no_grad():
            gp_rollout(model=model_at(float(t)), policy=pol, s0=S0[key],
                       T=STEPS_PER_HOUR,
                       p_dropout=0.0,            # deterministic
                       particle_pred=False,      # mean propagation
                       loss_fn=_loss, graph_mode="full")
            d = w2_cross_dim_torch(acc_var, EIG[key])
            w2_h.append(float(d.view(args.num_states, args.k_actions)
                              .mean(dim=1).mean()))
            a2_h.append(float((torch.cat(acc_a, 0) ** 2).sum(dim=1).mean()))
    return np.array(w2_h), np.array(a2_h), pk.get("hist", [None])[-1]


def lam_of(path):
    m = re.search(r"lam([0-9pEe+-]+)", os.path.basename(path))
    if not m:
        return None
    return float(m.group(1).replace("p", "."))



print(f"\nevaluating reference: {os.path.basename(args.ref)}")
w2_ref, a2_ref, hist_ref = evaluate(args.ref)
print(f"  W2 per hour: mean={w2_ref.mean():.5f}  ||a||^2: mean={a2_ref.mean():.4f}")

rows = []
cands = [c for c in sorted(set(sum([glob.glob(p) for p in args.cand], [])))
         if os.path.abspath(c) != os.path.abspath(args.ref)]
for c in cands:
    w2, a2, h = evaluate(c)
    ok_w2 = w2 <= w2_ref * (1.0 + args.tol)
    ok_a = a2 < a2_ref
    good = int((ok_w2 & ok_a).sum())
    rows.append(dict(path=c, lam=lam_of(c), good_hours=good,
                     hours_w2_ok=int(ok_w2.sum()), hours_a_ok=int(ok_a.sum()),
                     w2_mean=float(w2.mean()), a2_mean=float(a2.mean()),
                     w2_rel=float(w2.mean() / w2_ref.mean()),
                     a2_rel=float(a2.mean() / a2_ref.mean())))
    print(f"  {os.path.basename(c):40s} good={good:3d}/{len(EXPERT_TIMES)}")

print("\n" + "=" * 100)
print(f"{'policy':38s}{'lambda':>10}{'GOOD hrs':>10}{'W2 ok':>8}{'|a| ok':>8}"
      f"{'W2 /ref':>10}{'|a|^2 /ref':>12}")
print("-" * 100)
print(f"{'REFERENCE (lambda=0)':38s}{0.0:>10}{'-':>10}{'-':>8}{'-':>8}"
      f"{1.0:>10.3f}{1.0:>12.3f}")
for r in sorted(rows, key=lambda r: -r["good_hours"]):
    print(f"{os.path.basename(r['path']):38s}"
          f"{(r['lam'] if r['lam'] is not None else float('nan')):>10.4g}"
          f"{r['good_hours']:>10d}{r['hours_w2_ok']:>8d}{r['hours_a_ok']:>8d}"
          f"{r['w2_rel']:>10.3f}{r['a2_rel']:>12.3f}")

if rows:
    best = max(rows, key=lambda r: r["good_hours"])
    print(f"\nSELECTED: lambda = {best['lam']}  ({best['good_hours']} good hours of "
          f"{len(EXPERT_TIMES)}, W2 {best['w2_rel']:.3f}x reference, "
          f"||a||^2 {best['a2_rel']:.3f}x reference)")
    print("\nA 'good hour' means W2 within tol of the reference AND a smaller action")
    print("norm -- counted per hour, so two modes cannot average into a third.")

if args.select_out and rows:
    import shutil
    best = max(rows, key=lambda r: r["good_hours"])
    os.makedirs(os.path.dirname(args.select_out) or ".", exist_ok=True)
    shutil.copy2(best["path"], args.select_out)
    print(f"selected policy copied -> {args.select_out}")
    with open(os.path.splitext(args.select_out)[0] + "_selection.json", "w") as f:
        json.dump({"selected": best["path"], "lambda": best["lam"],
                   "good_hours": best["good_hours"],
                   "n_hours": len(EXPERT_TIMES), "tol": args.tol,
                   "w2_rel": best["w2_rel"], "a2_rel": best["a2_rel"]}, f, indent=2)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump({"tol": args.tol, "reference": args.ref,
           "phase_prefix": args.phase_prefix, "model": args.model,
           "w2_ref_mean": float(w2_ref.mean()), "a2_ref_mean": float(a2_ref.mean()),
           "n_hours": len(EXPERT_TIMES), "rows": rows}, open(args.out, "w"), indent=2)
print(f"\nsaved -> {args.out}")
