#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/cdil_policy_smooth.py

CDIL policy optimization, rebuilt clean. Writes to results_clean/ and does not touch
any existing file.

WHY THIS FILE EXISTS: THE LOSS IS NON-SMOOTH AND ADAM CANNOT DESCEND IT

    Two facts, both measured:

    1. Far better per-hour actions EXIST. Random search over 400 actions per hour beat
       the mean action by 87-90% at t = 50..150, and the winners differ completely
       between hours -- discharge +0.76 at t=75 against -1.59 at t=130.

    2. Gradient descent does not find them. Fifteen Adam steps on a single window,
       starting from theta_0, moved the loss by ~1% and INCREASED it on four of seven
       windows (t=30 +1.7%, t=75 +6.4%, t=100 +5.4%). The resulting per-hour policies
       phi_h were already near-identical, at 0.28-1.04% of the reference's action
       spread, BEFORE any meta-averaging -- so the outer average is not the cause.

    Chewi et al. ("Averaging on the Bures-Wasserstein manifold") state the reason
    directly: W2(Sigma, .) "is neither geodesically convex nor geodesically smooth,
    nor Euclidean convex nor Euclidean smooth ... it poses challenges for
    optimization", and they smooth the objective before optimising, with exactly
    W_{2,eps} := sqrt(W2^2 + eps^2). Tropical Gradient Descent reports the same
    symptom on Wasserstein projection problems: "stable local minima ... Classical
    descent, Adam, and Adamax are particularly susceptible", worst in low dimensions.

    This file applies that smoothing (wasserstein_loss.SMOOTH_EPS). The un-smoothed
    distance ends in sqrt(cost), whose derivative diverges as cost -> 0, so particles
    near a zero contribute enormous near-random directions that cancel when averaged
    over 500. Bounding the derivative at 1/(2*eps) is the standard fix.

    NOTE the distance is now biased upward by ~eps at its minimum, so W2 values are
    NOT comparable to earlier runs.

WHY THE IMITATION TERM WAS SILENT BEFORE THAT

    w2_cross_dim_torch now trace-normalises both spectra before comparing (see the
    note in that function). Without it the loss was EXACTLY ZERO -- no loss, no
    gradient -- for all six action channels across a in [-0.5, +0.5], measured at
    t=75 h with the summed 5-step covariance that training actually uses. The trained
    policies sat at z ~ 0.01-0.07, inside that dead zone.

    The cause was a normalisation mismatch, not the distance itself. The expert's
    cov_n comes from state/[20, 80, 2.5] with no centring; the GP's covariance comes
    from physical -> smpl min-max -> z-score, whose per-channel divisors span 103x.
    The two spectra overlapped only by coincidence of independent rescalings, and
    that overlap put the expert's eigenvalues inside the Cai-Lim clamp band, where
    s_star == gamma and the cost is identically zero.

    WHAT THAT EXPLAINS. With W2 silent, the only live gradients were the action L2 and
    the chance box, and BOTH pull toward z = 0 -- which, under z-scoring, IS the
    dataset-mean action. So the policy converged to the mean on every channel, at
    ~1% of the reference's action variance, and nine successive objectives (L1, L2,
    minimum-action, reward maximisation, Bernoulli gate, ASRE sparsity KL, a separate
    valve policy, PPO, Reptile) all produced the same collapse -- eight of them were
    regularisers layered on a term contributing nothing, and the meta-updates were
    redistributing a gradient that was zero.

    AFTER THE FIX, measured: no zeros anywhere, every channel varies with a range of
    0.10-0.29. The distance no longer reaches 0 (a 3-shape and an 8-shape cannot
    coincide, so there is a floor near 0.05-0.16) and the landscape is jagged rather
    than smooth. W2 VALUES FROM THIS FILE ARE NOT COMPARABLE to any earlier run.

WHAT IS IN
    W2      E_s[ E_{a|s}[ W2( P(s'|s,a) || P_expert(s'|s) ) ] ]
            Cai-Lim cross-dimensional distance, 8-D model vs 3-D expert. Means drop
            out by construction -- the objective matches covariance spectra only.
    L2      lambda * mean||a||^2 over ALL SIX channels, ADDITIVE.
    chance  Tan et al. Eq. 8-9 soft chance constraints: the static action box, and a
            floor on the predicted vessel weight.

WHAT IS OUT (deliberately, after all of these were tried and did not survive)
    Bernoulli gate on discharge      the gate contradicted the chance constraint
                                     (a two-point distribution has maximal variance,
                                     so the variance back-off flagged it in 150/150
                                     windows and the penalty reached 1156x the W2
                                     term). Adding a learned magnitude then gave the
                                     gate a degenerate optimum -- hold it open and set
                                     the level near zero -- which is the continuous
                                     head it was meant to replace.
    ASRE sparsity KL                 fixes the marginal duty cycle, not the temporal
                                     structure; the deployed policy still flickered
                                     (51 opens of 0.22 h against the reference's 6 of
                                     2.0 h).
    multiplicative violation penalty confounded with the L2 term.
    separate valve policy            its state-matching target was minimised by not
                                     discharging at all (duty went to 0 by iteration 7
                                     and stayed).
    discharge override               useful as an ablation, not as a policy.

VERIFYING THAT L2 ACTUALLY BINDS
    In an earlier sweep lambda=0.001 came back with an action norm 1.656x the
    lambda=0 reference -- more regularisation, larger actions. That is backwards, and
    either the term was not reaching the loss or run-to-run variance swamped it. This
    file therefore logs mean||a||^2 EVERY iteration alongside lambda*||a||^2, and
    prints the first-to-last change at the end, so the effect of lambda is visible
    directly rather than inferred from the sweep table.

STRUCTURE (MC-PILCO, Amadio et al. 2022)
    Each expert hour is one self-contained episode: T = 5 steps, a FRESH
    in-distribution start state, one graph, one backward, one update. A single long
    continuing rollout instead drove |s| to ~140 z-units against training data
    spanning [-5.5, 11.2], which killed the gradient twice over -- the GP's predictive
    variance saturated at the prior, and the policy's RBF basis abandoned its centres.

    python policy_learning/cdil_policy_smooth.py \\
        -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 -lam 0.01 -iters 20
"""
import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _c in (os.path.expanduser("~/Thesis/penicillin-dcfba"),
           os.path.expanduser("~/penicillin-dcfba")):
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c); break

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
from policy_learning.gp_particle_rollout import gp_rollout, sample_initial_particles
from policy_learning.policy_variants import build_policy, rebuild_policy
from policy_learning.wasserstein_loss import w2_cross_dim_torch
from policy_learning.chance_constraints import (action_chance_penalty,
                                                state_chance_penalty, phi_inv)
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

SAVE_DIR = os.path.join(_REPO, "results_smooth")
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

# --- E_s( E_{a|s}( . ) ) ---
NUM_STATES, K_ACTIONS = 100, 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

# --- episodic structure ---
T_START_HOURS, HOURS_PER_STEP, EXPERT_DT = 0.0, 0.2, 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))       # 5 = one hour
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0
WINDOWS_PER_ITER = 150
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0

# --- policy ---
NUM_BASIS, U_MAX = 200, 3.0
CENTER_RANGE_PAD = 1.10

# --- L2 ---
LAMBDA_A = 0.0                       # set with -lam; 0.0 is the sweep reference

# --- chance constraints (Tan et al. Eq. 8-9) ---
CC_EPS = 0.95                        # paper: 95% -> Phi^-1 = 1.6449
CC_ALPHA_ACT = 1000.0                # paper's alpha for the action box
CC_ALPHA_STATE = 1.0                 # calibrated: at 1000 the vessel penalty was
                                     # ~700x the W2 term and the policy optimised the
                                     # constraint alone
CC_WT_MIN_PHYS = 50000.0             # reference runs stay above 91000; batches start
                                     # near 62500

EXPERT_COV_KEY = "cov_n"             # the network's OWN normalised covariance. The
                                     # physical one has eigenvalues ~349x larger than
                                     # the GP's z-scored ones, which made the loss a
                                     # fixed unclosable offset.

_ap = argparse.ArgumentParser("CDIL policy optimization (clean)")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt, selected per "
                      "window by the expert time")
_ap.add_argument("-lam", type=float, default=0.0, help="L2 weight on ||a||^2")
_ap.add_argument("-iters", type=int, default=None)
_ap.add_argument("-out", default=None)
_ap.add_argument("-init_policy", default=None, help="warm start")
_args = _ap.parse_known_args()[0]
LAMBDA_A = _args.lam
if _args.iters:
    N_ITERS = _args.iters
OUT = _args.out or os.path.join(SAVE_DIR,
        f"sm_e{str(SMOOTH_EPS).replace('.','p')}_lam{str(LAMBDA_A).replace('.','p')}.pt")


# ================================================================ world models ===
def load_model(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    init = dict(active_dims=np.arange(0, GP_INPUT_DIM),
                lengthscales_init=np.ones(GP_INPUT_DIM), flg_train_lengthscales=True,
                lambda_init=np.ones(1), flg_train_lambda=True,
                sigma_n_init=1e-2 * np.ones(1), sigma_n_num=1e-4,
                flg_train_sigma_n=True, dtype=dtype, device=device)
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
    return m, ck


MODELS, CKS = {}, {}
for _p in (0, 1, 2):
    MODELS[_p], CKS[_p] = load_model(f"{_args.phase_prefix}_phase{_p}.pt")
stats = {k: np.asarray(CKS[0][k]) for k in
         ("std_obs_mu", "std_obs_sd", "std_act_mu", "std_act_sd")}

# every model must share one z-space or switching between them is meaningless
_ref_sd = np.asarray(CKS[0]["std_obs_sd"])
for _p in (1, 2):
    _d = float(np.abs(np.asarray(CKS[_p]["std_obs_sd"]) - _ref_sd).max())
    if _d > 1e-10:
        raise RuntimeError(f"phase {_p} standardized differently (max diff {_d:.3e})")

print("world models:")
for _p in (0, 1, 2):
    lo, hi = pdata.PHASES[_p]
    hi_s = "inf" if hi > 1e8 else f"{hi:g}"
    print(f"  phase {_p}: [{lo:g},{hi_s}) h  train pts={MODELS[_p].gp_inputs.shape[0]}")
print("  -> one shared z-space (verified)")

POOL = torch.cat([MODELS[p].gp_inputs[:, :STATE_DIM] for p in (0, 1, 2)], 0)
POOLS = {p: MODELS[p].gp_inputs[:, :STATE_DIM] for p in (0, 1, 2)}
s_lo, s_hi = POOL.min(0).values, POOL.max(0).values
print(f"combined state range: [{s_lo.min():.2f}, {s_hi.max():.2f}] z")


def phase_of(t_h):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t_h < hi:
            return p
    return 2


# ====================================================================== expert ===
_q = PFQuery(verbose=True)
EXPERT_TIMES = np.arange(EXPERT_T_MIN, EXPERT_T_MAX + 1e-9, EXPERT_DT)
print(f"pre-caching {len(EXPERT_TIMES)} expert distributions ...", flush=True)
EXPERT_EIGS = {}
for _t in EXPERT_TIMES:
    _d = _q.next_state_distribution(t=float(_t), source="traj")
    EXPERT_EIGS[round(float(_t), 6)] = torch.linalg.eigvalsh(
        torch.tensor(np.asarray(_d[EXPERT_COV_KEY]).tolist(), dtype=dtype,
                     device=device))
print(f"  eigenvalues @75h: {EXPERT_EIGS[75.0].numpy()}", flush=True)
_cnt = {}
for _t in EXPERT_TIMES:
    _cnt[phase_of(float(_t))] = _cnt.get(phase_of(float(_t)), 0) + 1
print("windows per model: " + "  ".join(f"phase {k}: {v}" for k, v in sorted(_cnt.items())))


# ====================================================================== policy ===
_warm = None
centers_init = lengthscales_init = None
if _args.init_policy and os.path.exists(_args.init_policy):
    _warm = torch.load(_args.init_policy, map_location=device, weights_only=False)
    _m = _warm["policy_meta"]
    # the standardizer refits on the union each iteration, so the same physical state
    # maps to a different z; the centres are remapped to preserve behaviour in
    # PHYSICAL units
    c0 = np.array(np.asarray(_m["centers_init"]).tolist(), dtype=np.float64)
    mo, so = np.asarray(_warm["std_obs_mu"]), np.asarray(_warm["std_obs_sd"])
    mn, sn = stats["std_obs_mu"], stats["std_obs_sd"]
    centers_init = (c0 * so + mo - mn) / sn
    lengthscales_init = (np.array(np.asarray(_m["lengthscales_init"]).tolist(),
                                  dtype=np.float64) * so / sn)
    print(f"[warm] from {os.path.basename(_args.init_policy)}, centres remapped "
          f"(max mean-shift {float(np.abs((mo-mn)/sn).max()):.3f} sigma)")

policy, policy_meta = build_policy(
    "rbf", STATE_DIM, INPUT_DIM, u_max=U_MAX, dtype=dtype, device=device,
    rng=np.random.default_rng(0), num_basis=NUM_BASIS,
    centers_init=centers_init, lengthscales_init=lengthscales_init,
    s_lo=s_lo.tolist(), s_hi=s_hi.tolist(), center_range_pad=CENTER_RANGE_PAD)
if _warm is not None:
    policy.load_state_dict(_warm["policy_state_dict"])

print(f"\npolicy rbf: in={STATE_DIM} out={INPUT_DIM} u_max={U_MAX} "
      f"params={policy_meta['n_params']}")
from policy_learning.wasserstein_loss import TRACE_NORMALIZE, SMOOTH_EPS
print(f"L2: lambda={LAMBDA_A} on ALL SIX channels (additive)")
print(f"W2 smoothing: eps={SMOOTH_EPS}"
      + ("  -- sqrt(W2^2 + eps^2), derivative bounded by 1/(2*eps)="
         f"{1/(2*SMOOTH_EPS):.1f}" if SMOOTH_EPS > 0 else "  -- OFF, derivative "
         "diverges as the cost approaches 0"))
print(f"W2 trace-normalisation: {'ON' if TRACE_NORMALIZE else 'OFF'}"
      + ("  -- both spectra divided by their own trace, so only SHAPE is compared. "
         "W2 values are NOT comparable to runs made before this change."
         if TRACE_NORMALIZE else
         "  -- WARNING: the raw comparison was measured to give W2 == 0 for every "
         "channel across a in [-0.5, +0.5]"))

# E_{a|s} is only real if replicas of one state draw DIFFERENT actions
with torch.no_grad():
    _sp = policy(states=POOL[:1].expand(K_ACTIONS, -1).contiguous(),
                 t=0, p_dropout=P_DROPOUT).std(0).mean().item()
print(f"action spread across {K_ACTIONS} replicas of one state: {_sp:.3e}"
      f"{'   <-- WARNING: E_a|s degenerate' if _sp < 1e-4 else '   (ok)'}")

# ------------------------------------------- constraint bounds, in z units ------
_amin = 2.0 * (pdata.MIN_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
_amax = 2.0 * (pdata.MAX_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
CC_LO = torch.tensor((_amin - stats["std_act_mu"]) / stats["std_act_sd"],
                     dtype=dtype, device=device)
CC_HI = torch.tensor((_amax - stats["std_act_mu"]) / stats["std_act_sd"],
                     dtype=dtype, device=device)
WT_IDX = pdata.OBS_NAMES.index("Wt")
_olo, _ohi = pdata.MIN_OBS[WT_IDX], pdata.MAX_OBS[WT_IDX]
CC_WT_MIN_Z = float(((2.0 * (CC_WT_MIN_PHYS - _olo) / (_ohi - _olo) - 1.0)
                     - stats["std_obs_mu"][WT_IDX]) / stats["std_obs_sd"][WT_IDX])
print(f"\nchance constraints: eps={CC_EPS} -> Phi^-1={phi_inv(CC_EPS):.4f}")
print(f"  action box  alpha={CC_ALPHA_ACT}")
print(f"  vessel floor alpha={CC_ALPHA_STATE}  Wt >= {CC_WT_MIN_PHYS:.0f} phys "
      f"({CC_WT_MIN_Z:.3f} z)")
_wt_tr = POOL[:, WT_IDX].numpy()
print(f"  training data below the floor: {100*float((_wt_tr < CC_WT_MIN_Z).mean()):.1f}%")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


# ================================================================== window loss ==
_acc = {"var": None, "t0": 0}
_acc_a = []
_eig = None
_log = {"w2": [], "l2": [], "cc_a": [], "cc_s": []}


def window_loss(t, s, a, mu, cov, s_next):
    """Accumulate the 5 steps into one 1-hour transition, then score it.

    The GP step is 0.2 h and the expert's is 1.0 h, so five per-step variances are
    summed (first order, treating the per-step noise as independent) before the
    comparison -- otherwise a 0.2 h prediction would be matched against a 1.0 h one
    and the expert's drift would look ~5x larger purely from the time span.
    """
    global _acc
    if _acc["var"] is None:
        _acc = {"var": torch.zeros_like(cov), "t0": t}
    _acc["var"] = _acc["var"] + cov
    _acc_a.append(a)

    if (t - _acc["t0"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=cov.dtype, device=cov.device)

    var_1h = _acc["var"]
    _acc = {"var": None, "t0": 0}

    d = w2_cross_dim_torch(var_1h, _eig)                       # (P,)
    w2 = d.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()      # E_a|s then E_s
    _log["w2"].append(float(w2.detach()))

    a_all = torch.cat(_acc_a, 0)
    n_rep = a_all.shape[0] // (NUM_STATES * K_ACTIONS)

    l2 = (a_all ** 2).sum(dim=1).mean()
    _log["l2"].append(float(l2.detach()))

    cc_a = action_chance_penalty(a_all, CC_LO, CC_HI,
                                 num_states=NUM_STATES * n_rep, k_actions=K_ACTIONS,
                                 eps=CC_EPS, alpha=CC_ALPHA_ACT)
    cc_s = state_chance_penalty(mu, cov, WT_IDX, lo=CC_WT_MIN_Z, hi=None,
                                num_states=NUM_STATES, k_actions=K_ACTIONS,
                                eps=CC_EPS, alpha=CC_ALPHA_STATE)
    _log["cc_a"].append(float(cc_a.detach()))
    _log["cc_s"].append(float(cc_s.detach()))
    _acc_a.clear()

    return w2 + LAMBDA_A * l2 + cc_a + cc_s


# ==================================================================== training ===
hist, l2_first = [], None
for it in range(N_ITERS):
    order = rng.permutation(len(EXPERT_TIMES))[:WINDOWS_PER_ITER]
    L, G, S = [], [], []
    for k in _log:
        _log[k].clear()

    for idx in order:
        t_h = float(EXPERT_TIMES[idx])
        _eig = EXPERT_EIGS[round(t_h, 6)]
        ph = phase_of(t_h)
        st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=rng,
                                      dtype=dtype, device=device)
        s0 = st.repeat_interleave(K_ACTIONS, dim=0)
        _acc_a.clear()

        out = gp_rollout(model=MODELS[ph], policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                         p_dropout=P_DROPOUT, particle_pred=True,
                         loss_fn=window_loss, graph_mode="full")
        loss = out["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                            if p.grad is not None)).item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
        optimizer.step()
        L.append(loss.item()); G.append(gn)
        S.append(out["S"].detach().abs().max().item())

    L, G, S = map(np.array, (L, G, S))
    w2m = float(np.mean(_log["w2"])); l2m = float(np.mean(_log["l2"]))
    if l2_first is None:
        l2_first = l2m
    hist.append(w2m)
    print(f"iter {it:3d}  W2={w2m:.5f}  ||a||^2={l2m:.5f}  "
          f"lam*||a||^2={LAMBDA_A*l2m:.3e}  loss={L.mean():.5f}", flush=True)
    print(f"          cc_act={np.mean(_log['cc_a']):.3e} "
          f"cc_state={np.mean(_log['cc_s']):.3e}   "
          f"|grad| med={np.median(G):.3e} DEAD={int((G<1e-12).sum())}/{len(G)}  "
          f"|s|max={np.median(S):.1f} (data {s_hi.max():.1f})", flush=True)

print(f"\n||a||^2  first iter {l2_first:.5f}  ->  last {l2m:.5f}  "
      f"({100*(l2m-l2_first)/max(l2_first,1e-12):+.1f}%)")
print("  (with lambda>0 this should DECREASE; if it rises, the L2 term is not binding)")

print("\naction spread ACROSS WINDOWS (the quantity that collapsed):")
with torch.no_grad():
    _acts = []
    for _t in (10.0, 50.0, 90.0, 130.0):
        _acts.append(policy(states=POOLS[phase_of(_t)][:64], t=0,
                            p_dropout=0.0).mean(0))
    _A = torch.stack(_acts)
print(f"  {'channel':14s}" + "".join(f"{f't={t:.0f}h':>10s}"
                                     for t in (10, 50, 90, 130)) + f"{'std':>10s}")
_SDA = np.asarray(stats["std_act_sd"])
_SPA = pdata.MAX_ACT - pdata.MIN_ACT
_GPEI = np.array([858.26, 24.98, 5.10, 8.90, 0.145, 151.21])
for _i, _nm in enumerate(pdata.ACT_NAMES):
    _sd = _A[:, _i].std().item()
    print(f"  {_nm:14s}" + "".join(f"{_A[j, _i].item():10.4f}" for j in range(4))
          + f"{_sd:10.4f}   ({100*_sd*_SDA[_i]*_SPA[_i]/2/_GPEI[_i]:.2f}% of gpei)")

torch.save({"policy_state_dict": policy.state_dict(), "policy_meta": policy_meta,
            "policy_kind": "rbf", "hist": hist, "lam": LAMBDA_A,
            "l2_first": l2_first, "l2_last": l2m,
            "phase_prefix": _args.phase_prefix,
            "std_obs_mu": stats["std_obs_mu"].tolist(),
            "std_obs_sd": stats["std_obs_sd"].tolist(),
            "std_act_mu": stats["std_act_mu"].tolist(),
            "std_act_sd": stats["std_act_sd"].tolist()}, OUT)
print(f"\nsaved -> {OUT}")
