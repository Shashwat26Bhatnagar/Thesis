#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

SAVE_DIR = os.path.join(_REPO, "results_phase")
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

NUM_STATES, K_ACTIONS = 100, 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

T_START_HOURS, HOURS_PER_STEP, EXPERT_DT = 0.0, 0.2, 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0
WINDOWS_PER_ITER = 150
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0

NUM_BASIS, U_MAX = 200, 3.0
CENTER_RANGE_PAD = 1.10

LAMBDA_A = 0.0

CC_EPS = 0.95
CC_ALPHA_ACT = 1000.0
CC_ALPHA_STATE = 1.0
CC_WT_MIN_PHYS = 50000.0

EXPERT_COV_KEY = "cov_n"

_ap = argparse.ArgumentParser("CDIL policy optimization (clean)")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt, selected per "
                      "window by the expert time")
_ap.add_argument("-fix_discharge", type=float, default=None,
                 help="hold discharge at this PHYSICAL value (e.g. 0) and optimise "
                      "the other five channels only")
_ap.add_argument("-phase", type=int, default=-1, choices=[-1, 0, 1, 2],
                 help="train on this phase's windows ONLY (-1 = all, the old "
                      "behaviour)")
_ap.add_argument("-lam", type=float, default=0.0, help="L2 weight on ||a||^2")
_ap.add_argument("-iters", type=int, default=None)
_ap.add_argument("-out", default=None)
_ap.add_argument("-init_policy", default=None, help="warm start")
_args = _ap.parse_known_args()[0]
LAMBDA_A = _args.lam
if _args.iters:
    N_ITERS = _args.iters
PHASE = _args.phase
OUT = _args.out or os.path.join(SAVE_DIR,
        f"ph{PHASE}_lam{str(LAMBDA_A).replace('.','p')}.pt")


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
if PHASE >= 0:
    _lo, _hi = pdata.PHASES[PHASE]
    _keep = (EXPERT_TIMES >= _lo) & (EXPERT_TIMES < _hi)
    if _keep.sum() < 5:
        raise SystemExit(f"phase {PHASE} covers only {int(_keep.sum())} expert hours "
                         f"in [{_lo}, {_hi}) -- too few to train on")
    EXPERT_TIMES = EXPERT_TIMES[_keep]
    print(f"PHASE {PHASE}: training on t in [{_lo:g}, "
          f"{'inf' if _hi > 1e8 else f'{_hi:g}'}) h  -> {len(EXPERT_TIMES)} windows "
          f"of the original 150")
    if WINDOWS_PER_ITER > len(EXPERT_TIMES):
        WINDOWS_PER_ITER = len(EXPERT_TIMES)
        print(f"  windows/iter reduced to {WINDOWS_PER_ITER} (all of them)")
else:
    print("PHASE -1: all 150 windows, one policy for the whole batch")

_cnt = {}
for _t in EXPERT_TIMES:
    _cnt[phase_of(float(_t))] = _cnt.get(phase_of(float(_t)), 0) + 1
print("windows per model: " + "  ".join(f"phase {k}: {v}" for k, v in sorted(_cnt.items())))


_warm = None
centers_init = lengthscales_init = None
if _args.init_policy and os.path.exists(_args.init_policy):
    _warm = torch.load(_args.init_policy, map_location=device, weights_only=False)
    _m = _warm["policy_meta"]
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
from policy_learning.wasserstein_loss import TRACE_NORMALIZE
print(f"L2: lambda={LAMBDA_A} on ALL SIX channels (additive)")
print(f"W2 trace-normalisation: {'ON' if TRACE_NORMALIZE else 'OFF'}"
      + ("  -- both spectra divided by their own trace, so only SHAPE is compared. "
         "W2 values are NOT comparable to runs made before this change."
         if TRACE_NORMALIZE else
         "  -- WARNING: the raw comparison was measured to give W2 == 0 for every "
         "channel across a in [-0.5, +0.5]"))

with torch.no_grad():
    _sp = policy(states=POOL[:1].expand(K_ACTIONS, -1).contiguous(),
                 t=0, p_dropout=P_DROPOUT).std(0).mean().item()
print(f"action spread across {K_ACTIONS} replicas of one state: {_sp:.3e}"
      f"{'   <-- WARNING: E_a|s degenerate' if _sp < 1e-4 else '   (ok)'}")

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

FIX_DISCHARGE = _args.fix_discharge
if FIX_DISCHARGE is not None:
    _lo_d, _hi_d = pdata.MIN_ACT[0], pdata.MAX_ACT[0]
    _smpl = 2.0 * (FIX_DISCHARGE - _lo_d) / (_hi_d - _lo_d) - 1.0
    FIX_Z = float((_smpl - stats["std_act_mu"][0]) / stats["std_act_sd"][0])

    class _PinDischarge(torch.nn.Module):
        """Overwrites channel 0 with a constant. The head still emits it, but the
        value is replaced, so that head receives no gradient and simply stays at its
        initial value."""

        def __init__(self, base, z):
            super().__init__()
            self.base, self.z = base, z
            self.state_dim, self.input_dim = base.state_dim, base.input_dim

        def forward(self, states, t=None, p_dropout=0.0):
            a = self.base(states=states, t=t, p_dropout=p_dropout)
            cols = [a[:, j] for j in range(a.shape[1])]
            cols[0] = torch.full_like(cols[0], self.z)
            return torch.stack(cols, dim=1)

    policy = _PinDischarge(policy, FIX_Z)
    print(f"\ndischarge PINNED at {FIX_DISCHARGE:.1f} physical ({FIX_Z:.3f} z); the "
          f"policy optimises the other five channels only")
    print(f"  (gpei holds discharge at exactly 0 for the first ~100 h)")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


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

    d = w2_cross_dim_torch(var_1h, _eig)
    w2 = d.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()
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
    _probe = ([10.0, 50.0, 90.0, 130.0] if PHASE < 0 else
              list(np.linspace(EXPERT_TIMES[0], EXPERT_TIMES[-1], 4)))
    for _t in _probe:
        _acts.append(policy(states=POOLS[phase_of(_t)][:64], t=0,
                            p_dropout=0.0).mean(0))
    _A = torch.stack(_acts)
print(f"  {'channel':14s}" + "".join(f"{f't={t:.0f}h':>10s}"
                                     for t in _probe) + f"{'std':>10s}")
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
