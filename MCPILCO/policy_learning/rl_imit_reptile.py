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

SAVE_DIR = os.path.join(_REPO, "results_rlrep")
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

NUM_STATES, K_ACTIONS = 100, 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

T_START_HOURS, HOURS_PER_STEP, EXPERT_DT = 0.0, 0.2, 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0
BATCH_T_MAX = 230.0
WINDOWS_PER_ITER = 150
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0

NUM_BASIS, U_MAX = 200, 3.0
CENTER_RANGE_PAD = 1.10

INNER_K = 15
META_EPS = 0.5
TASKS_PER_META = 15
SCALE_LR_WITH_K = True
K_REF = 5

ETA = 0.35
ALPHA_W2 = 15.0
KAPPA = 1.0
LAMBDA_A = 0.2

CC_EPS = 0.95
CC_ALPHA_ACT = 1000.0
CC_ALPHA_STATE = 1.0
CC_WT_MIN_PHYS = 50000.0

EXPERT_COV_KEY = "cov_n"

_ap = argparse.ArgumentParser("CDIL policy optimization (clean)")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt, selected per "
                      "window by the expert time")
_ap.add_argument("-reward_model", required=True, help="reward GP checkpoint")
_ap.add_argument("-eta", type=float, default=None)
_ap.add_argument("-alpha_w2", type=float, default=None)
_ap.add_argument("-kappa", type=float, default=None)
_ap.add_argument("-lam", type=float, default=None, help="L2 weight on ||a||^2")
_ap.add_argument("-inner_k", type=int, default=None, help="inner steps per task")
_ap.add_argument("-meta_eps", type=float, default=None, help="outer step size")
_ap.add_argument("-tasks", type=int, default=None,
                 help="windows per meta-iteration, split evenly across the 3 phases")
_ap.add_argument("-lam_growth", type=float, default=None,
                 help="penalty weight on PHASE-0 windows only")
_ap.add_argument("-phase", type=int, default=-1, choices=[-1, 0, 1, 2])
_ap.add_argument("-fix_discharge", type=float, default=None,
                 help="pin discharge at this physical value in the growth phase")
_ap.add_argument("-iters", type=int, default=None)
_ap.add_argument("-out", default=None)
_ap.add_argument("-init_policy", default=None, help="warm start")
_args = _ap.parse_known_args()[0]
if _args.lam is not None:       LAMBDA_A = _args.lam
PHASE = _args.phase
if _args.inner_k:  INNER_K = _args.inner_k
if _args.meta_eps: META_EPS = _args.meta_eps
if _args.tasks:    TASKS_PER_META = _args.tasks
LAM_GROWTH = 0.2 if _args.lam_growth is None else _args.lam_growth
if _args.eta is not None:       ETA = _args.eta
if _args.alpha_w2 is not None:  ALPHA_W2 = _args.alpha_w2
if _args.kappa is not None:     KAPPA = _args.kappa
if _args.iters:
    N_ITERS = _args.iters
OUT = _args.out or os.path.join(SAVE_DIR,
        f"rlrep_k{INNER_K}_t{TASKS_PER_META}_eta{str(ETA).replace('.','p')}.pt")


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
WINDOW_TIMES = np.arange(EXPERT_T_MIN, BATCH_T_MAX + 1e-9, EXPERT_DT)
print(f"pre-caching {len(EXPERT_TIMES)} expert distributions "
      f"(1..{EXPERT_T_MAX:.0f} h) ...", flush=True)
EXPERT_EIGS = {}
for _t in EXPERT_TIMES:
    _d = _q.next_state_distribution(t=float(_t), source="traj")
    EXPERT_EIGS[round(float(_t), 6)] = torch.linalg.eigvalsh(
        torch.tensor(np.asarray(_d[EXPERT_COV_KEY]).tolist(), dtype=dtype,
                     device=device))
print(f"  eigenvalues @75h: {EXPERT_EIGS[75.0].numpy()}", flush=True)
if PHASE >= 0:
    _lo, _hi = pdata.PHASES[PHASE]
    _k = (WINDOW_TIMES >= _lo) & (WINDOW_TIMES < _hi)
    WINDOW_TIMES = WINDOW_TIMES[_k]
    print(f"PHASE {PHASE}: t in [{_lo:g}, {'inf' if _hi > 1e8 else f'{_hi:g}'}) h "
          f"-> {len(WINDOW_TIMES)} windows")
    if WINDOWS_PER_ITER > len(WINDOW_TIMES):
        WINDOWS_PER_ITER = len(WINDOW_TIMES)

_cnt = {}
for _t in WINDOW_TIMES:
    _cnt[phase_of(float(_t))] = _cnt.get(phase_of(float(_t)), 0) + 1
print("windows per model: " + "  ".join(f"phase {k}: {v}" for k, v in sorted(_cnt.items())))
_n_imit = int((WINDOW_TIMES <= EXPERT_T_MAX).sum())
print(f"windows: {len(WINDOW_TIMES)} total over 1..{BATCH_T_MAX:.0f} h  ->  "
      f"{_n_imit} with the W2 constraint (t<={EXPERT_T_MAX:.0f}), "
      f"{len(WINDOW_TIMES)-_n_imit} reward-only")


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
print(f"L2: lambda={LAMBDA_A} on ALL SIX channels (additive)")

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

_rck = torch.load(_args.reward_model, map_location=device, weights_only=False)
_rinit = dict(active_dims=np.arange(0, GP_INPUT_DIM),
              lengthscales_init=np.ones(GP_INPUT_DIM), flg_train_lengthscales=True,
              lambda_init=np.ones(1), flg_train_lambda=True,
              sigma_n_init=1e-2 * np.ones(1), sigma_n_num=1e-4,
              flg_train_sigma_n=True, dtype=dtype, device=device)
RMODEL = ML.Model_learning_RBF(num_gp=1, init_dict_list=[dict(_rinit)],
                               approximation_mode=None, dtype=dtype, device=device,
                               flg_norm=False)
RMODEL.load_state_dict(_rck["state_dict"])
for _k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
           "K_X_inv_list", "gp_inputs_tr_list"):
    setattr(RMODEL, _k, _rck[_k])
RMODEL.num_samples = _rck["gp_inputs"].shape[0]
RMODEL.dim_state, RMODEL.dim_input = STATE_DIM, INPUT_DIM
RMODEL.norm_list = [1.0]
RMODEL.set_eval_mode()
R_MU, R_SD = float(_rck["reward_mu"]), float(_rck["reward_sd"])

print(f"\nobjective: -LCB_reward + {ALPHA_W2}*relu(W2_h - {ETA}) "
      f"+ {LAMBDA_A}*||a||^2 + chance")
print(f"  reward GP : {os.path.basename(_args.reward_model)}  "
      f"held-out R^2={_rck.get('held_out_r2'):.4f}  "
      f"yield/step mu={R_MU:.4f} sd={R_SD:.4f}")
print(f"  LCB       : mu_r - {KAPPA}*sigma_r  (not the mean)")
print(f"  eta       : {ETA}  -- measured, the converged W2 of imitation-only policies")
print(f"  alpha_W2  : {ALPHA_W2}  -- measured, puts the penalty on the reward's scale")

_lo_s = 2.0 * (pdata.MIN_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
_hi_s = 2.0 * (pdata.MAX_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
A_LO_Z = torch.tensor((_lo_s - stats["std_act_mu"]) / stats["std_act_sd"],
                      dtype=dtype, device=device)
A_SPAN_Z = torch.tensor((_hi_s - _lo_s) / stats["std_act_sd"],
                        dtype=dtype, device=device)
print("\npenalty targets (physical -> z):")
for _i, _nm in enumerate(pdata.ACT_NAMES):
    _mean_phys = ((stats["std_act_mu"][_i] + 1) / 2
                  * (pdata.MAX_ACT[_i] - pdata.MIN_ACT[_i]) + pdata.MIN_ACT[_i])
    print(f"    {_nm:14s} lo={pdata.MIN_ACT[_i]:8.1f} phys -> {A_LO_Z[_i].item():7.3f} z"
          f"   (z=0 is {_mean_phys:8.1f} phys, the dataset mean)")

_CUR = {"lam": 0.0, "phase": 0}

FIX_DISCHARGE = _args.fix_discharge
if FIX_DISCHARGE is not None:
    _fs = (2.0 * (FIX_DISCHARGE - pdata.MIN_ACT[0])
           / (pdata.MAX_ACT[0] - pdata.MIN_ACT[0]) - 1.0)
    _FZ = float((_fs - stats["std_act_mu"][0]) / stats["std_act_sd"][0])

    class _Pin(torch.nn.Module):
        """Overwrites discharge with a constant, so that head receives no gradient.
        gpei holds discharge at exactly 0 for the first ~100 h, and during growth the
        vessel should be filling rather than draining."""

        def __init__(self, base, z):
            super().__init__()
            self.base, self.z = base, z
            self.state_dim, self.input_dim = base.state_dim, base.input_dim

        def forward(self, states, t=None, p_dropout=0.0):
            a = self.base(states=states, t=t, p_dropout=p_dropout)
            cols = [a[:, j] for j in range(a.shape[1])]
            if _CUR["phase"] != 0:
                return a
            cols[0] = torch.full_like(cols[0], self.z)
            return torch.stack(cols, dim=1)

    policy = _Pin(policy, _FZ)
    print(f"discharge PINNED at {FIX_DISCHARGE:.1f} phys ({_FZ:.3f} z)")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


_acc = {"var": None, "t0": 0}
_acc_a, _acc_s = [], []
_eig = None
_log = {"w2": [], "l2": [], "cc_a": [], "cc_s": [], "r": [], "viol": [], "pen": [],
        "r_imit": [], "r_late": [], "n_reward_only": [0]}


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
    _acc_s.append(s)

    if (t - _acc["t0"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=cov.dtype, device=cov.device)

    var_1h = _acc["var"]
    _acc = {"var": None, "t0": 0}

    if _eig is not None:
        d = w2_cross_dim_torch(var_1h, _eig)
        w2 = d.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()
        _log["w2"].append(float(w2.detach()))
    else:
        w2 = None

    a_r, s_r = torch.cat(_acc_a, 0), torch.cat(_acc_s, 0)
    _m, _v = RMODEL.get_gp_estimate(gp_inputs=torch.cat([s_r, a_r], dim=1),
                                    gp_index_list=[0])
    reward = (_m[0].reshape(-1)
              - KAPPA * torch.sqrt(_v[0].reshape(-1).clamp_min(1e-12))).mean()
    _log["r"].append(float(reward.detach()))
    (_log["r_imit"] if _eig is not None else _log["r_late"]).append(
        float(reward.detach()))

    if w2 is not None:
        viol = torch.relu(w2 - ETA)
        pen = ALPHA_W2 * viol
        _log["viol"].append(float(viol.detach()))
        _log["pen"].append(float(pen.detach()))
    else:
        pen = torch.zeros((), dtype=reward.dtype, device=reward.device)
        _log["n_reward_only"][0] += 1

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
    _acc_a.clear(); _acc_s.clear()

    return -reward + pen + _CUR["lam"] * l2 + cc_a + cc_s


PHASE_WINDOWS = {p: np.array([t for t in WINDOW_TIMES if phase_of(float(t)) == p])
                 for p in (0, 1, 2)}
print("\nstratified task pool:")
for _p in (0, 1, 2):
    _lo, _hi = pdata.PHASES[_p]
    print(f"  phase {_p}: {len(PHASE_WINDOWS[_p]):3d} windows in "
          f"[{_lo:g}, {'inf' if _hi > 1e8 else f'{_hi:g}'}) h")
_ACTIVE = [p for p in (0, 1, 2) if len(PHASE_WINDOWS[p]) > 0]
_PER = max(1, TASKS_PER_META // len(_ACTIVE))
print(f"  -> {_PER} tasks from EACH of {len(_ACTIVE)} phases per meta-iteration "
      f"({_PER * len(_ACTIVE)} total)")
print(f"     phase 1 holds only {len(PHASE_WINDOWS[1])} of {len(WINDOW_TIMES)} "
      f"windows, so a uniform draw of {TASKS_PER_META} would often take NONE from it "
      f"and the outer average would be dominated by whichever phases were drawn")

INNER_LR = (LR * K_REF / INNER_K) if SCALE_LR_WITH_K else LR
print(f"\nReptile: k={INNER_K}  eps={META_EPS}  alpha={INNER_LR:.5f}  "
      f"(alpha*k={INNER_LR*INNER_K:.4f} held fixed)  "
      f"agreement coeff={0.5*INNER_K*(INNER_K-1)*INNER_LR:.5f}")
if INNER_K == 1:
    print("  WARNING: k=1 IS joint training -- the gradient-agreement term is exactly 0")
print(f"lambda: {LAM_GROWTH} on PHASE-0 windows, 0.0 on phases 1-2 (the sign of "
      f"gpei-vs-mean reverses at t=20 h)")


def _run_task(t_h, gen):
    """One forward+backward on a single window, with the growth-phase treatment
    applied only if that window is in phase 0."""
    global _eig
    _eig = EXPERT_EIGS.get(round(t_h, 6))
    ph = phase_of(t_h)
    _CUR["phase"] = ph
    _CUR["lam"] = LAM_GROWTH if ph == 0 else 0.0
    st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=gen,
                                  dtype=dtype, device=device)
    s0 = st.repeat_interleave(K_ACTIONS, dim=0)
    _acc_a.clear(); _acc_s.clear()
    out = gp_rollout(model=MODELS[ph], policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                     p_dropout=P_DROPOUT, particle_pred=True, loss_fn=window_loss,
                     graph_mode="full")
    return out["loss_total"], out


hist = []
for it in range(N_ITERS):
    theta0 = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    phis, L, G, S = [], [], [], []
    for k in _log:
        if k == "n_reward_only":
            _log[k][0] = 0
        else:
            _log[k].clear()

    tasks = []
    for _p in _ACTIVE:
        _pool = PHASE_WINDOWS[_p]
        _pick = rng.choice(len(_pool), size=min(_PER, len(_pool)), replace=False)
        tasks += [float(_pool[i]) for i in _pick]

    for t_h in tasks:
        policy.load_state_dict(theta0)
        inner = torch.optim.Adam(policy.parameters(), lr=INNER_LR)
        gen = np.random.default_rng(int(t_h * 100) + 10000 * it)
        for _ in range(INNER_K):
            loss, out = _run_task(t_h, gen)
            inner.zero_grad(); loss.backward()
            gn = torch.sqrt(sum((q.grad ** 2).sum() for q in policy.parameters()
                                if q.grad is not None)).item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
            inner.step()
            L.append(loss.item()); G.append(gn)
            S.append(out["S"].detach().abs().max().item())
        phis.append({k: v.detach().clone() for k, v in policy.state_dict().items()})

    new_state, drift = {}, 0.0
    for k in theta0:
        if theta0[k].dtype.is_floating_point:
            mphi = torch.stack([p[k] for p in phis]).mean(dim=0)
            new_state[k] = theta0[k] + META_EPS * (mphi - theta0[k])
            drift += float((mphi - theta0[k]).norm() ** 2)
        else:
            new_state[k] = theta0[k]
    policy.load_state_dict(new_state)

    L, G, S = map(np.array, (L, G, S))
    w2a = np.array(_log["w2"]) if _log["w2"] else np.array([np.nan])
    rm = float(np.mean(_log["r"])) if _log["r"] else float("nan")
    pm = float(np.mean(_log["pen"])) if _log["pen"] else 0.0
    l2m = float(np.mean(_log["l2"])) if _log["l2"] else 0.0
    hist.append(float(np.nanmean(w2a)))
    print(f"meta {it:3d}  loss={L.mean():.5f}  reward(LCB)={rm:.5f} "
          f"(phys {rm*R_SD+R_MU:.3f})  W2 mean={np.nanmean(w2a):.5f} "
          f"max={np.nanmax(w2a):.5f}  |phi-theta|={np.sqrt(drift):.3e}", flush=True)
    print(f"          eta={ETA} violating={int((w2a > ETA).sum())}/{len(w2a)}  "
          f"penalty={pm:.5f} (pen/|rew|={pm/max(abs(rm),1e-9):.2f}x)  "
          f"dist-from-min={l2m:.5f}  {len(tasks)} tasks x {INNER_K} steps  "
          f"|grad| med={np.median(G):.3e} DEAD={int((G < 1e-12).sum())}/{len(G)}",
          flush=True)

print("\naction by phase (the quantity that collapsed):")
_SDA = np.asarray(stats["std_act_sd"]); _SPA = pdata.MAX_ACT - pdata.MIN_ACT
_GPEI = np.array([858.26, 24.98, 5.10, 8.90, 0.145, 151.21])
with torch.no_grad():
    _AA = []
    for _p in _ACTIVE:
        _CUR["phase"] = _p
        _AA.append(policy(states=POOLS[_p][:64], t=0, p_dropout=0.0).mean(0))
    _AA = torch.stack(_AA)
print(f"  {'channel':14s}" + "".join(f"{f'phase {p}':>10s}" for p in _ACTIVE)
      + f"{'std':>9}{'% gpei':>9}")
for _i, _nm in enumerate(pdata.ACT_NAMES):
    _sd = _AA[:, _i].std().item()
    print(f"  {_nm:14s}" + "".join(f"{_AA[j, _i].item():10.4f}"
                                   for j in range(len(_ACTIVE)))
          + f"{_sd:9.4f}{100*_sd*_SDA[_i]*_SPA[_i]/2/_GPEI[_i]:8.2f}%")

torch.save({"policy_state_dict": policy.state_dict(), "policy_meta": policy_meta,
            "policy_kind": "rbf", "hist": hist, "lam": LAMBDA_A,
            "phase_prefix": _args.phase_prefix,
            "std_obs_mu": stats["std_obs_mu"].tolist(),
            "std_obs_sd": stats["std_obs_sd"].tolist(),
            "std_act_mu": stats["std_act_mu"].tolist(),
            "std_act_sd": stats["std_act_sd"].tolist()}, OUT)
print(f"\nsaved -> {OUT}")
