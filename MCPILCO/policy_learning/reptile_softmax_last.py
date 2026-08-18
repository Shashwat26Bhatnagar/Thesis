#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/reptile_softmax_last.py

CDIL policy optimization: Reptile meta-updates + softmax-weighted state aggregation
+ L1-on-discharge / L2-on-the-rest.

*** RECONSTRUCTED FILE -- READ THIS BEFORE RUNNING ***
This version was built by combining reptile_policy_opt.py (the Reptile meta-loop,
already carrying the l2_first save-crash fix) with the softmax state-weighting from
cdil_policy_softmax.py, and restoring the L1-on-discharge split that was present in
exp_policy_l1.py but MISSING from the softmax file (confirmed by reading its source
directly: no LAMBDA_L1, DISCHARGE_IDX, A_OFF_Z, or L2_MASK anywhere in it -- discharge
was under plain L2 across all six channels, which reproduces the original bug this
whole thread started from, independent of k/alpha/beta). If your actual
reptile_softmax_last.py has custom pieces beyond what was shown in this conversation,
diff this against it before overwriting -- this is a fresh build, not a patch of an
unseen file.

THREE FIXES IN THIS VERSION, ON TOP OF WHAT WAS RUNNING AS rsl_k15:

1. l2_first NameError (the crash). hist, l2_first = [], None before the loop;
   l2_first = l2m set once, on the first iteration. The previous run completed all
   20 meta-iterations and crashed on the final torch.save -- all that compute was
   lost because there was no earlier checkpoint to fall back on. Fixed by #3 below.

2. L1-on-discharge restored. Plain L2 on a valve-like channel is analytically
   "hands-full" (Nagahara et al. 2016) -- nonzero almost everywhere -- which is
   exactly the flat ~255, std~1.6, 100%-nonzero result seen in rep_k15_batch_0.csv.
   L1 (measured from the CLOSED level, not from z=0, which is the dataset mean) is
   the "hands-off"/bang-off-bang-compatible penalty. Restored verbatim from
   exp_policy_l1.py: LAMBDA_L1, DISCHARGE_IDX, A_OFF_Z, L2_MASK, and the two
   penalty terms in window_loss.

3. Periodic checkpointing. torch.save now also runs every SAVE_EVERY meta-iterations
   (default 5), not just at the very end -- so a late crash costs at most SAVE_EVERY
   iterations of compute, not the whole run.

SOFTMAX-WEIGHTED STATE AGGREGATION (from cdil_policy_softmax.py, unchanged in spirit)
    Under a plain mean over the 100 sampled states, every state's gradient weight is
    exactly 1/100 regardless of how far its induced covariance is from the expert's.
    The softmax (BETA, sharpness) upweights states whose distance dv is currently
    larger -- detached, so it reweights the gradient without giving the policy a path
    to lower the loss by reshaping the weights instead of the underlying distances.
    BETA=0 recovers the plain mean exactly.

    NOTE (flagged, not fixed here): this reweights across the 100 STATES only. The
    K_ACTIONS=5 action-replicas per state are still combined with a plain, uniform
    mean before the softmax ever sees them -- so this does nothing for within-state
    action diversity, and the standing "E_a|s degenerate" warning below is checked
    every run for exactly that reason. Also: Bures/covariance distance is
    translation-invariant in the mean at ANY weighting, so this reweighting -- by
    construction -- still cannot put gradient on the discharge action's MEAN. That
    is what the L1 term (fix #2) is for; the softmax term is about covariance SHAPE,
    not location.

REPTILE META-UPDATES
    for each meta-iteration:
        theta_0 = theta
        for each of TASKS_PER_META sampled windows, INDEPENDENTLY from theta_0:
            phi = theta_0
            for k steps:  phi <- phi - alpha * grad L_window(phi)
            record phi
        theta <- theta_0 + eps * mean_over_windows(phi - theta_0)

    At k=1 the within-task-agreement term vanishes and this is exactly joint
    training on the mixture (Nichol et al. 2018, Sec 5.1) -- the collapse mode this
    whole line of experiments is trying to escape. alpha and k are NOT independent:
    the paper's own Taylor expansion "only holds for small alpha*k". With
    SCALE_LR_WITH_K on (default), alpha = LR * K_REF / k, holding alpha*k = LR*K_REF
    fixed as k varies -- sweep -lr to move that product deliberately, sweep -inner_k
    at a fixed -lr to raise k without it drifting.

    python policy_learning/reptile_softmax_last.py \\
        -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \\
        -lam 0.01 -inner_k 15 -lr 0.02 -beta 5 -iters 20 -out results_rsl/rsl_k15.pt
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

SAVE_DIR = os.path.join(_REPO, "results_rsl")
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
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0
SAVE_EVERY = 5            # checkpoint every N meta-iterations, not just at the end

# --- Reptile ---
INNER_K = 5              # inner steps per task; k=1 reduces Reptile to joint training
SCALE_LR_WITH_K = True   # hold alpha*k fixed -- see module docstring
K_REF = 5                # the k at which alpha == LR
META_EPS = 0.5           # outer step size: theta <- theta + eps*(mean phi - theta)
TASKS_PER_META = 10      # windows per meta-iteration, each run INDEPENDENTLY from theta
USE_TIME_INPUT = False   # append normalised batch time as a 9th policy input

# --- softmax aggregation over the 100 states ---
BETA = 5.0                # sharpness; 0.0 recovers the plain mean exactly

# --- policy ---
NUM_BASIS, U_MAX = 200, 3.0
CENTER_RANGE_PAD = 1.10

# --- L2 (channels 1..5) / L1 (discharge) -- see fix #2 in the module docstring ---
LAMBDA_A = 0.01                      # L2 weight, channels 1..5; set with -lam
LAMBDA_L1 = 0.05                     # L1 weight on discharge; set with -lam_l1
DISCHARGE_IDX = 0
DISCHARGE_OFF_PHYS = 0.0             # "closed"

# --- chance constraints (Tan et al. Eq. 8-9) ---
CC_EPS = 0.95
CC_ALPHA_ACT = 1000.0
CC_ALPHA_STATE = 1.0
CC_WT_MIN_PHYS = 50000.0

EXPERT_COV_KEY = "cov_n"

_ap = argparse.ArgumentParser("Reptile + softmax-weighted CDIL policy optimization")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt, selected per "
                      "window by the expert time")
_ap.add_argument("-lam", type=float, default=None, help="L2 weight, channels 1..5")
_ap.add_argument("-lam_l1", type=float, default=None,
                 help="L1 weight on discharge, measured from the closed level")
_ap.add_argument("-beta", type=float, default=None,
                 help="softmax sharpness over the 100 states; 0 = the plain mean")
_ap.add_argument("-iters", type=int, default=None, help="meta-iterations")
_ap.add_argument("-inner_k", type=int, default=None, help="inner steps per task")
_ap.add_argument("-lr", type=float, default=None,
                 help="base learning rate. With LR-scaling on (default), "
                      "alpha = lr * K_REF / k, so alpha*k = lr * K_REF is held "
                      "fixed as -inner_k changes.")
_ap.add_argument("-no_lr_scale", action="store_true",
                 help="do NOT scale the inner learning rate with k")
_ap.add_argument("-meta_eps", type=float, default=None, help="outer step size")
_ap.add_argument("-tasks", type=int, default=None, help="windows per meta-iteration")
_ap.add_argument("-save_every", type=int, default=None,
                 help="checkpoint every N meta-iterations (0 = only at the end)")
_ap.add_argument("-time_input", action="store_true",
                 help="append normalised batch time to the policy input")
_ap.add_argument("-out", default=None)
_ap.add_argument("-init_policy", default=None, help="warm start")
_args = _ap.parse_known_args()[0]
if _args.lam is not None:    LAMBDA_A = _args.lam
if _args.lam_l1 is not None: LAMBDA_L1 = _args.lam_l1
if _args.beta is not None:   BETA = _args.beta
if _args.iters:               N_ITERS = _args.iters
if _args.inner_k:             INNER_K = _args.inner_k
if _args.lr is not None:      LR = _args.lr
if _args.no_lr_scale:         SCALE_LR_WITH_K = False
if _args.meta_eps:            META_EPS = _args.meta_eps
if _args.tasks:                TASKS_PER_META = _args.tasks
if _args.save_every is not None: SAVE_EVERY = _args.save_every
USE_TIME_INPUT = _args.time_input
OUT = _args.out or os.path.join(SAVE_DIR, "reptile_softmax_policy.pt")


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
    c0 = np.array(np.asarray(_m["centers_init"]).tolist(), dtype=np.float64)
    mo, so = np.asarray(_warm["std_obs_mu"]), np.asarray(_warm["std_obs_sd"])
    mn, sn = stats["std_obs_mu"], stats["std_obs_sd"]
    centers_init = (c0 * so + mo - mn) / sn
    lengthscales_init = (np.array(np.asarray(_m["lengthscales_init"]).tolist(),
                                  dtype=np.float64) * so / sn)
    print(f"[warm] from {os.path.basename(_args.init_policy)}, centres remapped "
          f"(max mean-shift {float(np.abs((mo-mn)/sn).max()):.3f} sigma)")

POLICY_IN = STATE_DIM + (1 if USE_TIME_INPUT else 0)
_s_lo = s_lo.tolist() + ([0.0] if USE_TIME_INPUT else [])
_s_hi = s_hi.tolist() + ([1.0] if USE_TIME_INPUT else [])

policy, policy_meta = build_policy(
    "rbf", POLICY_IN, INPUT_DIM, u_max=U_MAX, dtype=dtype, device=device,
    rng=np.random.default_rng(0), num_basis=NUM_BASIS,
    centers_init=centers_init, lengthscales_init=lengthscales_init,
    s_lo=_s_lo, s_hi=_s_hi, center_range_pad=CENTER_RANGE_PAD)
if _warm is not None:
    policy.load_state_dict(_warm["policy_state_dict"])

print(f"\npolicy rbf: in={POLICY_IN} out={INPUT_DIM} u_max={U_MAX} "
      f"params={policy_meta['n_params']}"
      + ("  (8 state + 1 normalised batch time)" if USE_TIME_INPUT else ""))
INNER_LR = (LR * K_REF / INNER_K) if SCALE_LR_WITH_K else LR
print(f"Reptile: k={INNER_K} inner steps, eps={META_EPS}, "
      f"{TASKS_PER_META} tasks/meta-iter, {N_ITERS} meta-iters")
print(f"  base lr={LR}  inner alpha={INNER_LR:.5f}"
      + (f"  (= lr*{K_REF}/k, so alpha*k = {INNER_LR*INNER_K:.4f} is held fixed "
         f"as k varies)"
         if SCALE_LR_WITH_K else f"  (unscaled; alpha*k = {INNER_LR*INNER_K:.4f}, "
         f"grows with k)"))
print(f"  gradient-agreement coefficient  0.5*k*(k-1)*alpha = "
      f"{0.5*INNER_K*(INNER_K-1)*INNER_LR:.5f}   (exactly 0 at k=1)")
if INNER_K == 1:
    print("  WARNING: k=1 makes Reptile EXACTLY joint training on the mixture")
print(f"aggregation over the {NUM_STATES} states: "
      + (f"softmax, beta={BETA}" if BETA > 0 else "PLAIN MEAN (beta=0)"))
print(f"L1: lambda_l1={LAMBDA_L1} on discharge, from CLOSED   "
      f"L2: lambda={LAMBDA_A} on the other 5 channels")
print(f"checkpointing every {SAVE_EVERY} meta-iterations"
      if SAVE_EVERY > 0 else "checkpointing ONLY at the end (SAVE_EVERY=0)")

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

# discharge's CLOSED level in the policy's z-space, and a mask for the L2 channels
_off_smpl = (2.0 * (DISCHARGE_OFF_PHYS - pdata.MIN_ACT[DISCHARGE_IDX])
             / (pdata.MAX_ACT[DISCHARGE_IDX] - pdata.MIN_ACT[DISCHARGE_IDX]) - 1.0)
A_OFF_Z = float((_off_smpl - stats["std_act_mu"][DISCHARGE_IDX])
                / stats["std_act_sd"][DISCHARGE_IDX])
_l2_mask = np.ones(INPUT_DIM); _l2_mask[DISCHARGE_IDX] = 0.0
L2_MASK = torch.tensor(_l2_mask, dtype=dtype, device=device)
print(f"\naction penalty: {LAMBDA_L1} * |a_disch - {A_OFF_Z:.3f}|  (L1, from CLOSED)")
print(f"                + {LAMBDA_A} * ||a_1..5||^2                (L2, from the mean)")

class _TimeAware(torch.nn.Module):
    def __init__(self, base, t_max=230.0):
        super().__init__()
        self.base, self.t_max = base, t_max
        self.state_dim, self.input_dim = STATE_DIM, INPUT_DIM
        self.t_now = 0.0

    def forward(self, states, t=None, p_dropout=0.0):
        tt = torch.full((states.shape[0], 1),
                        float(self.t_now) / self.t_max,
                        dtype=states.dtype, device=states.device)
        return self.base(states=torch.cat([states, tt], dim=1), t=t,
                         p_dropout=p_dropout)


if USE_TIME_INPUT:
    policy = _TimeAware(policy)

rng = np.random.default_rng(0)


# ================================================================== window loss ==
_acc = {"var": None, "t0": 0}
_acc_a = []
_eig = None
_log = {"w2": [], "l2": [], "l1": [], "closed": [], "cc_a": [], "cc_s": [],
        "eff": [], "wmax": []}


def window_loss(t, s, a, mu, cov, s_next):
    """Accumulate the 5 steps into one 1-hour transition, then score it."""
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
    dv = d.view(NUM_STATES, K_ACTIONS).mean(dim=1)             # E_a|s -> (NUM_STATES,)
    if BETA > 0:
        wt = torch.softmax(dv.detach() * BETA, dim=0)
        w2 = (wt * dv).sum()
        with torch.no_grad():
            _log["eff"].append(float(1.0 / (wt ** 2).sum()))
            _log["wmax"].append(float(wt.max() / wt.min()))
    else:
        w2 = dv.mean()
        _log["eff"].append(float(NUM_STATES)); _log["wmax"].append(1.0)
    _log["w2"].append(float(dv.mean().detach()))

    a_all = torch.cat(_acc_a, 0)
    n_rep = a_all.shape[0] // (NUM_STATES * K_ACTIONS)

    # L2 on channels 1..5 only
    l2 = ((a_all ** 2) * L2_MASK).sum(dim=1).mean()
    _log["l2"].append(float(l2.detach()))
    # L1 on discharge, measured from the CLOSED level
    l1 = torch.abs(a_all[:, DISCHARGE_IDX] - A_OFF_Z).mean()
    _log["l1"].append(float(l1.detach()))
    with torch.no_grad():
        _log["closed"].append(float((torch.abs(a_all[:, DISCHARGE_IDX] - A_OFF_Z)
                                     < 0.05).to(dtype).mean()))

    cc_a = action_chance_penalty(a_all, CC_LO, CC_HI,
                                 num_states=NUM_STATES * n_rep, k_actions=K_ACTIONS,
                                 eps=CC_EPS, alpha=CC_ALPHA_ACT)
    cc_s = state_chance_penalty(mu, cov, WT_IDX, lo=CC_WT_MIN_Z, hi=None,
                                num_states=NUM_STATES, k_actions=K_ACTIONS,
                                eps=CC_EPS, alpha=CC_ALPHA_STATE)
    _log["cc_a"].append(float(cc_a.detach()))
    _log["cc_s"].append(float(cc_s.detach()))
    _acc_a.clear()

    return w2 + LAMBDA_A * l2 + LAMBDA_L1 * l1 + cc_a + cc_s


# ============================================================== Reptile training ===
def _run_window(t_h, gen):
    global _eig
    _eig = EXPERT_EIGS[round(t_h, 6)]
    ph = phase_of(t_h)
    if USE_TIME_INPUT:
        policy.t_now = t_h
    st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=gen,
                                  dtype=dtype, device=device)
    s0 = st.repeat_interleave(K_ACTIONS, dim=0)
    _acc_a.clear()
    out = gp_rollout(model=MODELS[ph], policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                     p_dropout=P_DROPOUT, particle_pred=True,
                     loss_fn=window_loss, graph_mode="full")
    return out["loss_total"], out


def _save(path):
    torch.save({"policy_state_dict": policy.state_dict(), "policy_meta": policy_meta,
                "policy_kind": "rbf", "hist": hist, "lam": LAMBDA_A,
                "lam_l1": LAMBDA_L1, "beta": BETA,
                "inner_k": INNER_K, "lr": LR, "inner_lr": INNER_LR,
                "scale_lr_with_k": SCALE_LR_WITH_K, "meta_eps": META_EPS,
                "tasks_per_meta": TASKS_PER_META, "use_time_input": USE_TIME_INPUT,
                "l2_first": l2_first, "l2_last": l2m,
                "completed_iters": it + 1,
                "phase_prefix": _args.phase_prefix,
                "std_obs_mu": stats["std_obs_mu"].tolist(),
                "std_obs_sd": stats["std_obs_sd"].tolist(),
                "std_act_mu": stats["std_act_mu"].tolist(),
                "std_act_sd": stats["std_act_sd"].tolist()}, path)


hist, l2_first = [], None
for it in range(N_ITERS):
    theta0 = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    phis, L, G, S = [], [], [], []
    for k in _log:
        _log[k].clear()

    task_ids = rng.permutation(len(EXPERT_TIMES))[:TASKS_PER_META]
    for tid in task_ids:
        t_h = float(EXPERT_TIMES[tid])
        policy.load_state_dict(theta0)
        inner_opt = torch.optim.Adam(policy.parameters(), lr=INNER_LR)
        gen = np.random.default_rng(int(tid) + 10000 * it)
        for _ in range(INNER_K):
            loss, out = _run_window(t_h, gen)
            inner_opt.zero_grad()
            loss.backward()
            gn = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                                if p.grad is not None)).item()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
            inner_opt.step()
            L.append(loss.item()); G.append(gn)
            S.append(out["S"].detach().abs().max().item())

        phis.append({k: v.detach().clone() for k, v in policy.state_dict().items()})

    new_state, drift = {}, 0.0
    for k in theta0:
        if theta0[k].dtype.is_floating_point:
            mean_phi = torch.stack([p[k] for p in phis]).mean(dim=0)
            new_state[k] = theta0[k] + META_EPS * (mean_phi - theta0[k])
            drift += float((mean_phi - theta0[k]).norm() ** 2)
        else:
            new_state[k] = theta0[k]
    policy.load_state_dict(new_state)

    L, G, S = map(np.array, (L, G, S))
    w2m = float(np.mean(_log["w2"])); l2m = float(np.mean(_log["l2"]))
    if l2_first is None:
        l2_first = l2m
    hist.append(w2m)
    _l1m = np.mean(_log["l1"]); _cl = np.mean(_log["closed"])
    print(f"meta {it:3d}  W2={w2m:.5f}  ||a_1..5||^2={l2m:.5f}  "
          f"discharge L1={_l1m:.5f}  closed_frac={100*_cl:.1f}%  "
          f"loss={L.mean():.5f}  |phi-theta|={np.sqrt(drift):.4e}", flush=True)
    print(f"          {TASKS_PER_META} tasks x {INNER_K} inner steps  "
          f"eff_particles={np.mean(_log['eff']):.1f}/{NUM_STATES}  "
          f"max/min weight={np.mean(_log['wmax']):.1f}x (beta={BETA})", flush=True)
    print(f"          cc_act={np.mean(_log['cc_a']):.3e} "
          f"cc_state={np.mean(_log['cc_s']):.3e}  "
          f"|grad| med={np.median(G):.3e} DEAD={int((G<1e-12).sum())}/{len(G)}  "
          f"|s|max={np.median(S):.1f} (data {s_hi.max():.1f})", flush=True)

    if SAVE_EVERY > 0 and ((it + 1) % SAVE_EVERY == 0 or it == N_ITERS - 1):
        _save(OUT)
        print(f"          [checkpoint saved -> {OUT}]", flush=True)

print(f"\n||a_1..5||^2 {l2_first:.5f} -> {l2m:.5f} "
      f"({100*(l2m-l2_first)/max(l2_first,1e-12):+.1f}%)")

print("\naction spread ACROSS WINDOWS (the thing that collapsed before):")
with torch.no_grad():
    _acts = []
    for _t in (10.0, 50.0, 90.0, 130.0):
        if USE_TIME_INPUT:
            policy.t_now = _t
        _st = POOLS[phase_of(_t)][:64]
        _acts.append(policy(states=_st, t=0, p_dropout=0.0).mean(0))
    _A = torch.stack(_acts)
print(f"  {'channel':14s}" + "".join(f"{f't={t:.0f}h':>10s}"
                                     for t in (10, 50, 90, 130)) + f"{'std':>10s}")
for _i, _nm in enumerate(pdata.ACT_NAMES):
    print(f"  {_nm:14s}" + "".join(f"{_A[j, _i].item():10.4f}" for j in range(4))
          + f"{_A[:, _i].std().item():10.4f}")
print("  (a std of ~0 means one action for every window, whatever the meta-update)")

_save(OUT)
print(f"\nsaved -> {OUT}")
