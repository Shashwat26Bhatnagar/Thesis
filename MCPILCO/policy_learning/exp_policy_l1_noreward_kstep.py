#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/exp_policy_l1_noreward_kstep.py

CDIL policy optimization -- NO REWARD, K-STEP INNER LOOP PER WINDOW.

Two changes from exp_policy_l1.py:

1. REWARD REMOVED ENTIRELY.
   The objective is now, for every window over the whole 1..230 h batch:

       t <= 150 h :  ALPHA_W2*relu(W2_h - eta) + LAMBDA_A*||a_1..5||^2
                     + LAMBDA_L1*|a_disch - a_off| + chance
       t >  150 h :                               LAMBDA_A*||a_1..5||^2
                     + LAMBDA_L1*|a_disch - a_off| + chance

   No reward GP is loaded, no -reward_model argument exists, no KAPPA. Past 150 h
   there is no expert to imitate, so those windows are pure regularization + safety
   -- there is nothing pulling the policy toward "doing something" there beyond
   staying closed on discharge, small on the other channels, and inside the chance
   constraints. That is intentional: this run isolates whether the collapse is a
   training-dynamics artifact (per-window k=1 sequential SGD, per the Reptile
   argument that k=1 sequential single-task steps are equivalent to minimizing the
   AVERAGE loss across tasks) rather than something reward-related.

2. K-STEP INNER LOOP PER WINDOW, ADAPTIVE, NOT k=1.
   For each sampled window: draw ONE fixed initial particle state s0, then take
   repeated gradient steps on THAT SAME window (same s0, same _eig, same phase)
   until its own loss stops improving (relative change < INNER_TOL) or K_MAX steps
   are hit, THEN move to the next window. This is deliberately k>1, breaking the
   k=1-sequential-equals-joint-average-loss equivalence that a single step per
   window falls into.

   K_MAX bounds the opposite risk: a single unusual window (e.g. a rare
   discharge-spike hour) dominating so many consecutive steps that it overwrites
   what earlier windows in the same iteration taught the policy. Combined with the
   existing per-iteration random window reshuffling, this keeps that risk bounded
   without hand-picking a fixed K -- easy windows exit early, hard ones get more
   steps, nothing gets unboundedly many.

3. NO DEPLOYMENT CLIP.
   This script never applied -cliprecipe / BUGGY_RECIPE_CLIP itself -- that clip
   lives in the downstream run_expfull_loop.sh exploration step, not in policy
   training. Nothing here needs changing to satisfy "no clip during training", but
   if you re-run exploration after this, do NOT pass -cliprecipe or set
   BUGGY_RECIPE_CLIP=1 -- otherwise the policy this script produces will be
   evaluated through a transform it was never trained against.

Everything else -- the L1-on-discharge-from-CLOSED / L2-on-the-rest-5 split, the
Cai-Lim covariance-only W2 term, the two Tan et al. chance constraints, the
MC-PILCO one-hour / T=5-step / fresh-particle window structure -- is unchanged
from exp_policy_l1.py.

    python policy_learning/exp_policy_l1_noreward_kstep.py \\
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

SAVE_DIR = os.path.join(_REPO, "results_expl1_noreward_kstep")
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

# --- E_s( E_{a|s}( . ) ) ---
NUM_STATES, K_ACTIONS = 100, 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

# --- episodic structure ---
T_START_HOURS, HOURS_PER_STEP, EXPERT_DT = 0.0, 0.2, 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))       # 5 = one hour
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0     # where the expert exists -> W2 applies
BATCH_T_MAX = 230.0                          # the full episode
WINDOWS_PER_ITER = 150
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0

# --- k-step inner loop, per window ---
K_MAX = 10          # hard cap on gradient steps applied to one window before moving on
INNER_TOL = 1e-3    # stop early once |loss_t - loss_{t-1}| / |loss_{t-1}| < this

# --- policy ---
NUM_BASIS, U_MAX = 200, 3.0
CENTER_RANGE_PAD = 1.10

# --- REPS constraint + action penalties ---
ETA = 0.23                   # measured; see exp_policy_l1.py's docstring
ALPHA_W2 = 15.0               # measured: puts the penalty on a sensible scale
LAMBDA_A = 0.01               # L2 on channels 1..5 (sugar..water)
LAMBDA_L1 = 0.05              # L1 on discharge, measured from the CLOSED level
DISCHARGE_IDX = 0
DISCHARGE_OFF_PHYS = 0.0      # "closed"

# --- chance constraints (Tan et al. Eq. 8-9) ---
CC_EPS = 0.95
CC_ALPHA_ACT = 1000.0
CC_ALPHA_STATE = 1.0
CC_WT_MIN_PHYS = 50000.0

EXPERT_COV_KEY = "cov_n"

_ap = argparse.ArgumentParser("CDIL policy optimization (no reward, k-step inner loop)")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt, selected per "
                      "window by the expert time")
_ap.add_argument("-eta", type=float, default=None)
_ap.add_argument("-alpha_w2", type=float, default=None)
_ap.add_argument("-lam", type=float, default=None, help="L2 weight, channels 1..5")
_ap.add_argument("-lam_l1", type=float, default=None,
                 help="L1 weight on discharge, measured from the closed level")
_ap.add_argument("-k_max", type=int, default=None,
                 help="hard cap on inner-loop gradient steps per window")
_ap.add_argument("-inner_tol", type=float, default=None,
                 help="relative-improvement stopping tolerance for the inner loop")
_ap.add_argument("-iters", type=int, default=None)
_ap.add_argument("-out", default=None)
_ap.add_argument("-init_policy", default=None, help="warm start")
_args = _ap.parse_known_args()[0]
if _args.lam is not None:        LAMBDA_A = _args.lam
if _args.lam_l1 is not None:     LAMBDA_L1 = _args.lam_l1
if _args.eta is not None:        ETA = _args.eta
if _args.alpha_w2 is not None:   ALPHA_W2 = _args.alpha_w2
if _args.k_max is not None:      K_MAX = _args.k_max
if _args.inner_tol is not None:  INNER_TOL = _args.inner_tol
if _args.iters:
    N_ITERS = _args.iters
OUT = _args.out or os.path.join(SAVE_DIR, "exp_l1_noreward_kstep_policy.pt")


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
_n_imit = int((WINDOW_TIMES <= EXPERT_T_MAX).sum())
print(f"windows: {len(WINDOW_TIMES)} total over 1..{BATCH_T_MAX:.0f} h  ->  "
      f"{_n_imit} with the W2 constraint (t<={EXPERT_T_MAX:.0f}), "
      f"{len(WINDOW_TIMES)-_n_imit} regularization+safety only (no expert, no reward)")


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

policy, policy_meta = build_policy(
    "rbf", STATE_DIM, INPUT_DIM, u_max=U_MAX, dtype=dtype, device=device,
    rng=np.random.default_rng(0), num_basis=NUM_BASIS,
    centers_init=centers_init, lengthscales_init=lengthscales_init,
    s_lo=s_lo.tolist(), s_hi=s_hi.tolist(), center_range_pad=CENTER_RANGE_PAD)
if _warm is not None:
    policy.load_state_dict(_warm["policy_state_dict"])

print(f"\npolicy rbf: in={STATE_DIM} out={INPUT_DIM} u_max={U_MAX} "
      f"params={policy_meta['n_params']}")

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

print(f"\nobjective (NO REWARD): {ALPHA_W2}*relu(W2_h - {ETA})  [t<={EXPERT_T_MAX:.0f}h only]")
print(f"                      + {LAMBDA_L1} * |a_disch - {A_OFF_Z:.3f}|  (L1, from CLOSED)")
print(f"                      + {LAMBDA_A} * ||a_1..5||^2                (L2, from the mean)")
print(f"                      + chance constraints (action box + vessel floor)")
print(f"inner loop: up to K_MAX={K_MAX} gradient steps per window, "
      f"stop when relative loss change < {INNER_TOL}")
print("NOTE: no deployment clip is applied by this script. If you re-run "
      "exploration afterward, do NOT pass -cliprecipe / BUGGY_RECIPE_CLIP=1 -- "
      "this policy was never trained against that transform.")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


# ================================================================== window loss ==
_acc = {"var": None, "t0": 0}
_acc_a, _acc_s = [], []
_eig = None
_log = {"w2": [], "l2": [], "cc_a": [], "cc_s": [], "viol": [], "pen": [],
        "l1": [], "closed": [], "n_no_expert": [0], "inner_steps": []}


def window_loss(t, s, a, mu, cov, s_next):
    """Accumulate the 5 steps into one 1-hour transition, then score it.
    No reward term anywhere in this function.
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
        d = w2_cross_dim_torch(var_1h, _eig)                   # (P,)
        w2 = d.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()  # E_a|s then E_s
        _log["w2"].append(float(w2.detach()))
        viol = torch.relu(w2 - ETA)
        pen = ALPHA_W2 * viol
        _log["viol"].append(float(viol.detach()))
        _log["pen"].append(float(pen.detach()))
    else:
        pen = torch.zeros((), dtype=var_1h.dtype, device=var_1h.device)
        _log["n_no_expert"][0] += 1

    a_all = torch.cat(_acc_a, 0)
    n_rep = a_all.shape[0] // (NUM_STATES * K_ACTIONS)

    l2 = ((a_all ** 2) * L2_MASK).sum(dim=1).mean()
    _log["l2"].append(float(l2.detach()))
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
    _acc_a.clear(); _acc_s.clear()

    return pen + LAMBDA_A * l2 + LAMBDA_L1 * l1 + cc_a + cc_s


# ==================================================================== training ===
hist, l2_first = [], None
for it in range(N_ITERS):
    order = rng.permutation(len(WINDOW_TIMES))[:WINDOWS_PER_ITER]
    L, G, S = [], [], []
    for k in _log:
        if k in ("n_no_expert",):
            _log[k][0] = 0
        else:
            _log[k].clear()

    for idx in order:
        t_h = float(WINDOW_TIMES[idx])
        _eig = EXPERT_EIGS.get(round(t_h, 6))      # None past EXPERT_T_MAX
        ph = phase_of(t_h)
        # ONE fixed initial particle draw for this window -- the k-step inner loop
        # below repeatedly optimizes against this SAME s0, not a fresh draw each
        # step, so it is genuinely minimizing this window's loss rather than
        # averaging over resampled noise.
        st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=rng,
                                      dtype=dtype, device=device)
        s0 = st.repeat_interleave(K_ACTIONS, dim=0)

        prev_loss = None
        steps_taken = 0
        for k_step in range(K_MAX):
            _acc_a.clear(); _acc_s.clear()
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
            steps_taken += 1
            cur = loss.item()
            if prev_loss is not None:
                rel = abs(cur - prev_loss) / max(abs(prev_loss), 1e-12)
                if rel < INNER_TOL:
                    prev_loss = cur
                    break
            prev_loss = cur

        _log["inner_steps"].append(steps_taken)
        L.append(prev_loss); G.append(gn)
        S.append(out["S"].detach().abs().max().item())

    L, G, S = map(np.array, (L, G, S))
    w2m = float(np.mean(_log["w2"])) if _log["w2"] else float("nan")
    l2m = float(np.mean(_log["l2"]))
    if l2_first is None:
        l2_first = l2m
    hist.append(w2m if _log["w2"] else float("nan"))
    w2a = np.array(_log["w2"]) if _log["w2"] else np.array([np.nan])
    pm = np.mean(_log["pen"]) if _log["pen"] else 0.0
    n_no_exp = _log["n_no_expert"][0]
    ks = np.array(_log["inner_steps"])
    print(f"iter {it:3d}  loss={L.mean():.5f}  W2 mean={w2m:.5f} max={w2a.max():.5f}  "
          f"inner steps: mean={ks.mean():.2f} max={ks.max()} at_cap={int((ks==K_MAX).sum())}/{len(ks)}",
          flush=True)
    print(f"          eta={ETA} violating={int((w2a>ETA).sum())}/{len(w2a)} "
          f"excess={np.mean(_log['viol']) if _log['viol'] else 0:.5f} penalty={pm:.5f}   "
          f"||a_1..5||^2={l2m:.5f} lam*={LAMBDA_A*l2m:.3e}", flush=True)
    _l1m = np.mean(_log["l1"]); _cl = np.mean(_log["closed"])
    print(f"          discharge L1={_l1m:.5f} lam_l1*={LAMBDA_L1*_l1m:.3e}   "
          f"fraction near CLOSED={100*_cl:.1f}%   (gpei duty cycle -> 94.8% closed)",
          flush=True)
    print(f"          windows: {len(w2a) if _log['w2'] else 0} with W2 constraint, "
          f"{n_no_exp} regularization+safety only (t>{EXPERT_T_MAX:.0f} h)", flush=True)
    print(f"          cc_act={np.mean(_log['cc_a']):.3e} "
          f"cc_state={np.mean(_log['cc_s']):.3e}   "
          f"|grad| med={np.median(G):.3e} DEAD={int((G<1e-12).sum())}/{len(G)}  "
          f"|s|max={np.median(S):.1f} (data {s_hi.max():.1f})", flush=True)

print(f"\nfinal W2={w2m:.5f} vs eta={ETA} -> "
      f"{'FEASIBLE' if w2m <= ETA else 'still binding'}")
print(f"      ||a||^2 {l2_first:.5f} -> {l2m:.5f} "
      f"({100*(l2m-l2_first)/max(l2_first,1e-12):+.1f}%)")

torch.save({"policy_state_dict": policy.state_dict(), "policy_meta": policy_meta,
            "policy_kind": "rbf", "hist": hist, "lam": LAMBDA_A, "lam_l1": LAMBDA_L1,
            "k_max": K_MAX, "inner_tol": INNER_TOL,
            "l2_first": l2_first, "l2_last": l2m,
            "phase_prefix": _args.phase_prefix,
            "std_obs_mu": stats["std_obs_mu"].tolist(),
            "std_obs_sd": stats["std_obs_sd"].tolist(),
            "std_act_mu": stats["std_act_mu"].tolist(),
            "std_act_sd": stats["std_act_sd"].tolist()}, OUT)
print(f"\nsaved -> {OUT}")
