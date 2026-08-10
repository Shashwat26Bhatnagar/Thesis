#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/valve_policy_optimization.py

Train a SEPARATE discharge policy against a FROZEN policy for the other five
channels, so the discharge valve can be fixed without disturbing action
distributions that already work.

    python policy_learning/valve_policy_optimization.py \\
        -phase_prefix results_pensim/rbf_model_bnd_rbf_iter0 \\
        -frozen_policy results_pensim/cdil_policy_bnd_rbf_iter0.pt \\
        -iters 20

THREE POLICIES, two frozen
    P_B  frozen, L2-regularised on all six channels. Supplies channels 1..5. Its
         action distributions are the closest to the gpei reference of anything
         measured (sugar std 30.4 vs 24.3, water 203 vs 148, aeration 10.3 vs 10.6),
         so they are preserved exactly rather than retrained and hoped for. Its
         DISCHARGE is discarded -- that channel is what drained the vessel
         (61100 -> 25467, episode ending at 121 h).
    P_A  frozen, NOT regularised. Supplies only the STATE TARGET. Its next-state
         distribution is the good one -- vessel weight GREW, 62904 -> 75702, and its
         episodes complete the full 230 h. Its actions are not used at all.
    P_C  trainable, NEW. Emits discharge only.

    composite = concat(P_C(s_aug), P_B(s)[1:]) -- downstream code sees a normal
    6-channel policy, so the rollout, constraints and explore script need no changes.

    Freezing rather than co-training follows CHDP's finding that simultaneous updates
    of two hybrid-action policies conflict, and the standard HRL practice of freezing
    reused modules. The decomposition itself is H-PPO's two-actor structure
    (separate actors per action subset, objectives evaluated separately).

WHAT P_C OPTIMISES -- the part no hybrid-action paper answers
    Those papers assume both actors share a reward. Here the Wasserstein objective
    barely sees discharge, which is how the valve problem arose. So P_C is given a
    BRIDGING target instead:

        reference roll : P_A alone, all six channels        -> S_ref
        candidate roll : P_C on discharge, P_B on 1..5      -> S_cand
        loss = ||S_cand(end) - S_ref(end)||^2  +  valve-shape terms

    i.e. "choose a valve-like discharge such that P_B's good actions land the state
    where P_A landed it". The target is 8-D, fully observed and generated on the fly
    -- no expert, no reward model, no cross-domain projection.

    Note the target comes from a DIFFERENT policy than the one supplying channels
    1..5. That is deliberate: if both came from the same policy the target would be
    nearly free and P_C would have almost nothing to learn.

MATCHING AT THE WINDOW END, NOT EVERY STEP
    A holds discharge nearly constant (std 1.92); a valve pulses. A pulse cannot look
    like a constant at every instant, so per-step matching would be unsatisfiable by
    construction. Matching the state at the END of each 1-hour window permits pulsing
    within the window while keeping the trajectory on track.

B'S EXTRA INPUT: ELAPSED OPEN TIME
    The 8-D state cannot express a timed pulse -- nothing in it distinguishes "just
    opened" from "open for two hours", and discharging drives vessel weight
    monotonically further into the region that triggered the open, so there is no
    return path. Every previous configuration therefore opened at the right time
    (t=102.2 h against the recipe's 102.0) and never closed. B is a fresh network, so
    a 9th input costs nothing in compatibility: it sees [state(8), hours_open].

DISCHARGE IS VALVE-LIKE, NOT BINARY
    The reference is at zero 94.8% of the time and reaches ~4000 when open, but the
    peak varies (3705..4058 across batches). B therefore emits GATE x MAGNITUDE: a
    Bernoulli gate for open/shut, and a continuous level for how far open. A pure
    binary gate would throw away that second degree of freedom.
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
from policy_learning.gp_particle_rollout import gp_rollout, sample_initial_particles
from policy_learning.policy_variants import build_policy, rebuild_policy
from policy_learning.chance_constraints import (gate_sparsity_kl, state_chance_penalty,
                                                phi_inv)

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

# ------------------------------------------------------------------- config ---
SAVE_DIR = os.path.join(_REPO, "results_valve")        # SEPARATE from results_pensim
os.makedirs(SAVE_DIR, exist_ok=True)

STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM     # 8, 6
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
DISCHARGE_IDX = 0

NUM_STATES, K_ACTIONS = 100, 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS
HOURS_PER_STEP, EXPERT_DT = 0.2, 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))      # 5 = window length
T_START_HOURS = 0.0
WINDOWS_PER_ITER = 150
N_ITERS, LR, P_DROPOUT, CLIP = 20, 0.01, 0.25, 10.0

# --- B's valve levels, physical units ---
VALVE_OFF_PHYS = 0.0
VALVE_MAX_PHYS = 4100.0          # env ceiling; the level when open is LEARNED below it
VALVE_OPEN_THRESH_FRAC = 0.5     # "open" = above this fraction of the max, for the
                                 # elapsed-time counter and the duty-cycle log
GATE_GAIN = 2.0

# --- loss weights ---
W_STATE = 1.0                    # ||S_cand(end) - S_ref(end)||^2
W_SPARSE = 0.1                   # KL(gate || target duty)
TARGET_DUTY = 0.052              # measured on gpei_batch_161.csv (6 opens in 230 h)
W_VESSEL = 1.0                   # vessel-weight floor
WT_MIN_PHYS = 50000.0

_ap = argparse.ArgumentParser("train a separate discharge (valve) policy")
_ap.add_argument("-phase_prefix", required=True,
                 help="three world models <prefix>_phase{0,1,2}.pt")
_ap.add_argument("-policy_actions", required=True,
                 help="P_B (frozen): supplies channels 1..5. Use the L2-regularised "
                      "run, whose action distributions match the reference best.")
_ap.add_argument("-policy_states", required=True,
                 help="P_A (frozen): supplies the STATE TARGET only, its actions are "
                      "unused. Use the run whose vessel weight GREW and whose "
                      "episodes completed the full 230 h.")
_ap.add_argument("-out", default=os.path.join(SAVE_DIR, "valve_policy.pt"))
_ap.add_argument("-iters", type=int, default=None)
_ap.add_argument("-w_state", type=float, default=None)
_ap.add_argument("-w_sparse", type=float, default=None)
_args = _ap.parse_known_args()[0]
if _args.iters:    N_ITERS = _args.iters
if _args.w_state is not None:  W_STATE = _args.w_state
if _args.w_sparse is not None: W_SPARSE = _args.w_sparse


# =============================================================== world models ===
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
for p in (0, 1, 2):
    MODELS[p], CKS[p] = load_model(f"{_args.phase_prefix}_phase{p}.pt")
stats = {k: np.asarray(CKS[0][k]) for k in
         ("std_obs_mu", "std_obs_sd", "std_act_mu", "std_act_sd")}
print("world models:")
for p in (0, 1, 2):
    lo, hi = pdata.PHASES[p]
    hi_s = "inf" if hi > 1e8 else f"{hi:g}"
    print(f"  phase {p}: [{lo:g},{hi_s}) h  train pts={MODELS[p].gp_inputs.shape[0]}")
POOL = torch.cat([MODELS[p].gp_inputs[:, :STATE_DIM] for p in (0, 1, 2)], 0)


def model_at(t_h):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t_h < hi:
            return MODELS[p]
    return MODELS[2]


# --------------------------- physical <-> z for the discharge channel ---------
_lo_a, _hi_a = pdata.MIN_ACT[DISCHARGE_IDX], pdata.MAX_ACT[DISCHARGE_IDX]
_mu_a = float(stats["std_act_mu"][DISCHARGE_IDX])
_sd_a = float(stats["std_act_sd"][DISCHARGE_IDX])


def _phys_to_z(v):
    return ((2.0 * (v - _lo_a) / (_hi_a - _lo_a) - 1.0) - _mu_a) / _sd_a


Z_OFF = _phys_to_z(VALVE_OFF_PHYS)
Z_MAX = _phys_to_z(VALVE_MAX_PHYS)
Z_OPEN_THRESH = Z_OFF + VALVE_OPEN_THRESH_FRAC * (Z_MAX - Z_OFF)
print(f"\nvalve levels: off={VALVE_OFF_PHYS:.0f} phys ({Z_OFF:.3f} z)   "
      f"max={VALVE_MAX_PHYS:.0f} phys ({Z_MAX:.3f} z)")

# vessel-weight floor, z-scored
_WT_IDX = pdata.OBS_NAMES.index("Wt")
_o_lo, _o_hi = pdata.MIN_OBS[_WT_IDX], pdata.MAX_OBS[_WT_IDX]
WT_MIN_Z = float(((2.0 * (WT_MIN_PHYS - _o_lo) / (_o_hi - _o_lo) - 1.0)
                  - stats["std_obs_mu"][_WT_IDX]) / stats["std_obs_sd"][_WT_IDX])
print(f"vessel floor: Wt >= {WT_MIN_PHYS:.0f} phys ({WT_MIN_Z:.3f} z), channel {_WT_IDX}")


# ==================================================================== policies ===
def _load_frozen(path, role):
    ck = torch.load(path, map_location=device, weights_only=False)
    pol = rebuild_policy(ck["policy_meta"], dtype=dtype, device=device)
    pol.load_state_dict(ck["policy_state_dict"])
    pol.eval()
    for q in pol.parameters():
        q.requires_grad_(False)                   # gradients must reach P_C only
    print(f"  {role:26s} {os.path.basename(path):40s} kind={ck['policy_meta']['kind']}")
    return pol


print("\nfrozen policies:")
policy_B = _load_frozen(_args.policy_actions, "P_B  -> channels 1..5")
policy_A = _load_frozen(_args.policy_states, "P_A  -> state target only")

# P_C sees [state(8), hours_open] and emits [gate_logit, level_logit]
policy_C, meta_C = build_policy(
    "rbf", STATE_DIM + 1, 2, u_max=3.0, dtype=dtype, device=device,
    rng=np.random.default_rng(0), num_basis=200,
    s_lo=(POOL.min(0).values.tolist() + [0.0]),
    s_hi=(POOL.max(0).values.tolist() + [8.0]))    # up to 8 h open
print(f"\ntrainable P_C: in={STATE_DIM+1} (state + hours_open) out=2 (gate, level)  "
      f"params={meta_C['n_params']}")


class SplitPolicy(nn.Module):
    """P_C controls discharge; frozen P_B supplies channels 1..5. Presents a normal
    6-channel policy to everything downstream.

    Carries `hours_open` across the steps of a rollout, so P_C can learn to CLOSE --
    the information a memoryless 8-D state cannot provide, and the reason every
    previous configuration opened at the right time (t=102.2 h against the recipe's
    102.0) and never closed. reset() must be called before each rollout or the
    counter leaks between windows.
    """

    def __init__(self, B, C):
        super().__init__()
        self.B, self.C = B, C
        self.state_dim, self.input_dim = STATE_DIM, INPUT_DIM
        self._open_h = None
        self._last_p = None

    def reset(self, n_particles):
        self._open_h = torch.zeros(n_particles, dtype=dtype, device=device)

    def forward(self, states, t=None, p_dropout=0.0):
        P = states.shape[0]
        if self._open_h is None or self._open_h.shape[0] != P:
            self.reset(P)
        with torch.no_grad():
            a_B = self.B(states=states, t=t, p_dropout=p_dropout)   # channels 1..5

        u = self.C(states=torch.cat([states, self._open_h.unsqueeze(1)], 1),
                   t=t, p_dropout=p_dropout)
        p_open = torch.sigmoid(GATE_GAIN * u[:, 0])
        hard = torch.bernoulli(p_open)
        gate = hard + p_open - p_open.detach()          # straight-through
        level = Z_OFF + torch.sigmoid(u[:, 1]) * (Z_MAX - Z_OFF)   # how far open
        d = Z_OFF + gate * (level - Z_OFF)
        self._last_p = p_open

        with torch.no_grad():                          # counter carries no gradient
            is_open = (d.detach() > Z_OPEN_THRESH).to(dtype)
            self._open_h = (self._open_h + HOURS_PER_STEP) * is_open   # 0 when shut

        return torch.cat([d.unsqueeze(1), a_B[:, 1:]], dim=1)


policy = SplitPolicy(policy_B, policy_C)
optimizer = torch.optim.Adam(policy_C.parameters(), lr=LR)
rng = np.random.default_rng(0)
_TGT = torch.tensor([TARGET_DUTY], dtype=dtype, device=device)

print(f"\nobjective:  {W_STATE} * ||S_cand(end) - S_ref(end)||^2"
      f"  +  {W_SPARSE} * KL(gate || {TARGET_DUTY})"
      f"  +  {W_VESSEL} * vessel floor")
print(f"windows/iter={WINDOWS_PER_ITER}  iters={N_ITERS}  "
      f"particles={NUM_PARTICLES}  window={STEPS_PER_EXPERT} steps ({EXPERT_DT} h)")


def _noloss(**kw):
    return torch.zeros((), dtype=dtype, device=device)


# ===================================================================== training ==
EXPERT_TIMES = np.arange(1.0, 150.0 + 1e-9, 1.0)
hist = []
for it in range(N_ITERS):
    order = rng.permutation(len(EXPERT_TIMES))[:WINDOWS_PER_ITER]
    L, G, DUTY, SERR = [], [], [], []

    for idx in order:
        t_h = float(EXPERT_TIMES[idx])
        mdl = model_at(t_h)
        s_states = sample_initial_particles(POOL, NUM_STATES, generator=rng,
                                            dtype=dtype, device=device)
        s0 = s_states.repeat_interleave(K_ACTIONS, dim=0)

        # ---- reference: P_A alone (its ACTIONS are discarded, only the states
        #      it reaches are used as the target) ----
        with torch.no_grad():
            out_ref = gp_rollout(model=mdl, policy=policy_A, s0=s0,
                                 T=STEPS_PER_EXPERT, p_dropout=0.0,
                                 particle_pred=False, loss_fn=_noloss,
                                 graph_mode="full")
        S_ref_end = out_ref["S"][:, -1, :].detach()

        # ---- candidate: P_C on discharge, P_B on channels 1..5 ----
        policy.reset(s0.shape[0])
        out = gp_rollout(model=mdl, policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                         p_dropout=P_DROPOUT, particle_pred=False,
                         loss_fn=_noloss, graph_mode="full")
        S_end = out["S"][:, -1, :]

        # end-of-window state match: a pulse cannot look like a constant at every
        # step, so the trajectory is pinned at the window boundary instead
        state_err = ((S_end - S_ref_end) ** 2).sum(1).mean()

        # valve shape: pull the duty cycle toward the reference's
        kl = gate_sparsity_kl(policy._last_p.unsqueeze(1), _TGT)

        # vessel floor on the candidate's own predicted states
        vessel = state_chance_penalty(
            out["Mu"].reshape(-1, STATE_DIM), out["Cov"].reshape(-1, STATE_DIM),
            _WT_IDX, lo=WT_MIN_Z, hi=None, eps=0.95, alpha=W_VESSEL)

        loss = W_STATE * state_err + W_SPARSE * kl + vessel
        optimizer.zero_grad()
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in policy_C.parameters()
                            if p.grad is not None)).item()
        torch.nn.utils.clip_grad_norm_(policy_C.parameters(), CLIP)
        optimizer.step()

        L.append(loss.item()); G.append(gn); SERR.append(state_err.item())
        with torch.no_grad():
            DUTY.append(float((out["A"][:, :, DISCHARGE_IDX] > Z_OPEN_THRESH)
                              .to(dtype).mean()))

    L, G, DUTY, SERR = map(np.array, (L, G, DUTY, SERR))
    hist.append(float(L.mean()))
    print(f"iter {it:3d}  loss={L.mean():.4e}  state_err={SERR.mean():.4e}  "
          f"duty={DUTY.mean():.4f} (target {TARGET_DUTY})", flush=True)
    print(f"          |grad| med={np.median(G):.3e} max={G.max():.3e}  "
          f"DEAD={int((G < 1e-12).sum())}/{len(G)}", flush=True)

    if it % 5 == 0:
        torch.save({"policy_state_dict": policy_C.state_dict(),
                    "policy_meta": meta_C, "iter": it, "hist": hist,
                    "policy_actions": _args.policy_actions,
                    "policy_states": _args.policy_states,
                    "z_off": Z_OFF, "z_max": Z_MAX, "gate_gain": GATE_GAIN},
                   os.path.join(SAVE_DIR, f"valve_policy_it{it}.pt"))

torch.save({"policy_state_dict": policy_C.state_dict(), "policy_meta": meta_C,
            "hist": hist, "policy_actions": _args.policy_actions,
            "policy_states": _args.policy_states,
            "phase_prefix": _args.phase_prefix,
            "z_off": Z_OFF, "z_max": Z_MAX, "gate_gain": GATE_GAIN,
            "open_thresh_z": Z_OPEN_THRESH, "target_duty": TARGET_DUTY,
            "std_obs_mu": stats["std_obs_mu"].tolist(),
            "std_obs_sd": stats["std_obs_sd"].tolist(),
            "std_act_mu": stats["std_act_mu"].tolist(),
            "std_act_sd": stats["std_act_sd"].tolist()}, _args.out)
print(f"\nsaved -> {_args.out}")
print("NOTE: this is P_C ONLY. Deployment needs the SplitPolicy wrapper with the "
      "same frozen P_B for channels 1..5 (P_A is a training target only and is not "
      "needed at deployment).")
