#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

for _cand in (
    os.path.expanduser("~/penicillin-dcfba"),
    os.path.join(os.path.dirname(_REPO), "penicillin-dcfba"),
    os.path.join(os.path.dirname(os.path.dirname(_REPO)), "penicillin-dcfba"),
):
    if os.path.isdir(_cand):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break
else:
    raise FileNotFoundError("penicillin-dcfba not found -- set the path manually")

import argparse

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
import policy_learning.Policy as Policy
from policy_learning.gp_particle_rollout import gp_rollout, sample_initial_particles
from policy_learning.wasserstein_loss import w2_cross_dim_torch
from policy_learning.policy_variants import build_policy
from policy_learning.chance_constraints import (action_chance_penalty, phi_inv,
                                                RecipeBounds, state_chance_penalty,
                                                action_violation_multiplier)
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

SAVE_DIR = os.path.join(_REPO, "results_pensim")

_ap = argparse.ArgumentParser("CDIL policy optimization")
_ap.add_argument("-model", type=str, default=None,
                 help="single world-model checkpoint (Dyna loop). Omit to use the "
                      "three phase models.")
_ap.add_argument("-init_policy", type=str, default=None,
                 help="warm-start from this policy checkpoint (Dyna loop). The RBF "
                      "CENTRES are taken from it too -- regenerating them would make "
                      "the loaded weights meaningless.")
_ap.add_argument("-out", type=str, default=None, help="output policy path")
_ap.add_argument("-iters", type=int, default=None, help="override N_ITERS")
_ap.add_argument("-policy_kind", type=str, default=None,
                 choices=["rbf", "mlp", "kan"],
                 help="policy architecture (default: POLICY_KIND below). "
                      "rbf = Sum_of_gaussians (joint Gaussian basis, MC-PILCO's own); "
                      "mlp = feed-forward; "
                      "kan = Kolmogorov-Arnold with radial-basis edge functions")
_args = _ap.parse_known_args()[0]

USE_PHASE_MODELS = _args.model is None
MODEL_PATHS = {0: os.path.join(SAVE_DIR, "rbf_model_phase0.pt"),
               1: os.path.join(SAVE_DIR, "rbf_model_phase1.pt"),
               2: os.path.join(SAVE_DIR, "rbf_model_phase2.pt")}
ALL_MODEL_PATH = _args.model or os.path.join(SAVE_DIR, "rbf_model_all.pt")

STATE_DIM = pdata.OBS_DIM
INPUT_DIM = pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

NUM_STATES = 100
K_ACTIONS = 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

T_START_HOURS = 0.0
HOURS_PER_STEP = 0.2
EXPERT_DT = 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0
WINDOWS_PER_ITER = 150
N_ITERS = 20
LR = 0.01
P_DROPOUT = 0.25
CLIP = 10.0

EXPERT_COV_KEY = "cov_n"

USE_CHANCE_CONSTRAINT = True
CC_EPS = 0.95
CC_ALPHA = 1000.0
CC_USE_STATIC = True
CC_USE_RECIPE = False
CC_RECIPE_FRAC = 0.10
CC_RECIPE_FLOOR = 0.05
CC_RECIPE_SMOOTH_H = 2.0
CC_ALPHA_RECIPE = 1000.0

CC_USE_STATE = True
CC_STATE_CHANNEL = "Wt"
CC_WT_MIN_PHYS = 50000.0
CC_ALPHA_STATE = 1000.0

USE_ACTION_MULTIPLIER = True
AM_BETA = 1000.0
AM_CAP = 100.0

PHYS_L2_FRAC = 0.10


POLICY_KIND = "rbf"
MLP_HIDDEN = (48, 48)
KAN_HIDDEN, KAN_GRID = 10, 20
KAN_RANGE = (-3.0, 3.0)

NUM_BASIS = 200
ACTION_LIMIT_FRAC = 0.10
U_MAX_FLAT = 3.0
ENFORCE_ACTION_LIMITS = False
CENTER_RANGE_PAD = 1.10


def load_rbf_model(path):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    init_dict = dict(
        active_dims=np.arange(0, GP_INPUT_DIM),
        lengthscales_init=np.ones(GP_INPUT_DIM), flg_train_lengthscales=True,
        lambda_init=np.ones(1), flg_train_lambda=True,
        sigma_n_init=1e-2 * np.ones(1), sigma_n_num=1e-4, flg_train_sigma_n=True,
        dtype=dtype, device=device,
    )
    model = ML.Model_learning_RBF(
        num_gp=STATE_DIM,
        init_dict_list=[dict(init_dict) for _ in range(STATE_DIM)],
        approximation_mode=None, dtype=dtype, device=device, flg_norm=False,
    )
    model.load_state_dict(ckpt["state_dict"])
    for k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
              "K_X_inv_list", "gp_inputs_tr_list"):
        setattr(model, k, ckpt[k])
    model.num_samples = ckpt["gp_inputs"].shape[0]
    model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM
    model.norm_list = [1.0] * STATE_DIM
    stats = {k: np.asarray(ckpt[k]) for k in
             ("std_obs_mu", "std_obs_sd", "std_act_mu", "std_act_sd")}
    meta = {k: ckpt.get(k) for k in ("phase", "phase_tag", "phase_t_lo", "phase_t_hi",
                                     "n_epoch", "select_mode", "train_t_min",
                                     "train_t_max")}
    return model, stats, meta


MODELS, POOLS, METAS = {}, {}, {}
_paths = MODEL_PATHS if USE_PHASE_MODELS else {-1: ALL_MODEL_PATH}
for _ph, _path in _paths.items():
    if not os.path.exists(_path):
        raise FileNotFoundError(f"world model for phase {_ph} not found: {_path}")
    _m, stats, _meta = load_rbf_model(_path)
    _m.set_eval_mode()
    MODELS[_ph], POOLS[_ph], METAS[_ph] = _m, _m.gp_inputs[:, :STATE_DIM], _meta

_sd_ref = None
for _ph in sorted(MODELS):
    _sd = np.asarray(torch.load(_paths[_ph], map_location="cpu",
                                weights_only=False)["std_obs_sd"])
    if _sd_ref is None:
        _sd_ref = _sd
    _d = float(np.abs(_sd - _sd_ref).max())
    if _d > 1e-10:
        raise RuntimeError(
            f"phase {_ph} was standardized differently (max|sd-sd_ref|={_d:.3e}). "
            "All phase models must share one z-space -- retrain with the Standardizer "
            "fitted on the FULL dataset before filtering.")

print("world models loaded:")
for _ph in sorted(MODELS):
    _mt = METAS[_ph]
    _hi = "inf" if (_mt["phase_t_hi"] or 0) > 1e8 else f"{_mt['phase_t_hi']:.0f}"
    print(f"  phase {_ph}: [{_mt['phase_t_lo']:.0f},{_hi}) h  "
          f"train pts={MODELS[_ph].gp_inputs.shape[0]}  epochs={_mt['n_epoch']}  "
          f"select={_mt['select_mode']}")
print("  -> all models share one z-space (verified)")

_all_states = torch.cat([POOLS[p] for p in sorted(POOLS)], dim=0)
s_lo = _all_states.min(0).values
s_hi = _all_states.max(0).values
print(f"combined training state range: [{s_lo.min().item():.2f}, "
      f"{s_hi.max().item():.2f}] (z-units)")


def phase_of(t_hours):
    """Which world model covers this expert time? Uses pdata.PHASES so the boundaries
    cannot drift out of sync with the trainer."""
    if not USE_PHASE_MODELS:
        return -1
    for ph in (0, 1, 2):
        lo, hi = pdata.PHASES[ph]
        if lo <= t_hours < hi:
            return ph
    return 2


class ExpertOracle:
    def __init__(self, source="traj", verbose=True):
        self.q = PFQuery(verbose=verbose)
        self.source = source
        self._cache = {}

    def at_time(self, t_hours):
        key = round(float(t_hours), 6)
        if key not in self._cache:
            d = self.q.next_state_distribution(t=key, source=self.source)
            self._cache[key] = {
                "b": torch.tensor(np.asarray(d["b"]).tolist(), dtype=dtype, device=device),
                "cov": torch.tensor(np.asarray(d["cov"]).tolist(), dtype=dtype, device=device),
                "cov_n": torch.tensor(np.asarray(d["cov_n"]).tolist(), dtype=dtype, device=device),
            }
        return self._cache[key]


expert = ExpertOracle(source="traj")
EXPERT_TIMES = np.arange(EXPERT_T_MIN, EXPERT_T_MAX + 1e-9, EXPERT_DT)
print(f"pre-caching {len(EXPERT_TIMES)} expert distributions "
      f"({EXPERT_T_MIN}..{EXPERT_T_MAX} h) ...", flush=True)
EXPERT_EIGS = {round(float(t), 6):
               torch.linalg.eigvalsh(expert.at_time(t)[EXPERT_COV_KEY].detach())
               for t in EXPERT_TIMES}
print(f"  done. example eigenvalues @75h: {EXPERT_EIGS[75.0].numpy()}", flush=True)

_cnt = {}
for _t in EXPERT_TIMES:
    _cnt[phase_of(float(_t))] = _cnt.get(phase_of(float(_t)), 0) + 1
print("expert windows per world model: " +
      "  ".join(f"phase {k}: {v}" for k, v in sorted(_cnt.items())))


def action_limits_z(frac=ACTION_LIMIT_FRAC):
    """+/- frac of setpoint, expressed in the model's Z-SCORED action units.

    Chain: physical -> smpl min-max -> z-score.
        a_smpl = 2 (a_phys - lo) / (hi - lo) - 1
        a_z    = (a_smpl - mu) / sd
    A physical delta of frac*setpoint therefore becomes
        delta_z = 2 * frac * setpoint_phys / ((hi - lo) * sd)
    The setpoint is taken as the dataset mean action (in physical units).
    """
    lo, hi = pdata.MIN_ACT, pdata.MAX_ACT
    mu_z, sd_z = stats["std_act_mu"], stats["std_act_sd"]
    setpoint_smpl = mu_z
    setpoint_phys = (setpoint_smpl + 1.0) / 2.0 * (hi - lo) + lo
    delta_z = 2.0 * frac * np.abs(setpoint_phys) / ((hi - lo) * sd_z)
    return delta_z, setpoint_phys


U_MAX_Z, SETPOINT_PHYS = action_limits_z()
print(f"\nper-channel action limits (+/-{ACTION_LIMIT_FRAC*100:.0f}% of setpoint):")
for i, nm in enumerate(pdata.ACT_NAMES):
    print(f"    {nm:14s} setpoint={SETPOINT_PHYS[i]:10.3f}   u_max_z={U_MAX_Z[i]:.4f}")
print(f"  (previous flat u_max was 3.0 -> "
      f"{np.round(3.0/U_MAX_Z, 1)}x too wide per channel)")


class BoundedPolicy(torch.nn.Module):
    """Per-channel action scaling. NOT used by default -- see ENFORCE_ACTION_LIMITS.

    CAUTION: wrapping a base policy that squashes to [-1, 1] and then scaling by a
    small per-channel factor crushes the dropout-induced action diversity. With
    +/-10% bounds the spread across replicas fell to ~1e-5, i.e. E_{a|s} collapsed to
    a single sample. If you re-enable this, verify the 'action spread' print below.
    """

    def __init__(self, base, u_max_vec):
        super().__init__()
        self.base = base
        self.register_buffer("u_scale",
                             torch.tensor(u_max_vec, dtype=dtype, device=device))
        self.state_dim = base.state_dim
        self.input_dim = base.input_dim

    def forward(self, states, t=None, p_dropout=0.0):
        return self.base(states=states, t=t, p_dropout=p_dropout) * self.u_scale


_warm = None
POLICY_KIND = _args.policy_kind or POLICY_KIND
_warm_meta = None
if _args.init_policy and os.path.exists(_args.init_policy):
    _warm = torch.load(_args.init_policy, map_location=device, weights_only=False)
    _warm_meta = _warm.get("policy_meta")
    if _warm_meta and _warm_meta.get("kind") != POLICY_KIND:
        raise RuntimeError(
            f"warm start mismatch: checkpoint is '{_warm_meta['kind']}' but "
            f"POLICY_KIND is '{POLICY_KIND}'. Architectures are not interchangeable.")

centers_init = lengthscales_init = None
if POLICY_KIND == "rbf" and _warm_meta is not None:
    centers_init = np.array(np.asarray(_warm_meta["centers_init"]).tolist(),
                            dtype=np.float64)
    _mu_old = np.array(np.asarray(_warm["std_obs_mu"]).tolist(), dtype=np.float64)
    _sd_old = np.array(np.asarray(_warm["std_obs_sd"]).tolist(), dtype=np.float64)
    _mu_new = np.array(np.asarray(stats["std_obs_mu"]).tolist(), dtype=np.float64)
    _sd_new = np.array(np.asarray(stats["std_obs_sd"]).tolist(), dtype=np.float64)
    _shift = float(np.abs((_mu_old - _mu_new) / _sd_new).max())
    _scale = float(np.abs(_sd_old / _sd_new - 1.0).max())
    centers_init = (centers_init * _sd_old + _mu_old - _mu_new) / _sd_new
    lengthscales_init = (np.array(np.asarray(_warm_meta["lengthscales_init"]).tolist(),
                                  dtype=np.float64) * _sd_old / _sd_new)
    print(f"[loop] z-space drift: max mean-shift={_shift:.3f} sigma, "
          f"max scale change={100*_scale:.1f}%  -> centres remapped")
    if _scale > 0.5:
        print("[loop] WARNING: >50% scale change; the ACTION space rescaled too and "
              "the output squashing makes that non-invertible -- warm start is "
              "approximate on the action side.")
elif _warm_meta is not None:
    print(f"[loop] warm start for '{POLICY_KIND}': weights loaded as-is. Unlike the "
          f"rbf centres, these cannot be remapped when the standardizer refits, so "
          f"the transfer is approximate.")

policy, policy_meta = build_policy(
    POLICY_KIND, STATE_DIM, INPUT_DIM,
    u_max=(1.0 if ENFORCE_ACTION_LIMITS else U_MAX_FLAT),
    dtype=dtype, device=device, rng=np.random.default_rng(0),
    num_basis=NUM_BASIS, centers_init=centers_init,
    lengthscales_init=lengthscales_init,
    s_lo=s_lo.tolist(), s_hi=s_hi.tolist(), center_range_pad=CENTER_RANGE_PAD,
    mlp_hidden=MLP_HIDDEN,
    kan_hidden=KAN_HIDDEN, kan_grid=KAN_GRID, kan_range=KAN_RANGE)

if ENFORCE_ACTION_LIMITS:
    policy = BoundedPolicy(policy, U_MAX_Z)

if _warm is not None:
    policy.load_state_dict(_warm["policy_state_dict"])
    _h = _warm.get("hist", [float("nan")])
    print(f"[loop] warm-started from {_args.init_policy} (previous final W2 = {_h[-1]:.4f})")

print(f"\npolicy '{POLICY_KIND}': in={STATE_DIM} out={INPUT_DIM} "
      f"params={policy_meta['n_params']}"
      f"{'  (per-channel bounded)' if ENFORCE_ACTION_LIMITS else f'  u_max={U_MAX_FLAT}'}")
print(f"EPISODIC: T={STEPS_PER_EXPERT} steps ({EXPERT_DT} h) per window, "
      f"{WINDOWS_PER_ITER} windows/iter x {N_ITERS} iters = "
      f"{WINDOWS_PER_ITER*N_ITERS} policy updates")
print(f"objective: E_s(E_a|s(W2))  states={NUM_STATES} x actions={K_ACTIONS} "
      f"= {NUM_PARTICLES} particles;  means EXCLUDED (cross_dim)")
print(f"world models: {'PHASE-SPECIFIC (3)' if USE_PHASE_MODELS else 'SINGLE'}")

with torch.no_grad():
    _st = POOLS[sorted(POOLS)[0]][:1].expand(K_ACTIONS, -1).contiguous()
    _sp = policy(states=_st, t=0, p_dropout=P_DROPOUT).std(0).mean().item()
print(f"action spread across {K_ACTIONS} replicas of ONE state: {_sp:.3e}"
      f"{'   <-- WARNING: ~0 means E_a|s is degenerate' if _sp < 1e-4 else '   (ok)'}")

_amin = 2.0 * (pdata.MIN_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
_amax = 2.0 * (pdata.MAX_ACT - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
_mu_a = np.asarray(stats["std_act_mu"], dtype=np.float64)
_sd_a = np.asarray(stats["std_act_sd"], dtype=np.float64)
CC_LO = torch.tensor((_amin - _mu_a) / _sd_a, dtype=dtype, device=device)
CC_HI = torch.tensor((_amax - _mu_a) / _sd_a, dtype=dtype, device=device)
RECIPE_BOUNDS = None
if USE_CHANCE_CONSTRAINT and CC_USE_RECIPE:
    from pensimpy.examples.recipe import Recipe, RecipeCombo
    from pensimpy.data.constants import (
        FS, FOIL, FG, PRES, DISCHARGE, WATER,
        FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
        PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE)
    _keys = [DISCHARGE, FS, FOIL, FG, PRES, WATER]
    _rc = RecipeCombo(recipe_dict={
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        FS:        Recipe(FS_DEFAULT_PROFILE, FS),
        FOIL:      Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG:        Recipe(FG_DEFAULT_PROFILE, FG),
        PRES:      Recipe(PRESS_DEFAULT_PROFILE, PRES),
        WATER:     Recipe(WATER_DEFAULT_PROFILE, WATER)})
    RECIPE_BOUNDS = RecipeBounds(_rc, _keys, pdata.MIN_ACT, pdata.MAX_ACT,
                                 _mu_a, _sd_a, frac=CC_RECIPE_FRAC,
                                 floor_frac=CC_RECIPE_FLOOR,
                                 smooth_h=CC_RECIPE_SMOOTH_H)

if USE_CHANCE_CONSTRAINT:
    print(f"\nchance constraints ON (Tan et al. Eq. 9): eps={CC_EPS} "
          f"-> Phi^-1={phi_inv(CC_EPS):.4f}")
    print(f"  static box   : {'ON' if CC_USE_STATIC else 'off'}  alpha={CC_ALPHA}")
    print(f"  recipe band  : {'ON' if CC_USE_RECIPE else 'off'}  alpha={CC_ALPHA_RECIPE}"
          f"  +/-{100*CC_RECIPE_FRAC:.0f}% of profile, floor {100*CC_RECIPE_FLOOR:.0f}% "
          f"of span, smoothed +/-{CC_RECIPE_SMOOTH_H} h")
    for i, nm in enumerate(pdata.ACT_NAMES):
        line = f"    {nm:14s} static z-box [{CC_LO[i].item():7.3f}, {CC_HI[i].item():7.3f}]"
        if RECIPE_BOUNDS is not None:
            for _t in (10.0, 110.0):
                _l, _h = RECIPE_BOUNDS.at(_t)
                line += f"   t={_t:5.0f}h [{_l[i]:6.3f}, {_h[i]:6.3f}]"
        print(line)

CC_STATE_IDX = pdata.OBS_NAMES.index(CC_STATE_CHANNEL)
_o_lo, _o_hi = pdata.MIN_OBS[CC_STATE_IDX], pdata.MAX_OBS[CC_STATE_IDX]
_wt_smpl = 2.0 * (CC_WT_MIN_PHYS - _o_lo) / (_o_hi - _o_lo) - 1.0
CC_WT_MIN_Z = float((_wt_smpl - np.asarray(stats["std_obs_mu"])[CC_STATE_IDX])
                    / np.asarray(stats["std_obs_sd"])[CC_STATE_IDX])
if USE_CHANCE_CONSTRAINT and CC_USE_STATE:
    print(f"  state floor  : ON  alpha={CC_ALPHA_STATE}  "
          f"{CC_STATE_CHANNEL} >= {CC_WT_MIN_PHYS:.0f} physical "
          f"= {CC_WT_MIN_Z:.3f} z  (channel {CC_STATE_IDX})")

print(f"  physics      : ON  PHYS_L2_FRAC={PHYS_L2_FRAC}  "
      f"(L_physics = {PHYS_L2_FRAC}*w2.detach() * mean(||a_t_z||^2), capped at "
      f"{PHYS_L2_FRAC*100:.0f}% of each window's own W2 by construction)")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


_acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}
_current_eig = None
_acc_actions = []
_acc_mu, _acc_cov = [], []
_cc_log = []
_am_log = []
_phys_log = []


def window_loss(t, s, a, mu, cov, s_next):
    global _acc
    if _acc["mean"] is None:
        _acc = {"mean": torch.zeros_like(mu), "var": torch.zeros_like(cov),
                "s_start": s, "t_start": t}
    _acc["mean"] = _acc["mean"] + (mu - s)
    _acc["var"] = _acc["var"] + cov
    _acc_actions.append(a)
    _acc_mu.append(mu)
    _acc_cov.append(cov)


    if (t - _acc["t_start"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=mu.dtype, device=mu.device)

    var_1h = _acc["var"]
    t_hours = T_START_HOURS + _acc["t_start"] * HOURS_PER_STEP
    _acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}

    d_all = w2_cross_dim_torch(var_1h, _current_eig)
    w2 = d_all.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()

    _mult = 1.0
    if USE_ACTION_MULTIPLIER and _acc_actions:
        _a_w = torch.cat(_acc_actions, dim=0)
        _mult = action_violation_multiplier(_a_w, CC_LO, CC_HI,
                                            beta=AM_BETA, cap=AM_CAP)
        _am_log.append(float(_mult.detach()))
        w2 = w2 * _mult

    if _acc_actions:
        _l2 = sum((a_i ** 2).sum(dim=-1).mean() for a_i in _acc_actions) / len(_acc_actions)
        _coef = PHYS_L2_FRAC * w2.detach()
        phys = _coef * _l2
        _phys_log.append((float(_l2.detach()), float(_coef.detach()), float(phys.detach())))
        w2 = w2 + phys

    if USE_CHANCE_CONSTRAINT and _acc_actions:
        a_all = torch.cat(_acc_actions, dim=0)
        n_rep = a_all.shape[0] // (NUM_STATES * K_ACTIONS)
        pen = torch.zeros((), dtype=a_all.dtype, device=a_all.device)
        p_static = p_recipe = 0.0

        if CC_USE_STATIC:
            ps = action_chance_penalty(a_all, CC_LO, CC_HI,
                                       num_states=NUM_STATES * n_rep,
                                       k_actions=K_ACTIONS,
                                       eps=CC_EPS, alpha=CC_ALPHA)
            pen = pen + ps
            p_static = float(ps.detach())

        if CC_USE_RECIPE and RECIPE_BOUNDS is not None:
            _lo, _hi = RECIPE_BOUNDS.at(t_hours)
            lo_t = torch.tensor(_lo, dtype=a_all.dtype, device=a_all.device)
            hi_t = torch.tensor(_hi, dtype=a_all.dtype, device=a_all.device)
            pr = action_chance_penalty(a_all, lo_t, hi_t,
                                       num_states=NUM_STATES * n_rep,
                                       k_actions=K_ACTIONS,
                                       eps=CC_EPS, alpha=CC_ALPHA_RECIPE)
            pen = pen + pr
            p_recipe = float(pr.detach())

        p_state = 0.0
        if CC_USE_STATE and _acc_mu:
            mu_all = torch.cat(_acc_mu, dim=0)
            cov_all = torch.cat(_acc_cov, dim=0)
            n_rep_s = mu_all.shape[0] // (NUM_STATES * K_ACTIONS)
            pst = state_chance_penalty(mu_all, cov_all, CC_STATE_IDX,
                                       lo=CC_WT_MIN_Z, hi=None,
                                       num_states=NUM_STATES * n_rep_s,
                                       k_actions=K_ACTIONS,
                                       eps=CC_EPS, alpha=CC_ALPHA_STATE)
            pen = pen + pst
            p_state = float(pst.detach())

        _cc_log.append((float(pen.detach()), p_static, p_recipe, p_state))
        _acc_actions.clear(); _acc_mu.clear(); _acc_cov.clear()
        return w2 + pen
    _acc_actions.clear(); _acc_mu.clear(); _acc_cov.clear()
    return w2


hist = []
for it in range(N_ITERS):
    order = rng.permutation(len(EXPERT_TIMES))[:WINDOWS_PER_ITER]
    losses, gnorms, smax, amax = [], [], [], []
    per_phase = {p: [] for p in MODELS}

    for idx in order:
        t_h = float(EXPERT_TIMES[idx])
        _current_eig = EXPERT_EIGS[round(t_h, 6)]

        ph = phase_of(t_h)
        mdl, pool = MODELS[ph], POOLS[ph]

        s_states = sample_initial_particles(pool, NUM_STATES, generator=rng,
                                            dtype=dtype, device=device)
        s0 = s_states.repeat_interleave(K_ACTIONS, dim=0)

        out = gp_rollout(model=mdl, policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                         p_dropout=P_DROPOUT, particle_pred=True,
                         loss_fn=window_loss, graph_mode="full")

        loss = out["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                            if p.grad is not None)).item()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
        optimizer.step()

        losses.append(loss.item()); gnorms.append(gn)
        smax.append(out["S"].detach().abs().max().item())
        amax.append(out["A"].detach().abs().max().item())
        per_phase[ph].append(loss.item())

    L, G, S, A = map(np.array, (losses, gnorms, smax, amax))
    hist.append(L.mean())
    print(f"iter {it:3d}  W2 mean={L.mean():.6e} min={L.min():.4e} max={L.max():.4e}", flush=True)
    print(f"          |grad| med={np.median(G):.3e} max={G.max():.3e}  "
          f"DEAD(<1e-12)={int((G < 1e-12).sum())}/{len(G)}  "
          f"CLIPPED={int((G > CLIP).sum())}/{len(G)}", flush=True)
    print(f"          |s|max med={np.median(S):.2f} max={S.max():.2f} "
          f"(data {s_hi.max().item():.2f})   |a|max={A.max():.4f}"
          f"{f' (limit {U_MAX_Z.max():.4f})' if ENFORCE_ACTION_LIMITS else ''}",
          flush=True)
    if _am_log:
        _am = np.array(_am_log); _am_log.clear()
        print(f"          action multiplier: mean={_am.mean():.3f} max={_am.max():.3f}  "
              f"windows with violation={int((_am > 1.0 + 1e-9).sum())}/{len(_am)}",
              flush=True)
    if _phys_log:
        _pl = np.array(_phys_log); _phys_log.clear()
        print(f"          physics: mean ||a_z||^2={_pl[:,0].mean():.3e}  "
              f"mean coef={_pl[:,1].mean():.3e}  "
              f"mean L_physics={_pl[:,2].mean():.3e} "
              f"({100*_pl[:,2].mean()/max(L.mean(),1e-12):.1f}% of window-mean W2)",
              flush=True)
    if _cc_log:
        _cc = np.array(_cc_log); _cc_log.clear()
        print(f"          chance penalty: total={_cc[:,0].mean():.3e} "
              f"static={_cc[:,1].mean():.3e} recipe={_cc[:,2].mean():.3e} "
              f"state={_cc[:,3].mean():.3e}", flush=True)
        print(f"          windows violating: static={int((_cc[:,1] > 0).sum())}/{len(_cc)} "
              f"recipe={int((_cc[:,2] > 0).sum())}/{len(_cc)} "
              f"state={int((_cc[:,3] > 0).sum())}/{len(_cc)}", flush=True)
    print("          per-phase W2: " + "  ".join(
        f"ph{p}={np.mean(v):.4e}(n={len(v)})" for p, v in sorted(per_phase.items()) if v),
        flush=True)

    if it % 5 == 0:
        torch.save({"policy_state_dict": policy.state_dict(), "iter": it,
                    "loss": float(L.mean()), "hist": hist,
                    "u_max_z": U_MAX_Z, "policy_meta": policy_meta,
                    "policy_kind": POLICY_KIND,
                    "use_phase_models": USE_PHASE_MODELS},
                   os.path.join(SAVE_DIR, f"cdil_policy_it{it}.pt"))

os.makedirs(SAVE_DIR, exist_ok=True)
_tag = "phasemodels" if USE_PHASE_MODELS else "allmodel"
_out = _args.out or os.path.join(SAVE_DIR, f"cdil_policy_{_tag}.pt")
torch.save({"policy_state_dict": policy.state_dict(), "hist": hist,
            "u_max_z": U_MAX_Z, "policy_meta": policy_meta,
            "policy_kind": POLICY_KIND,
            "use_phase_models": USE_PHASE_MODELS,
            "model_paths": _paths,
            "std_act_mu": stats["std_act_mu"], "std_act_sd": stats["std_act_sd"],
            "std_obs_mu": stats["std_obs_mu"], "std_obs_sd": stats["std_obs_sd"]}, _out)
print(f"\nsaved -> {_out}")
