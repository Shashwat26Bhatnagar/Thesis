#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/cdil_policy_optimization.py

Cross-Domain Imitation Learning (CDIL) policy optimization -- EPISODIC,
with PHASE-SPECIFIC world models.

STRUCTURE (mirrors MC-PILCO, Amadio et al. 2022):
    J(theta) = sum_{t=0..T} (1/M) sum_m c(x_t^(m)),  ONE backward, ONE update.
    MC-PILCO horizons are SHORT (cart-pole 3 s / 0.05 s = 60 steps) and p(x_0) is
    RESAMPLED at every optimization step -- the trajectory never continues past T.

    Each expert comparison window (T = STEPS_PER_EXPERT = 5 steps = 1 h) is therefore
    a self-contained episode with a FRESH in-distribution start state. A single long
    continuing rollout instead drove |s| to ~140 z-units (training data spans
    [-5.5, 11.2]), which killed the gradient twice over:
        GP predictive variance saturated at the prior lambda -> dvar/dinput = 0
        policy RBF basis abandoned its centres               -> da/dtheta   = 0

PHASE-SPECIFIC WORLD MODELS  (the change in this revision)
    Fermentation has distinct regimes, so one RBF world model was trained per phase:
        phase 0: t <  35 h       phase 1: 35 <= t < 51 h      phase 2: t >= 51 h
    The expert is queried BY TIME, so each window already knows its t_h -- the same
    clock therefore selects the world model. Because every window is a self-contained
    episode with a fresh start state, no rollout ever crosses a phase boundary, so
    switching models introduces no discontinuity.

    Start states are drawn from the SELECTED model's own training inputs, so the
    particles begin inside the region that model actually covers. This is what keeps
    the gradient alive, so it matters more than it looks.

    All phase models share ONE z-space (the trainer fits the Standardizer on the full
    dataset BEFORE filtering); this is asserted at load time.

    USE_PHASE_MODELS = False falls back to the single all-data model, so the
    "did phase-splitting help?" comparison is one config change.

OBJECTIVE:  E_s( E_{a|s}( W2 ) )
    NUM_STATES start states, each replicated K_ACTIONS times; replicas share a state
    but draw independent actions (dropout), so the inner mean is E_{a|s} and the outer
    mean is E_s. W2 is the Cai-Lim cross-dimensional distance (8-D model vs 3-D
    expert): no projection needed, and the MEANS DROP OUT by construction --
    a deliberate choice; the objective matches covariance spectra only.

POLICY:
    RBF centres are spread over the OBSERVED state range instead of randn's [-3, 3];
    training states span [-5.5, 11.2], so randn centres under-cover it. With phase
    models the range is the UNION over all phases, since one policy serves them all.

    NOTE ON ACTION BOUNDS. The PenSim docs specify +/-10% of setpoint, which in
    z-scored units is u_max_z = [0.023, 0.323, 0.554, 0.628, 0.702, 0.103] -- i.e.
    a flat u_max = 3.0 is 4x to 128x wider. Enforcing that bound was tried and made
    the objective un-optimizable:
        - gradients fell ~30x (0.104 -> 0.0037 median) and the loss went flat
          (0.219 -> 0.206 over 20 iters, vs 0.190 -> 0.105 unbounded)
        - the widest channel saturated its bound at EVERY step (|a|max == 0.7021)
        - |s|max was UNCHANGED (~12.5 med), so the divergence is driven by the GP's
          own delta predictions, not by action magnitude -- the bound cost signal
          without buying in-distribution behaviour
    Covariance-spectrum matching needs the policy to move the state appreciably, and
    +/-10% does not. Flat u_max = 3.0 is therefore retained here; ACTION_LIMIT_FRAC
    below is kept for reference/experimentation.
"""
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
_args = _ap.parse_known_args()[0]

# --- world models: one per fermentation phase, or a single all-data model ---
USE_PHASE_MODELS = _args.model is None
MODEL_PATHS = {0: os.path.join(SAVE_DIR, "rbf_model_phase0.pt"),
               1: os.path.join(SAVE_DIR, "rbf_model_phase1.pt"),
               2: os.path.join(SAVE_DIR, "rbf_model_phase2.pt")}
ALL_MODEL_PATH = _args.model or os.path.join(SAVE_DIR, "rbf_model_all.pt")

STATE_DIM = pdata.OBS_DIM          # 8
INPUT_DIM = pdata.ACT_DIM          # 6
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

# --- E_s( E_{a|s}( . ) ) sampling ---
NUM_STATES = 100
K_ACTIONS = 5
NUM_PARTICLES = NUM_STATES * K_ACTIONS

# --- episodic structure ---
HOURS_PER_STEP = 0.2
EXPERT_DT = 1.0
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))     # = 5 = episode length T
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0
WINDOWS_PER_ITER = 150
N_ITERS = 20
LR = 0.01
P_DROPOUT = 0.25
CLIP = 10.0

EXPERT_COV_KEY = "cov_n"

# --- policy ---
NUM_BASIS = 200
ACTION_LIMIT_FRAC = 0.10           # reference only -- see the note above
U_MAX_FLAT = 3.0                   # actually used (baseline value)
ENFORCE_ACTION_LIMITS = False      # True -> per-channel +/-10% bounds
CENTER_RANGE_PAD = 1.10            # spread centres slightly beyond the data range


# =====================================================================================
# 1. FROZEN GP WORLD MODELS  (one per phase)
# =====================================================================================
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
    _m.set_eval_mode()                     # freeze hyperparameters (input-grad stays live)
    MODELS[_ph], POOLS[_ph], METAS[_ph] = _m, _m.gp_inputs[:, :STATE_DIM], _meta

# all models must share ONE z-space or switching between them is meaningless
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

# policy basis must cover the UNION of the phases' state ranges (one policy, all phases)
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


# =====================================================================================
# 2. EXPERT ORACLE
# =====================================================================================
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

# how many expert windows fall to each world model
_cnt = {}
for _t in EXPERT_TIMES:
    _cnt[phase_of(float(_t))] = _cnt.get(phase_of(float(_t)), 0) + 1
print("expert windows per world model: " +
      "  ".join(f"phase {k}: {v}" for k, v in sorted(_cnt.items())))


# =====================================================================================
# 3. POLICY  --  per-channel action limits + range-covering RBF centres
# =====================================================================================
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
    setpoint_smpl = mu_z                                   # z-space mean is 0 -> smpl mean = mu
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


# CHANGE: centres spread over the OBSERVED state range, not randn's [-3, 3].
# (Random within the range -- NOT sampled from data points.)
# Conversions go via python lists: torch->numpy buffer sharing trips the
# duplicate-numpy ABI mismatch present in this environment.
_warm = None
if _args.init_policy and os.path.exists(_args.init_policy):
    _warm = torch.load(_args.init_policy, map_location=device, weights_only=False)
    centers_init = np.array(np.asarray(_warm["centers_init"]).tolist(), dtype=np.float64)

    # The standardizer is refitted on the UNION each iteration, so the same physical
    # state maps to a DIFFERENT z. The loaded weights describe a function of the OLD
    # z, so the centres are remapped into the new z-space:
    #     centres_new = (centres_old * sd_old + mu_old - mu_new) / sd_new
    # This preserves the policy's behaviour in PHYSICAL units. Without it the warm
    # start would silently describe a different function.
    _mu_old = np.array(np.asarray(_warm["std_obs_mu"]).tolist(), dtype=np.float64)
    _sd_old = np.array(np.asarray(_warm["std_obs_sd"]).tolist(), dtype=np.float64)
    _mu_new = np.array(np.asarray(stats["std_obs_mu"]).tolist(), dtype=np.float64)
    _sd_new = np.array(np.asarray(stats["std_obs_sd"]).tolist(), dtype=np.float64)
    _shift = float(np.abs((_mu_old - _mu_new) / _sd_new).max())
    _scale = float(np.abs(_sd_old / _sd_new - 1.0).max())
    centers_init = (centers_init * _sd_old + _mu_old - _mu_new) / _sd_new

    _ls_old = np.array(np.asarray(_warm.get("lengthscales_init",
                                            np.ones(STATE_DIM))).tolist(),
                       dtype=np.float64)
    lengthscales_init = _ls_old * _sd_old / _sd_new

    print(f"[loop] warm-starting policy from {_args.init_policy}")
    print(f"[loop] z-space drift since that policy: max mean-shift={_shift:.3f} sigma, "
          f"max scale change={100*_scale:.1f}%")
    print(f"[loop] centres and lengthscales remapped into the new z-space")
    if _scale > 0.5:
        print("[loop] WARNING: >50% scale change -- the ACTION space rescaled too, and "
              "the output squashing makes that non-invertible. The warm start is "
              "approximate on the action side.")
else:
    _lo = np.array([float(x) * CENTER_RANGE_PAD for x in s_lo.tolist()], dtype=np.float64)
    _hi = np.array([float(x) * CENTER_RANGE_PAD for x in s_hi.tolist()], dtype=np.float64)
    _span = _hi - _lo
    centers_init = np.array(
        [[_lo[j] + _span[j] * float(np.random.rand()) for j in range(STATE_DIM)]
         for _ in range(NUM_BASIS)], dtype=np.float64)
    lengthscales_init = np.array([float(v) / 4.0 for v in _span], dtype=np.float64)

_u_max_base = 1.0 if ENFORCE_ACTION_LIMITS else U_MAX_FLAT
base_policy = Policy.Sum_of_gaussians(
    state_dim=STATE_DIM, input_dim=INPUT_DIM, num_basis=NUM_BASIS,
    u_max=_u_max_base,
    flg_squash=True, flg_drop=True,
    centers_init=centers_init,
    lengthscales_init=lengthscales_init,
    weight_init=0.1 * np.random.randn(INPUT_DIM, NUM_BASIS),
    dtype=dtype, device=device)
policy = BoundedPolicy(base_policy, U_MAX_Z) if ENFORCE_ACTION_LIMITS else base_policy
if _warm is not None:
    policy.load_state_dict(_warm["policy_state_dict"])
    print(f"[loop] loaded policy weights (previous final W2 = "
          f"{_warm.get('hist', [float('nan')])[-1]:.4f})")

print(f"\npolicy {type(base_policy).__name__}"
      f"{' (per-channel bounded)' if ENFORCE_ACTION_LIMITS else f' (flat u_max={U_MAX_FLAT})'}: "
      f"in={STATE_DIM} out={INPUT_DIM} "
      f"params={sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
print(f"  centres span per-dim [{centers_init.min():.2f}, {centers_init.max():.2f}] "
      f"(data [{s_lo.min().item():.2f}, {s_hi.max().item():.2f}])")
print(f"  lengthscales: {np.round(lengthscales_init, 3)}")
print(f"EPISODIC: T={STEPS_PER_EXPERT} steps ({EXPERT_DT} h) per window, "
      f"{WINDOWS_PER_ITER} windows/iter x {N_ITERS} iters = "
      f"{WINDOWS_PER_ITER*N_ITERS} policy updates")
print(f"objective: E_s(E_a|s(W2))  states={NUM_STATES} x actions={K_ACTIONS} "
      f"= {NUM_PARTICLES} particles;  means EXCLUDED (cross_dim)")
print(f"world models: {'PHASE-SPECIFIC (3)' if USE_PHASE_MODELS else 'SINGLE (all data)'}")

# sanity: replicas of one state must get DIFFERENT actions or E_{a|s} is degenerate
with torch.no_grad():
    _st = POOLS[sorted(POOLS)[0]][:1].expand(K_ACTIONS, -1).contiguous()
    _sp = policy(states=_st, t=0, p_dropout=P_DROPOUT).std(0).mean().item()
print(f"action spread across {K_ACTIONS} replicas of ONE state: {_sp:.3e}"
      f"{'   <-- WARNING: ~0 means E_a|s is degenerate' if _sp < 1e-4 else '   (ok)'}")

if _args.iters:
    N_ITERS = _args.iters
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


# =====================================================================================
# 4. WINDOW LOSS
# =====================================================================================
_acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}
_current_eig = None


def window_loss(t, s, a, mu, cov, s_next):
    global _acc
    if _acc["mean"] is None:
        _acc = {"mean": torch.zeros_like(mu), "var": torch.zeros_like(cov),
                "s_start": s, "t_start": t}
    _acc["mean"] = _acc["mean"] + (mu - s)
    _acc["var"] = _acc["var"] + cov

    if (t - _acc["t_start"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=mu.dtype, device=mu.device)

    var_1h = _acc["var"]
    _acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}

    d_all = w2_cross_dim_torch(var_1h, _current_eig)               # (P,)
    return d_all.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()    # E_a|s then E_s


# =====================================================================================
# 5. TRAINING LOOP
# =====================================================================================
hist = []
for it in range(N_ITERS):
    order = rng.permutation(len(EXPERT_TIMES))[:WINDOWS_PER_ITER]
    losses, gnorms, smax, amax = [], [], [], []
    per_phase = {p: [] for p in MODELS}

    for idx in order:
        t_h = float(EXPERT_TIMES[idx])
        _current_eig = EXPERT_EIGS[round(t_h, 6)]

        # the expert time selects the world model AND the start-state pool, so the
        # particles begin inside the region that model was trained on
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
    print("          per-phase W2: " + "  ".join(
        f"ph{p}={np.mean(v):.4e}(n={len(v)})" for p, v in sorted(per_phase.items()) if v),
        flush=True)

    if it % 5 == 0:
        torch.save({"policy_state_dict": policy.state_dict(), "iter": it,
                    "loss": float(L.mean()), "hist": hist,
                    "u_max_z": U_MAX_Z, "centers_init": centers_init,
                    "use_phase_models": USE_PHASE_MODELS},
                   os.path.join(SAVE_DIR, f"cdil_policy_it{it}.pt"))

os.makedirs(SAVE_DIR, exist_ok=True)
_tag = "phasemodels" if USE_PHASE_MODELS else "allmodel"
_out = _args.out or os.path.join(SAVE_DIR, f"cdil_policy_{_tag}.pt")
torch.save({"policy_state_dict": policy.state_dict(), "hist": hist,
            "u_max_z": U_MAX_Z, "centers_init": centers_init,
            "lengthscales_init": lengthscales_init,
            "use_phase_models": USE_PHASE_MODELS,
            "model_paths": _paths,
            "std_act_mu": stats["std_act_mu"], "std_act_sd": stats["std_act_sd"],
            "std_obs_mu": stats["std_obs_mu"], "std_obs_sd": stats["std_obs_sd"]}, _out)
print(f"\nsaved -> {_out}")
