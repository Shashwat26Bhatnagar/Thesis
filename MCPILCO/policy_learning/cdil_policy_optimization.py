#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/cdil_policy_optimization.py

Cross-Domain Imitation Learning (CDIL) policy optimization -- EPISODIC.

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

OBJECTIVE:  E_s( E_{a|s}( W2 ) )
    NUM_STATES start states, each replicated K_ACTIONS times; replicas share a state
    but draw independent actions (dropout), so the inner mean is E_{a|s} and the outer
    mean is E_s. W2 is the Cai-Lim cross-dimensional distance (8-D model vs 3-D
    expert): no projection needed, and the MEANS DROP OUT by construction --
    a deliberate choice; the objective matches covariance spectra only.

POLICY:
    RBF centres are spread over the OBSERVED state range instead of randn's [-3, 3];
    training states span [-5.5, 11.2], so randn centres under-cover it.

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
MODEL_PATH = os.path.join(SAVE_DIR, "rbf_model.pt")

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
# 1. FROZEN GP WORLD MODEL
# =====================================================================================
def load_rbf_model(path=MODEL_PATH):
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
    return model, stats


model, stats = load_rbf_model()
model.set_eval_mode()
state_pool = model.gp_inputs[:, :STATE_DIM]
s_lo = state_pool.min(0).values
s_hi = state_pool.max(0).values
print(f"frozen GP model: num_gp={model.num_gp}  train pts={model.gp_inputs.shape[0]}")
print(f"training state range: [{s_lo.min().item():.2f}, {s_hi.max().item():.2f}] (z-units)")


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

# sanity: replicas of one state must get DIFFERENT actions or E_{a|s} is degenerate
with torch.no_grad():
    _st = state_pool[:1].expand(K_ACTIONS, -1).contiguous()
    _sp = policy(states=_st, t=0, p_dropout=P_DROPOUT).std(0).mean().item()
print(f"action spread across {K_ACTIONS} replicas of ONE state: {_sp:.3e}"
      f"{'   <-- WARNING: ~0 means E_a|s is degenerate' if _sp < 1e-4 else '   (ok)'}")

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

    for idx in order:
        t_h = float(EXPERT_TIMES[idx])
        _current_eig = EXPERT_EIGS[round(t_h, 6)]

        s_states = sample_initial_particles(state_pool, NUM_STATES, generator=rng,
                                            dtype=dtype, device=device)
        s0 = s_states.repeat_interleave(K_ACTIONS, dim=0)

        out = gp_rollout(model=model, policy=policy, s0=s0, T=STEPS_PER_EXPERT,
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

    if it % 5 == 0:
        torch.save({"policy_state_dict": policy.state_dict(), "iter": it,
                    "loss": float(L.mean()), "hist": hist,
                    "u_max_z": U_MAX_Z, "centers_init": centers_init},
                   os.path.join(SAVE_DIR, f"cdil_policy_it{it}.pt"))

os.makedirs(SAVE_DIR, exist_ok=True)
_out = os.path.join(SAVE_DIR, "cdil_policy.pt")
torch.save({"policy_state_dict": policy.state_dict(), "hist": hist,
            "u_max_z": U_MAX_Z, "centers_init": centers_init,
            "std_act_mu": stats["std_act_mu"], "std_act_sd": stats["std_act_sd"],
            "std_obs_mu": stats["std_obs_mu"], "std_obs_sd": stats["std_obs_sd"]}, _out)
print(f"\nsaved -> {_out}")
