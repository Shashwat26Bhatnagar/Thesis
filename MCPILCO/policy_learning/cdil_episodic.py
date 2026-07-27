#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/cdil_episodic.py

CDIL policy optimization, restructured as EPISODIC short-horizon windows --
mirroring MC-PILCO's actual scheme rather than one long continuing rollout.

WHY (from the MC-PILCO paper, Amadio et al. 2022):
    J(theta) = sum_{t=0..T} (1/M) sum_m c(x_t^(m)),  ONE backward, ONE update.
    Their horizons are SHORT: cart-pole 3 s / 0.05 s = 60 steps; UR5 200; Furuta 90.
    And p(x_0) is RESAMPLED at every optimization step -- the trajectory never
    continues past T.

    Our previous scheme rolled 750 steps CONTINUING (no reset), so |s| reached ~140
    z-units against training data spanning [-5.5, 11.2]. That killed the gradient in
    two places at once:
        - GP predictive variance saturated at the prior lambda  -> dvar/dinput = 0
        - policy RBF basis abandoned its centres ([-3, 3])      -> da/dtheta   = 0

HERE: each expert comparison window (k = STEPS_PER_EXPERT = 5 steps = 1 h) is a
self-contained episode with T = 5 and a FRESH in-distribution start state, exactly
analogous to one MC-PILCO optimization step. Same total compute, but every window
begins inside the region the world model actually covers.
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
NUM_STATES = 100                   # outer expectation E_s
K_ACTIONS = 5                      # inner expectation E_{a|s}
NUM_PARTICLES = NUM_STATES * K_ACTIONS

# --- EPISODIC structure: each window is its own short-horizon MC-PILCO episode ---
HOURS_PER_STEP = 0.2               # PenSim sampling interval [h]
EXPERT_DT = 1.0                    # pf_query.DT [h]
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))     # = 5 -> the episode length T
EXPERT_T_MIN, EXPERT_T_MAX = 1.0, 150.0                       # valid range of source="traj"
WINDOWS_PER_ITER = 150             # expert times visited per iteration
N_ITERS = 20
LR = 0.01
P_DROPOUT = 0.25
CLIP = 10.0

EXPERT_COV_KEY = "cov_n"


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
    return model


model = load_rbf_model()
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

# pre-cache every expert time we will use (deterministic in t; avoids repeated queries)
EXPERT_TIMES = np.arange(EXPERT_T_MIN, EXPERT_T_MAX + 1e-9, EXPERT_DT)
print(f"pre-caching {len(EXPERT_TIMES)} expert distributions "
      f"({EXPERT_T_MIN}..{EXPERT_T_MAX} h) ...", flush=True)
EXPERT_EIGS = {}
for _t in EXPERT_TIMES:
    EXPERT_EIGS[round(float(_t), 6)] = torch.linalg.eigvalsh(
        expert.at_time(_t)[EXPERT_COV_KEY].detach())
print(f"  done. example eigenvalues @75h: "
      f"{EXPERT_EIGS[75.0].numpy()}", flush=True)


# =====================================================================================
# 3. POLICY
# =====================================================================================
num_basis = 200
policy = Policy.Sum_of_gaussians(
    state_dim=STATE_DIM, input_dim=INPUT_DIM, num_basis=num_basis,
    u_max=3.0, flg_squash=True, flg_drop=True,
    centers_init=np.random.randn(num_basis, STATE_DIM),
    lengthscales_init=np.ones(STATE_DIM),
    weight_init=0.1 * np.random.randn(INPUT_DIM, num_basis),
    dtype=dtype, device=device)
print(f"policy {type(policy).__name__}: in={STATE_DIM} out={INPUT_DIM} "
      f"params={sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
print(f"EPISODIC: T={STEPS_PER_EXPERT} steps ({EXPERT_DT} h) per window, "
      f"{WINDOWS_PER_ITER} windows/iter x {N_ITERS} iters = "
      f"{WINDOWS_PER_ITER*N_ITERS} policy updates")
print(f"objective: E_s(E_a|s(W2))  states={NUM_STATES} x actions={K_ACTIONS} "
      f"= {NUM_PARTICLES} particles;  means EXCLUDED (cross_dim)")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)


# =====================================================================================
# 4. WINDOW LOSS  -- one 1-hour episode
# =====================================================================================
_acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}
_current_eig = None


def window_loss(t, s, a, mu, cov, s_next):
    """Accumulate STEPS_PER_EXPERT steps, then compare to the expert for this window."""
    global _acc
    if _acc["mean"] is None:
        _acc = {"mean": torch.zeros_like(mu), "var": torch.zeros_like(cov),
                "s_start": s, "t_start": t}
    _acc["mean"] = _acc["mean"] + (mu - s)
    _acc["var"] = _acc["var"] + cov

    if (t - _acc["t_start"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=mu.dtype, device=mu.device)

    var_1h = _acc["var"]                                   # (P, 8) diagonal
    _acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}

    d_all = w2_cross_dim_torch(var_1h, _current_eig)       # (P,)
    return d_all.view(NUM_STATES, K_ACTIONS).mean(dim=1).mean()   # E_a|s then E_s


# =====================================================================================
# 5. TRAINING LOOP  -- episodic, MC-PILCO style
# =====================================================================================
hist = []
for it in range(N_ITERS):
    order = rng.permutation(len(EXPERT_TIMES))[:WINDOWS_PER_ITER]
    losses, gnorms, smax = [], [], []

    for w, idx in enumerate(order):
        t_h = float(EXPERT_TIMES[idx])
        _current_eig = EXPERT_EIGS[round(t_h, 6)]

        # FRESH in-distribution start state for THIS window (MC-PILCO resamples p(x0))
        s_states = sample_initial_particles(state_pool, NUM_STATES, generator=rng,
                                            dtype=dtype, device=device)
        s0 = s_states.repeat_interleave(K_ACTIONS, dim=0)

        # short-horizon episode: T = 5, ONE graph, ONE backward
        out = gp_rollout(model=model, policy=policy, s0=s0, T=STEPS_PER_EXPERT,
                         p_dropout=P_DROPOUT, particle_pred=True,
                         loss_fn=window_loss, graph_mode="full")

        loss = out["loss_total"]
        optimizer.zero_grad()
        loss.backward()
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                            if p.grad is not None)).item()          # BEFORE clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), CLIP)
        optimizer.step()

        losses.append(loss.item()); gnorms.append(gn)
        smax.append(out["S"].detach().abs().max().item())

    L, G, S = np.array(losses), np.array(gnorms), np.array(smax)
    hist.append(L.mean())
    print(f"iter {it:3d}  W2 mean={L.mean():.6e} min={L.min():.4e} max={L.max():.4e}", flush=True)
    print(f"          |grad| med={np.median(G):.3e} max={G.max():.3e}  "
          f"DEAD(<1e-12)={int((G < 1e-12).sum())}/{len(G)}  "
          f"CLIPPED={int((G > CLIP).sum())}/{len(G)}", flush=True)
    print(f"          |s|max over windows: med={np.median(S):.2f} max={S.max():.2f}  "
          f"(training data max {s_hi.max().item():.2f})", flush=True)

    if it % 5 == 0:
        torch.save({"policy_state_dict": policy.state_dict(), "iter": it,
                    "loss": float(L.mean()), "hist": hist},
                   os.path.join(SAVE_DIR, f"cdil_ep_policy_it{it}.pt"))

os.makedirs(SAVE_DIR, exist_ok=True)
_out = os.path.join(SAVE_DIR, "cdil_ep_policy.pt")
torch.save({"policy_state_dict": policy.state_dict(), "hist": hist}, _out)
print(f"\nsaved -> {_out}")
