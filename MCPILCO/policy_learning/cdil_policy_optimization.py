#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cdil_policy_optimization.py   (MC-PILCO repo root)

Cross-Domain Imitation Learning (CDIL) policy optimization.

Per time step t:
    1. POLICY      a_t = pi(s_t)
    2. GP TRANSITION  P(s'|s_t,a_t) = N(mu_t, diag(cov_t))   [reparam trick, reused]
    3. LOSS        per-step CDIL loss vs the expert distribution   [TODO]
    4. BACKPROP    graph-managed update                            [TODO]

The rollout mechanics live in policy_learning/gp_particle_rollout.py and should not
need editing. Everything you will replace is fenced with a TODO marker below.

=== THE CROSS-DOMAIN GAP (read this before writing the loss) ===
The two distributions live in DIFFERENT SPACES, which is what makes this "cross
domain". They cannot be compared elementwise without an explicit map:

    GP world model      8-D  [pH, T, Fa, Fb, Fc, Fh, Wt, DO2]
                        DIAGONAL covariance (8 independent scalar GPs)
                        z-scored on smpl-normalized PenSim units
                        step = 12 min, horizon ~230 h

    FL-GFN expert       3-D  (S g/L, X g, V L)
                        FULL 3x3 covariance
                        physical units
                        DT = 1.0 h, horizon 150 h  (dFBA fed-batch, not PenSim)

Three things must therefore be decided by you, each marked TODO below:
    - PROJECTION : 8-D model state -> 3-D expert space
    - TIME MAP   : rollout step index -> expert time argument t [h]
    - LOSS       : how to compare N(mu_p, cov_p) with N(b, Sigma)

Note the expert is NON-differentiable (torch.no_grad + numpy inside PFQuery), which
is correct -- it is the target. Gradients flow only through the GP/policy side.
"""
import os
import sys
import numpy as np
import torch

# expert repo is a sibling clone; no setup.py, so add its root to sys.path
# (PEP 420 namespace packages: no __init__.py needed)
EXPERT_REPO = os.path.expanduser("~/penicillin-dcfba")
if EXPERT_REPO not in sys.path:
    sys.path.insert(0, EXPERT_REPO)

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
import policy_learning.Policy as Policy
from policy_learning.gp_particle_rollout import (gp_rollout, sample_initial_particles,
                                                 rollout_step_stats, cov_full)
from policy_learning.wasserstein_loss import cdil_w2_loss
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

MODEL_PATH = "results_pensim/rbf_model.pt"
STATE_DIM = pdata.OBS_DIM          # 8
INPUT_DIM = pdata.ACT_DIM          # 6
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
EXPERT_DIM = 3                     # (S, X, V)

NUM_PARTICLES = 200
T_ROLLOUT = 40
N_ITERS = 300
LR = 0.01
P_DROPOUT = 0.25

# graph handling: "full" (one backward after the rollout) | "truncated" | "per_step"
GRAPH_MODE = "full"
TRUNCATE_EVERY = 10


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
    # cached posterior pieces: get_estimate_from_alpha cannot run without these
    model.gp_inputs = ckpt["gp_inputs"]
    model.gp_output_list = ckpt["gp_output_list"]
    model.alpha_list = ckpt["alpha_list"]
    model.m_X_list = ckpt["m_X_list"]
    model.K_X_inv_list = ckpt["K_X_inv_list"]
    model.gp_inputs_tr_list = ckpt["gp_inputs_tr_list"]
    model.num_samples = ckpt["gp_inputs"].shape[0]
    model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM
    model.norm_list = [1.0] * STATE_DIM     # get_next_state scales var by norm**2
    stats = {k: np.asarray(ckpt[k]) for k in
             ("std_obs_mu", "std_obs_sd", "std_act_mu", "std_act_sd")}
    return model, stats


model, stats = load_rbf_model()
# freeze: requires_grad=False on GP hyperparameters. Does NOT block gradients
# w.r.t. the GP input, which is what the policy gradient needs.
model.set_eval_mode()
print(f"frozen GP model: num_gp={model.num_gp}  train pts={model.gp_inputs.shape[0]}")


# =====================================================================================
# 2. EXPERT ORACLE  (non-differentiable target)
# =====================================================================================
class ExpertOracle:
    """Thin cache around PFQuery.next_state_distribution.

    Returns torch tensors built with torch.tensor(<python list>) rather than
    torch.from_numpy: the latter shares the numpy buffer and trips the duplicate-
    numpy ABI mismatch present in this environment.
    """

    def __init__(self, source="traj", verbose=True):
        self.q = PFQuery(verbose=verbose)
        self.source = source
        self._cache = {}

    def at_time(self, t_hours):
        """Expert P_F(s'|s(t)) at absolute time t [h]. Cached: deterministic in t."""
        key = round(float(t_hours), 6)
        if key not in self._cache:
            d = self.q.next_state_distribution(t=key, source=self.source)
            self._cache[key] = {
                "a": torch.tensor(np.asarray(d["a"]).tolist(), dtype=dtype, device=device),
                "b": torch.tensor(np.asarray(d["b"]).tolist(), dtype=dtype, device=device),
                "cov": torch.tensor(np.asarray(d["cov"]).tolist(), dtype=dtype, device=device),
                "sigma": torch.tensor(np.asarray(d["sigma"]).tolist(), dtype=dtype, device=device),
                "raw": d,        # keep pdf/logpdf/sample callables available
            }
        return self._cache[key]

    def at_state(self, s_expert_np, t_hours=0.0):
        """Expert distribution conditioned on a GIVEN 3-D state (bypasses the time
        lookup) -- the natural hook when the state comes from our rollout."""
        d = self.q.next_state_distribution(t=t_hours, state=np.asarray(s_expert_np, float))
        return {
            "a": torch.tensor(np.asarray(d["a"]).tolist(), dtype=dtype, device=device),
            "b": torch.tensor(np.asarray(d["b"]).tolist(), dtype=dtype, device=device),
            "cov": torch.tensor(np.asarray(d["cov"]).tolist(), dtype=dtype, device=device),
            "raw": d,
        }


expert = ExpertOracle(source="traj")
_probe = expert.at_time(75.0)
print(f"expert probe t=75h: a={_probe['a'].numpy()}  b={_probe['b'].numpy()}")
print(f"expert cov diag  : {torch.diagonal(_probe['cov']).numpy()}")


# =====================================================================================
# TODO: CDIL State Projection      8-D model state  ->  3-D expert space (S, X, V)
# -------------------------------------------------------------------------------------
# The model's channels [pH, T, Fa, Fb, Fc, Fh, Wt, DO2] contain NO substrate or
# biomass measurement, so there is no identity map. Options: a fixed linear readout,
# a fitted regressor from paired data, or restricting the loss to whatever subspace
# genuinely corresponds. Must be DIFFERENTIABLE (torch ops only) -- it sits on the
# gradient path from the loss back to the policy.
#
# Returned units must match the expert's PHYSICAL units (S g/L, X g, V L).
# =====================================================================================
_PROJ = torch.zeros(STATE_DIM, EXPERT_DIM, dtype=dtype, device=device)
_PROJ[6, 1] = 1.0        # PLACEHOLDER: vessel weight -> biomass axis (NOT correct)


def project_to_expert(s_model):
    """(P, 8) model state -> (P, 3) expert space. PLACEHOLDER."""
    return s_model @ _PROJ


def project_cov_to_expert(cov_diag_model):
    """(P, 8) diagonal model covariance -> (P, 3, 3) in expert space.
    For a linear map M: Cov_e = M^T diag(cov) M."""
    C = torch.diag_embed(cov_diag_model)                      # (P, 8, 8)
    M = _PROJ.unsqueeze(0).expand(C.shape[0], -1, -1)         # (P, 8, 3)
    return M.transpose(1, 2) @ C @ M                          # (P, 3, 3)


# =====================================================================================
# TIME ALIGNMENT   -- the two models predict over DIFFERENT horizons
# -------------------------------------------------------------------------------------
#   GP  P(s'|s,a) : one PenSim step = 12 min = 0.2 h
#   expert P_F(s'|s): pf_query.DT = 1.0 h   (t_next = t + DT; fdrift was TRAINED on
#                     1-hour transitions, so the horizon is baked into the network --
#                     querying at finer t does NOT shorten it)
#
# Comparing one GP step against one expert step would pit a 0.2 h prediction against a
# 1.0 h one, so the expert drift is ~5x larger purely from the time span. We therefore
# accumulate STEPS_PER_EXPERT = 5 GP steps into a single 1-hour transition and compare
# that against one expert step.
#
# Aggregation over the window (first order, differentiable):
#     mean : mu_1h  = s_start + sum_k delta_mean_k
#     var  : var_1h = sum_k delta_var_k        (per-step noise treated as independent)
# =====================================================================================
T_START_HOURS = 0.0
HOURS_PER_STEP = 0.2                                   # PenSim sampling interval [h]
EXPERT_DT = 1.0                                        # pf_query.DT [h]
STEPS_PER_EXPERT = int(round(EXPERT_DT / HOURS_PER_STEP))     # = 5


def step_to_hours(t_step):
    return T_START_HOURS + t_step * HOURS_PER_STEP


# =====================================================================================
# 3. POLICY   (REUSED -- confirm the class with: grep -n '^class' policy_learning/Policy.py)
# =====================================================================================
num_basis = 200
policy_par = dict(
    state_dim=STATE_DIM,                 # MLP/basis input  == model state dim
    input_dim=INPUT_DIM,                 # output           == action dim
    num_basis=num_basis, u_max=3.0, flg_squash=True, flg_drop=True,
    centers_init=np.random.randn(num_basis, STATE_DIM),
    lengthscales_init=np.ones(STATE_DIM),
    weight_init=0.1 * np.random.randn(INPUT_DIM, num_basis),
    dtype=dtype, device=device,
)
policy = Policy.Sum_of_gaussians(**policy_par)
assert policy.state_dim == STATE_DIM and policy.input_dim == INPUT_DIM
print(f"policy {type(policy).__name__}: in={STATE_DIM} out={INPUT_DIM} "
      f"params={sum(p.numel() for p in policy.parameters() if p.requires_grad)}")
print(f"time alignment: {STEPS_PER_EXPERT} GP steps ({HOURS_PER_STEP} h each) = "
      f"1 expert step ({EXPERT_DT} h) -> {T_ROLLOUT // STEPS_PER_EXPERT} comparisons "
      f"over {T_ROLLOUT * HOURS_PER_STEP:.1f} h;  W2 mode = {W2_MODE!r}")

optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
state_pool = model.gp_inputs[:, :STATE_DIM]
rng = np.random.default_rng(0)


# =====================================================================================
# CDIL LOSS  --  2-Wasserstein between the GP transition Gaussian and the expert's
# -------------------------------------------------------------------------------------
# W2_MODE:
#   "cross_dim"  Cai-Lim projection distance, 8-D vs 3-D directly. NO projection matrix
#                needed -- BUT the MEANS DROP OUT (they are absorbed by a free
#                translation), so this matches only the covariance spectrum. The policy
#                gets no signal about WHERE the trajectory goes.
#   "projected"  project the model to 3-D, then Bures-Wasserstein. Means matter, but
#                requires a correct project_to_expert() (still a modelling decision).
#   "hybrid"     cross-dim covariance term + explicit projected mean term.
#
# Called every 12 min; emits a real loss only on the 5th step of each window, when the
# accumulated GP transition spans the same 1 h as one expert step.
# =====================================================================================
W2_MODE = "cross_dim"
MEAN_WEIGHT = 1.0

_acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}


def cdil_step_loss(t, s, a, mu, cov, s_next):
    global _acc
    if _acc["mean"] is None:                       # open a new 1-hour window
        _acc = {"mean": torch.zeros_like(mu), "var": torch.zeros_like(cov),
                "s_start": s, "t_start": t}

    _acc["mean"] = _acc["mean"] + (mu - s)         # sum per-step delta means
    _acc["var"] = _acc["var"] + cov                # sum per-step variances

    if (t - _acc["t_start"] + 1) < STEPS_PER_EXPERT:
        return torch.zeros((), dtype=mu.dtype, device=mu.device)   # not a checkpoint

    # --- aggregated 1-hour GP transition distribution ---
    mu_1h = _acc["s_start"] + _acc["mean"]         # (P, 8)
    var_1h = _acc["var"]                           # (P, 8) diagonal
    t_hours = step_to_hours(_acc["t_start"])
    _acc = {"mean": None, "var": None, "s_start": None, "t_start": 0}

    e = expert.at_time(t_hours)                    # expert 1-h step from the same time

    return cdil_w2_loss(
        mu_model=mu_1h, cov_model_diag=var_1h,
        expert_b=e["b"], expert_cov=e["cov"],
        mode=W2_MODE,
        project_mu=project_to_expert,
        project_cov=project_cov_to_expert,
        mean_weight=MEAN_WEIGHT,
    )


# =====================================================================================
# TODO: Per-Step Backprop / Graph Management
# -------------------------------------------------------------------------------------
# flush_fn is called by the rollout at graph boundaries (only in "truncated"/
# "per_step" mode). In "full" mode it is never called and the single backward below
# runs after the rollout. Put clipping / schedules / accumulation here.
# =====================================================================================
def flush_fn(loss_segment):
    optimizer.zero_grad()
    loss_segment.backward()
    torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
    optimizer.step()


for it in range(N_ITERS):
    s0 = sample_initial_particles(state_pool, NUM_PARTICLES, generator=rng,
                                  dtype=dtype, device=device)

    out = gp_rollout(
        model=model, policy=policy, s0=s0, T=T_ROLLOUT,
        p_dropout=P_DROPOUT, particle_pred=True,
        loss_fn=cdil_step_loss,          # per-step CDIL loss
        flush_fn=flush_fn,               # used only in truncated / per_step modes
        graph_mode=GRAPH_MODE,
        truncate_every=TRUNCATE_EVERY,
    )

    # ---- TODO: Per-Step Backprop / Graph Management (full-mode branch) ----
    if GRAPH_MODE == "full" and out["loss_total"] is not None:
        optimizer.zero_grad()
        out["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
        optimizer.step()
    # ----------------------------------------------------------------------

    if it % 20 == 0:
        with torch.no_grad():
            total = torch.stack([l.detach() for l in out["step_losses"]]).sum().item()
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                                   if p.grad is not None)).item()
            _, _, pstd = rollout_step_stats(out["Dmean"], out["Dvar"])
        print(f"iter {it:4d}  sum step-loss={total:.6e}  |grad|={gnorm:.3e}  "
              f"GP std step0={pstd[0].mean():.3e} -> step{T_ROLLOUT-1}={pstd[-1].mean():.3e}")

torch.save({"policy_state_dict": policy.state_dict()},
           "results_pensim/cdil_policy.pt")
print("\nsaved -> results_pensim/cdil_policy.pt")
