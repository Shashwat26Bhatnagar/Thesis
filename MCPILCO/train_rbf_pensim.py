#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_rbf_pensim.py   (repo root)   -- STEP 1 of the PenSim pipeline

Train an RBF Gaussian-process world model on the PenSim offline data and save it.
MODEL LEARNING ONLY.

    python train_rbf_pensim.py                # all data       -> rbf_model_all.pt
    python train_rbf_pensim.py -phase 0       # t <  35 h      -> rbf_model_phase0.pt
    python train_rbf_pensim.py -phase 1       # 35 <= t < 51 h -> rbf_model_phase1.pt
    python train_rbf_pensim.py -phase 2       # t >= 51 h      -> rbf_model_phase2.pt

    optional: -n_keep 1500  -n_epoch 2001  -select pivchol

PHASE-SPECIFIC MODELS
    Fermentation has distinct regimes (lag/growth, transition, production) and a
    single stationary RBF has to compromise across all of them. Training one model
    per phase lets each specialise.

    The Standardizer is fitted on the FULL dataset BEFORE filtering, so every phase
    model lives in the SAME z-space -- otherwise their outputs would not be
    comparable and a rollout could not switch between them.

    Time comes from raw observation column 0 (physical hours) and is used ONLY to
    segment the dataset. It is NOT a GP input: a time-indexed model would break the
    Markov assumption.

POINT SELECTION (-select)
    stride  : every Nth row. Fast, but keeps NEAR-DUPLICATES. In a narrow time
              window (phase 1 spans 16 h of 230 h) the process barely moves, so the
              kernel matrix becomes numerically rank-deficient and torch.cholesky
              fails with "leading minor of order 229 is not positive-definite".
    pivchol : greedy pivoted-Cholesky selection -- picks the points that add the most
              rank, skipping near-duplicates. Use this whenever stride hits the
              Cholesky error.

One independent scalar GP per state dimension (num_gp = 8), each mapping the 14-D
input [state(8), action(6)] to the delta of one state channel.
"""
import argparse
import os
import json
import numpy as np
import torch

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

# ------------------------------------------------------------------ args ----
_p = argparse.ArgumentParser("train a (phase-specific) PenSim world model")
_p.add_argument("-phase", type=int, default=-1,
                help="0: t<35h   1: 35<=t<51h   2: t>=51h   -1: all data (default)")
_p.add_argument("-n_keep", type=int, default=800,
                help="training points after subsampling (exact GP is O(N^3) per GP)")
_p.add_argument("-n_epoch", type=int, default=501,
                help="hyperparameter optimisation steps")
_p.add_argument("-select", type=str, default="stride", choices=["stride", "pivchol"],
                help="stride: every Nth row | pivchol: skip near-duplicates")
_p.add_argument("-data_dir", type=str, default=None,
                help="dataset folder (also settable via PENSIM_DATA_DIR)")
_p.add_argument("-tag", type=str, default=None,
                help="output tag; default is the phase tag. Dyna loop uses e.g. unb_iter2")
_p.add_argument("-std_from", type=str, default=None,
                help="checkpoint whose std_obs_*/std_act_* stats to REUSE. Required from "
                     "iteration 1 on: refitting them would drift the z-space and "
                     "invalidate a warm-started policy.")
_args = _p.parse_known_args()[0]

if _args.data_dir:
    os.environ["PENSIM_DATA_DIR"] = _args.data_dir

PHASE = _args.phase
TAG = _args.tag or pdata.phase_tag(PHASE)   # "all" | "phase0" | ... | "unb_iter2"
SELECT_MODE = _args.select         # "stride" | "pivchol"
N_KEEP = _args.n_keep
N_EPOCH = _args.n_epoch
N_TEST = 200

STATE_DIM = pdata.OBS_DIM          # 8 (time channel dropped)
INPUT_DIM = pdata.ACT_DIM          # 6
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
SAVE_DIR = "results_pensim"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=== training world model: phase={PHASE} ({TAG})  "
      f"n_keep={N_KEEP}  n_epoch={N_EPOCH}  select={SELECT_MODE} ===")

# ---------------------------------------------------------------- data ----
# standardizer is fitted on ALL data here, so every phase model shares one z-space
# The standardizer is refitted on the UNION of old + new data every iteration.
# (-std_from is accepted but ignored: the warm-started policy's RBF centres are
#  remapped into the new z-space by cdil_policy_optimization.py instead.)
obs, act, nobs, std_obs, std_act, t_h = pdata.load_offline(
    max_transitions=None, return_time=True)
N_ALL = obs.shape[0]               # total BEFORE phase filtering

# then restrict to the requested phase
obs, act, nobs, t_h = pdata.filter_by_phase(obs, act, nobs, t_h, PHASE)
N = obs.shape[0]
if N < N_TEST + 50:
    raise RuntimeError(f"phase {PHASE} has only {N} transitions -- too few to train")

# RANDOM held-out split within the phase (not the tail of one batch)
rng = np.random.default_rng(0)
perm = rng.permutation(N)
te_idx, tr_idx = perm[:N_TEST], perm[N_TEST:]
obs_te, act_te, nobs_te = obs[te_idx], act[te_idx], nobs[te_idx]
obs_tr, act_tr, nobs_tr = obs[tr_idx], act[tr_idx], nobs[tr_idx]
t_tr = t_h[tr_idx]

if SELECT_MODE == "pivchol":
    Z_tr = np.hstack([obs_tr, act_tr])
    sel = pdata.select_pivoted_cholesky(Z_tr, N_KEEP)
    obs_s, act_s, nobs_s, t_s = obs_tr[sel], act_tr[sel], nobs_tr[sel], t_tr[sel]
else:
    obs_s, act_s, nobs_s, t_s = pdata.subsample(obs_tr, act_tr, nobs_tr,
                                                n_keep=N_KEEP, t_hours=t_tr)

print(f"train N={obs_s.shape[0]}   test N={obs_te.shape[0]}   "
      f"gp input dim={GP_INPUT_DIM}   num_gp={STATE_DIM}")
if len(t_s):
    print(f"training-set time span: {t_s.min():.2f} .. {t_s.max():.2f} h")

X  = torch.tensor(np.hstack([obs_s, act_s]), dtype=dtype, device=device)
DY = torch.tensor(nobs_s - obs_s,            dtype=dtype, device=device)
X_te = torch.tensor(np.hstack([obs_te, act_te]), dtype=dtype, device=device)

# ------------------------------------------------------------ RBF config ----
init_dict = dict(
    active_dims=np.arange(0, GP_INPUT_DIM),
    lengthscales_init=np.ones(GP_INPUT_DIM),
    flg_train_lengthscales=True,
    lambda_init=np.ones(1), flg_train_lambda=True,
    sigma_n_init=1e-2 * np.ones(1),
    sigma_n_num=1e-4,                  # numerical noise floor: stops sigma_n -> 0
    flg_train_sigma_n=True,
    dtype=dtype, device=device,
)

model = ML.Model_learning_RBF(
    num_gp=STATE_DIM,
    init_dict_list=[dict(init_dict) for _ in range(STATE_DIM)],
    approximation_mode=None,
    dtype=dtype, device=device, flg_norm=False,
)
# inject the dataset directly: the base add_data() derives targets as s[1:] - s[:-1],
# which assumes consecutive rows -- false after subsampling/filtering.
model.gp_inputs = X
model.gp_output_list = [DY[:, i].reshape(-1, 1) for i in range(STATE_DIM)]
model.num_samples = X.shape[0]
model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM

# ---------------------------------------------------------------- train ----
opt = dict(f_optimizer="lambda p : torch.optim.Adam(p, lr=0.01)",
           criterion=Likelihood.Marginal_log_likelihood,
           N_epoch=N_EPOCH, N_epoch_print=max(1, N_EPOCH // 2))
model.reinforce_model([dict(opt) for _ in range(STATE_DIM)])
print("\nTrained hyperparameters:"); model.print_model()

# ------------------------------------- held-out NEXT-STATE prediction ----
model.set_eval_mode()
with torch.no_grad():
    mean_list, var_list = model.get_gp_estimate(gp_inputs=X_te,
                                                gp_index_list=range(STATE_DIM))
pred_delta = torch.cat(mean_list, 1)
pred_var   = torch.cat([v.reshape(-1, 1) for v in var_list], 1)

cur_state  = torch.tensor(obs_te,  dtype=dtype, device=device)
true_next  = torch.tensor(nobs_te, dtype=dtype, device=device)
pred_next  = cur_state + pred_delta                      # s_hat_{t+1} = s_t + delta_hat

mse  = torch.mean((pred_next - true_next) ** 2, dim=0)   # next-state MSE
base = torch.mean((cur_state - true_next) ** 2, dim=0)   # persistence: s_hat = s_t

print(f"\nheld-out NEXT-STATE MSE (model units) -- phase {PHASE} ({TAG}):")
for i, nm in enumerate(pdata.OBS_NAMES):
    tag = "OK" if mse[i] < base[i] else "WORSE"
    print(f"  {i} {nm:6s} model={mse[i].item():.6e}  persistence={base[i].item():.6e}  {tag}")
print(f"\nmean MSE {mse.mean().item():.6e}   persistence {base.mean().item():.6e}")
print("model beats persistence:", bool(mse.mean() < base.mean()))

# ----------------------------------------------------------------- save ----
# torch.save handles tensors natively; standardizer stats stored as plain lists
# to avoid the numpy-identity pickling error.
model_path   = os.path.join(SAVE_DIR, f"rbf_model_{TAG}.pt")
metrics_path = os.path.join(SAVE_DIR, f"rbf_metrics_{TAG}.json")

torch.save({
    "state_dict":        model.state_dict(),
    "init_dict_list":    [{k: (v.tolist() if isinstance(v, np.ndarray) else v)
                           for k, v in init_dict.items() if k not in ("dtype", "device")}
                          for _ in range(STATE_DIM)],
    "gp_inputs":         model.gp_inputs,
    "gp_output_list":    model.gp_output_list,
    "alpha_list":        model.alpha_list,
    "m_X_list":          model.m_X_list,
    "K_X_inv_list":      model.K_X_inv_list,
    "gp_inputs_tr_list": model.gp_inputs_tr_list,
    "num_gp": STATE_DIM, "state_dim": STATE_DIM, "input_dim": INPUT_DIM,
    "gp_input_dim": GP_INPUT_DIM, "n_keep": N_KEEP, "n_epoch": N_EPOCH,
    "select_mode": SELECT_MODE, "data_dir": os.environ.get("PENSIM_DATA_DIR"),
    "std_frozen": bool(_args.std_from),
    # phase provenance -- the policy needs this to pick the right model per window
    "phase": PHASE, "phase_tag": TAG,
    "phase_t_lo": float(pdata.PHASES[PHASE][0]),
    "phase_t_hi": float(pdata.PHASES[PHASE][1]),
    "train_t_min": float(t_s.min()) if len(t_s) else None,
    "train_t_max": float(t_s.max()) if len(t_s) else None,
    # standardizer stats are from the FULL dataset -> shared across phase models
    "std_obs_mu": std_obs.mu.tolist(), "std_obs_sd": std_obs.sd.tolist(),
    "std_act_mu": std_act.mu.tolist(), "std_act_sd": std_act.sd.tolist(),
    "obs_names": pdata.OBS_NAMES, "act_names": pdata.ACT_NAMES,
    "mse_per_dim": mse.tolist(), "persistence_per_dim": base.tolist(),
}, model_path)

with open(metrics_path, "w") as f:
    json.dump({
        "phase": PHASE, "phase_tag": TAG,
        "phase_t_lo": float(pdata.PHASES[PHASE][0]),
        "phase_t_hi": float(pdata.PHASES[PHASE][1]),
        "n_total_transitions_all_phases": int(N_ALL),
        "n_transitions_in_phase": int(N),
        "n_train": int(obs_s.shape[0]), "n_test": int(obs_te.shape[0]),
        "n_epoch": int(N_EPOCH), "select_mode": SELECT_MODE,
        "mse_per_dim":        {n: float(mse[i])  for i, n in enumerate(pdata.OBS_NAMES)},
        "persistence_per_dim":{n: float(base[i]) for i, n in enumerate(pdata.OBS_NAMES)},
        "mean_mse": float(mse.mean()), "mean_persistence": float(base.mean()),
        "beats_persistence": bool(mse.mean() < base.mean()),
        "std_obs_mu": std_obs.mu.tolist(), "std_obs_sd": std_obs.sd.tolist(),
        "std_act_mu": std_act.mu.tolist(), "std_act_sd": std_act.sd.tolist(),
    }, f, indent=2)

pdata.save_stats(os.path.join(SAVE_DIR, f"pensim_stats_{TAG}"), std_obs, std_act,
                 extra={"phase": PHASE, "n_keep": N_KEEP, "select_mode": SELECT_MODE,
                        "n_train": int(obs_s.shape[0])})

print(f"\nsaved -> {model_path}")
print(f"saved -> {metrics_path}")
