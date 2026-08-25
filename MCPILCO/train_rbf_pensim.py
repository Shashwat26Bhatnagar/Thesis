#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

_p = argparse.ArgumentParser("train a (phase-specific) PenSim world model")
_p.add_argument("-phase", type=int, default=-1,
                help="0,1,2: the phases set by PENSIM_PHASE_BOUNDS   -1: all data")
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
_p.add_argument("-save_dir", type=str, default="results_pensim",
                help="where the model, metrics and stats are written")
_p.add_argument("-std_from", type=str, default=None,
                help="checkpoint whose std_obs_*/std_act_* stats to REUSE. Required from "
                     "the z-space is pinned instead of refitted. Without it the "
                     "statistics follow whatever is in the data folder at that moment, "
                     "and the folder grows as exploration adds CSVs.")
_args = _p.parse_known_args()[0]

if _args.data_dir:
    os.environ["PENSIM_DATA_DIR"] = _args.data_dir

PHASE = _args.phase
TAG = _args.tag or pdata.phase_tag(PHASE)
SELECT_MODE = _args.select
N_KEEP = _args.n_keep
N_EPOCH = _args.n_epoch
N_TEST = 200

STATE_DIM = pdata.OBS_DIM
INPUT_DIM = pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
SAVE_DIR = _args.save_dir
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=== training world model: phase={PHASE} ({TAG})  "
      f"n_keep={N_KEEP}  n_epoch={N_EPOCH}  select={SELECT_MODE} ===")

_std_obs_stats = _std_act_stats = None
if _args.std_from:
    _ck = torch.load(_args.std_from, map_location="cpu", weights_only=False)
    _std_obs_stats = (np.asarray(_ck["std_obs_mu"]), np.asarray(_ck["std_obs_sd"]))
    _std_act_stats = (np.asarray(_ck["std_act_mu"]), np.asarray(_ck["std_act_sd"]))
    print(f"[loop] reusing frozen standardizer stats from {_args.std_from}")

obs, act, nobs, std_obs, std_act, t_h = pdata.load_offline(
    max_transitions=None, return_time=True,
    std_obs_stats=_std_obs_stats, std_act_stats=_std_act_stats)
N_ALL = obs.shape[0]
print(f"[std] fitted on {N_ALL} transitions ({N_ALL/1150:.2f} CSVs)"
      f"{'  [PINNED via -std_from]' if _args.std_from else '  [freshly fitted]'}")
print(f"[std] obs_sd = {np.round(std_obs.sd, 5)}")
print(f"[std] act_sd = {np.round(std_act.sd, 5)}")

obs, act, nobs, t_h = pdata.filter_by_phase(obs, act, nobs, t_h, PHASE)
N = obs.shape[0]
if N < N_TEST + 50:
    raise RuntimeError(f"phase {PHASE} has only {N} transitions -- too few to train")

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

init_dict = dict(
    active_dims=np.arange(0, GP_INPUT_DIM),
    lengthscales_init=np.ones(GP_INPUT_DIM),
    flg_train_lengthscales=True,
    lambda_init=np.ones(1), flg_train_lambda=True,
    sigma_n_init=1e-2 * np.ones(1),
    sigma_n_num=1e-4,
    flg_train_sigma_n=True,
    dtype=dtype, device=device,
)

model = ML.Model_learning_RBF(
    num_gp=STATE_DIM,
    init_dict_list=[dict(init_dict) for _ in range(STATE_DIM)],
    approximation_mode=None,
    dtype=dtype, device=device, flg_norm=False,
)
model.gp_inputs = X
model.gp_output_list = [DY[:, i].reshape(-1, 1) for i in range(STATE_DIM)]
model.num_samples = X.shape[0]
model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM

opt = dict(f_optimizer="lambda p : torch.optim.Adam(p, lr=0.01)",
           criterion=Likelihood.Marginal_log_likelihood,
           N_epoch=N_EPOCH, N_epoch_print=max(1, N_EPOCH // 2))
model.reinforce_model([dict(opt) for _ in range(STATE_DIM)])
print("\nTrained hyperparameters:"); model.print_model()

model.set_eval_mode()
with torch.no_grad():
    mean_list, var_list = model.get_gp_estimate(gp_inputs=X_te,
                                                gp_index_list=range(STATE_DIM))
pred_delta = torch.cat(mean_list, 1)
pred_var   = torch.cat([v.reshape(-1, 1) for v in var_list], 1)

cur_state  = torch.tensor(obs_te,  dtype=dtype, device=device)
true_next  = torch.tensor(nobs_te, dtype=dtype, device=device)
pred_next  = cur_state + pred_delta

mse  = torch.mean((pred_next - true_next) ** 2, dim=0)
base = torch.mean((cur_state - true_next) ** 2, dim=0)

print(f"\nheld-out NEXT-STATE MSE (model units) -- phase {PHASE} ({TAG}):")
for i, nm in enumerate(pdata.OBS_NAMES):
    tag = "OK" if mse[i] < base[i] else "WORSE"
    print(f"  {i} {nm:6s} model={mse[i].item():.6e}  persistence={base[i].item():.6e}  {tag}")
print(f"\nmean MSE {mse.mean().item():.6e}   persistence {base.mean().item():.6e}")
print("model beats persistence:", bool(mse.mean() < base.mean()))

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
    "phase": PHASE, "phase_tag": TAG,
    "phase_t_lo": float(pdata.PHASES[PHASE][0]),
    "phase_t_hi": float(pdata.PHASES[PHASE][1]),
    "train_t_min": float(t_s.min()) if len(t_s) else None,
    "train_t_max": float(t_s.max()) if len(t_s) else None,
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
