#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_reward_gp.py   (repo root)

Train a SEPARATE GP reward model  (state, action) -> yield-per-step, mirroring
train_rbf_pensim.py exactly: same 14-D input, same standardizer, same pivoted-Cholesky
selection, same marginal-likelihood fitting. The state world models are left untouched.

    python train_reward_gp.py                          # all data, 12 CSVs
    python train_reward_gp.py -data_dir <dir> -tag bnd_rbf_iter0

WHY A SEPARATE MODEL
The state models regress next_obs - obs (8 channels); reward is the CSV's 16th column
and was never a target. Rather than widening those models, this fits ONE scalar GP on
the same inputs, so the state models keep their held-out numbers and provenance.

THE error_reward ROWS ARE EXCLUDED
PenSim returns error_reward = -100 INSTEAD of a yield at termination. Older CSVs
recorded that row. Left in, the GP would learn that certain (state, action) pairs are
worth -100, and a reward-maximising policy would steer hard away from them for an
entirely spurious reason. Rows at or below ERROR_REWARD are therefore dropped, and the
count is reported.

USED WITH A LOWER CONFIDENCE BOUND
cdil_policy_optimization.py maximises mu - kappa*sigma, not mu. Maximising the mean of
a GP invites the policy into regions where the model is uncertain and optimistically
wrong -- the standard model-based RL failure, and one we have already seen here when
predictive variance saturated at the prior once particles left the data.
"""
import argparse
import json
import os

import numpy as np
import torch

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0)
torch.manual_seed(0)

ERROR_REWARD = -100.0

_p = argparse.ArgumentParser("train the GP reward model")
_p.add_argument("-data_dir", type=str, default=None,
                help="dataset folder (also settable via PENSIM_DATA_DIR)")
_p.add_argument("-tag", type=str, default="all", help="output tag")
_p.add_argument("-n_keep", type=int, default=800)
_p.add_argument("-n_epoch", type=int, default=2001)
_p.add_argument("-select", type=str, default="pivchol", choices=["stride", "pivchol"])
_args = _p.parse_known_args()[0]

if _args.data_dir:
    os.environ["PENSIM_DATA_DIR"] = _args.data_dir

TAG = _args.tag
N_KEEP, N_EPOCH, N_TEST = _args.n_keep, _args.n_epoch, 200
STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
SAVE_DIR = "results_pensim"
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"=== training REWARD model: tag={TAG}  n_keep={N_KEEP}  n_epoch={N_EPOCH}"
      f"  select={_args.select} ===")


# ------------------------------------------------------------------- data ----
# PeniControlData does not expose the reward column, so the CSVs are read directly.
# The standardizer is still fitted through load_offline so the GP input space is
# IDENTICAL to the state models' -- otherwise the two could not be used together.
import csv
import glob

folder = _args.data_dir or pdata.default_dataset_folder()
files = sorted(glob.glob(os.path.join(folder, "*.csv")))
if not files:
    raise FileNotFoundError(f"no CSVs in {folder}")

obs_p, act_p, rew_p = [], [], []
n_err = 0
for f in files:
    hdr = [h.strip() for h in next(csv.reader(open(f)))]
    d = np.atleast_2d(np.genfromtxt(f, delimiter=",", skip_header=1))
    if d.size == 0:
        continue
    yi = hdr.index("Yield Per Step")
    keep = d[:, yi] > ERROR_REWARD + 1e-9          # drop terminal error_reward rows
    n_err += int((~keep).sum())
    d = d[keep]
    obs_p.append(d[:, 7:15])                       # 8 observations (time already out)
    act_p.append(d[:, 1:7])                        # 6 actions
    rew_p.append(d[:, yi])
obs_p = np.vstack(obs_p); act_p = np.vstack(act_p); rew_p = np.concatenate(rew_p)
print(f"[reward] {len(files)} CSVs -> {obs_p.shape[0]} transitions "
      f"({n_err} error_reward rows dropped)")

# same two-stage normalisation as the state models: smpl min-max, then z-score
_o_n = 2.0 * (obs_p - pdata.MIN_OBS) / (pdata.MAX_OBS - pdata.MIN_OBS) - 1.0
_a_n = 2.0 * (act_p - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
std_obs = pdata.Standardizer(_o_n, names=pdata.OBS_NAMES, name="obs")
std_act = pdata.Standardizer(_a_n, names=pdata.ACT_NAMES, name="act")
obs = std_obs.transform(_o_n)
act = std_act.transform(_a_n)

# reward is z-scored too: raw yields span ~0..6.6, and an unscaled target makes the
# GP's signal variance and noise hard to initialise consistently with the state models
R_MU, R_SD = float(rew_p.mean()), float(rew_p.std())
rew = (rew_p - R_MU) / max(R_SD, 1e-12)
print(f"[reward] yield/step  mean={R_MU:.4f}  std={R_SD:.4f}  "
      f"min={rew_p.min():.4f}  max={rew_p.max():.4f}")

N = obs.shape[0]
rng = np.random.default_rng(0)
perm = rng.permutation(N)
te, tr = perm[:N_TEST], perm[N_TEST:]
X_te = torch.tensor(np.hstack([obs[te], act[te]]), dtype=dtype, device=device)
y_te = torch.tensor(rew[te].reshape(-1, 1), dtype=dtype, device=device)

Z_tr = np.hstack([obs[tr], act[tr]])
if _args.select == "pivchol":
    sel = pdata.select_pivoted_cholesky(Z_tr, N_KEEP)
else:
    sel = np.linspace(0, len(tr) - 1, min(N_KEEP, len(tr))).astype(int)
X = torch.tensor(Z_tr[sel], dtype=dtype, device=device)
Y = torch.tensor(rew[tr][sel].reshape(-1, 1), dtype=dtype, device=device)
print(f"train N={X.shape[0]}   test N={X_te.shape[0]}   gp input dim={GP_INPUT_DIM}")

# ------------------------------------------------------------------ model ----
init_dict = dict(
    active_dims=np.arange(0, GP_INPUT_DIM),
    lengthscales_init=np.ones(GP_INPUT_DIM), flg_train_lengthscales=True,
    lambda_init=np.ones(1), flg_train_lambda=True,
    sigma_n_init=1e-2 * np.ones(1), sigma_n_num=1e-4, flg_train_sigma_n=True,
    dtype=dtype, device=device,
)
model = ML.Model_learning_RBF(
    num_gp=1,                                  # ONE scalar GP: reward
    init_dict_list=[dict(init_dict)],
    approximation_mode=None, dtype=dtype, device=device, flg_norm=False)
model.gp_inputs = X
model.gp_output_list = [Y]
model.num_samples = X.shape[0]
model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM

opt = dict(f_optimizer="lambda p : torch.optim.Adam(p, lr=0.01)",
           criterion=Likelihood.Marginal_log_likelihood,
           N_epoch=N_EPOCH, N_epoch_print=max(1, N_EPOCH // 2))
model.reinforce_model([dict(opt)])
print("\nTrained hyperparameters:"); model.print_model()

# ------------------------------------------------------------ held-out fit ----
model.set_eval_mode()
with torch.no_grad():
    m_list, v_list = model.get_gp_estimate(gp_inputs=X_te, gp_index_list=[0])
pred, var = m_list[0], v_list[0].reshape(-1, 1)

mse = torch.mean((pred - y_te) ** 2).item()
base = torch.mean((y_te - y_te.mean()) ** 2).item()          # predict the mean
r2 = 1.0 - mse / base
mse_phys = mse * R_SD ** 2
print(f"\nheld-out reward prediction:")
print(f"  MSE (z)      {mse:.6e}   predict-the-mean baseline {base:.6e}")
print(f"  R^2          {r2:.4f}")
print(f"  RMSE (yield/step, physical units)  {np.sqrt(mse_phys):.4f}   "
      f"(target std {R_SD:.4f})")
print(f"  mean predictive std  {torch.sqrt(var).mean().item():.4f} (z)")
if r2 < 0.5:
    print("  WARNING: R^2 below 0.5 -- a policy maximising this model would be "
          "optimising noise. Consider more data or a different input representation.")

# ------------------------------------------------------------------- save ----
path = os.path.join(SAVE_DIR, f"reward_model_{TAG}.pt")
torch.save({
    "state_dict": model.state_dict(),
    "gp_inputs": model.gp_inputs, "gp_output_list": model.gp_output_list,
    "alpha_list": model.alpha_list, "m_X_list": model.m_X_list,
    "K_X_inv_list": model.K_X_inv_list, "gp_inputs_tr_list": model.gp_inputs_tr_list,
    "gp_input_dim": GP_INPUT_DIM, "n_keep": N_KEEP, "n_epoch": N_EPOCH,
    "reward_mu": R_MU, "reward_sd": R_SD,          # to map predictions back to yield
    "std_obs_mu": std_obs.mu.tolist(), "std_obs_sd": std_obs.sd.tolist(),
    "std_act_mu": std_act.mu.tolist(), "std_act_sd": std_act.sd.tolist(),
    "held_out_mse_z": mse, "held_out_r2": r2, "n_error_rows_dropped": n_err,
    "data_dir": folder, "n_csvs": len(files),
}, path)
with open(os.path.join(SAVE_DIR, f"reward_metrics_{TAG}.json"), "w") as f:
    json.dump({"tag": TAG, "n_csvs": len(files), "n_transitions": int(N),
               "n_train": int(X.shape[0]), "n_test": int(N_TEST),
               "n_error_rows_dropped": int(n_err),
               "reward_mu": R_MU, "reward_sd": R_SD,
               "held_out_mse_z": mse, "held_out_r2": r2,
               "rmse_physical": float(np.sqrt(mse_phys))}, f, indent=2)
print(f"\nsaved -> {path}")
