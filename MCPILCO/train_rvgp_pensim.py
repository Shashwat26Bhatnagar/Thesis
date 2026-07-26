#!/usr/bin/env python3
"""Train the RVGP world model on PenSim (state model only; reward ignored).
 
Dyna setup: fit here, freeze, then sample with model.predict_transition(obs, act).
Run from the MC-PILCO repo root.
"""
import numpy as np
import torch
 
import model_learning.Model_learning as ML
import model_learning.pensim_data as pdata
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
 
torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)
 
N_KEEP = 300          # -> joint block (300*9)^2 = 2700x2700
N_EIG = 50
 
# ---- data (normalized) ----
obs, act, nobs, std_obs, std_act = pdata.load_offline(max_transitions=4000)
n_te = 200
obs_te, act_te, nobs_te = obs[-n_te:], act[-n_te:], nobs[-n_te:]
obs, act, nobs = obs[:-n_te], act[:-n_te], nobs[:-n_te]
obs_s, act_s, nobs_s = pdata.subsample(obs, act, nobs, n_keep=N_KEEP)
print(f"train N={obs_s.shape[0]}  joint block {obs_s.shape[0]*9} x {obs_s.shape[0]*9}")
 
# ---- model ----
matern_init = dict(
    nu_init=1.5, kappa_init=5.0, sigma_f_init=1.0,
    flg_train_nu=False, flg_train_kappa=True, flg_train_sigma_f=True,
    sigma_n_init=1e-2 * np.ones(1), sigma_n_num=None, flg_train_sigma_n=True,
    dtype=dtype, device=device,
)
rvgp_dict = dict(n_neighbors=10, explained_variance=0.9, n_eigenpairs=N_EIG)
 
model = ML.Model_learning_RVGP(
    init_dict=matern_init, rvgp_dict=rvgp_dict,
    angle_indices=[],                          # PenSim has no periodic states
    not_angle_indices=list(range(pdata.OBS_DIM)),
    dtype=dtype, device=device,
)
model.add_transitions(obs_s, act_s, nobs_s)    # NOT add_data (rows not consecutive)
 
opt = dict(f_optimizer="lambda p : torch.optim.Adam(p, lr=0.01)",
           criterion=Likelihood.Marginal_log_likelihood,
           N_epoch=501, N_epoch_print=100)
model.reinforce_model([opt])
print("\nTrained hyperparameters:"); model.print_model()
 
# ---- one-step check vs persistence baseline ----
model.set_eval_mode()
pred, dmean, dvar = model.predict_transition(obs_te, act_te, particle_pred=False)
true = torch.as_tensor(nobs_te, dtype=dtype, device=device)
cur = torch.as_tensor(obs_te, dtype=dtype, device=device)
 
mse = torch.mean((pred - true) ** 2, dim=0)
base = torch.mean((cur - true) ** 2, dim=0)
np.set_printoptions(precision=6, suppress=True)
print("\nper-dim one-step MSE:", mse.detach().numpy())
print("persistence baseline:", base.detach().numpy())
print("mean MSE %.6e   baseline %.6e" % (mse.mean().item(), base.mean().item()))
print("model beats persistence:", bool(mse.mean() < base.mean()))
