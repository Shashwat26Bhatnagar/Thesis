#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
import policy_learning.Policy as Policy
from policy_learning.gp_particle_rollout import (gp_rollout, sample_initial_particles,
                                            rollout_step_stats)

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0)
torch.manual_seed(0)

MODEL_PATH = "results_pensim/rbf_model.pt"
STATE_DIM = pdata.OBS_DIM
INPUT_DIM = pdata.ACT_DIM
GP_INPUT_DIM = STATE_DIM + INPUT_DIM

NUM_PARTICLES = 200
T_ROLLOUT = 40
N_ITERS = 300
LR = 0.01
P_DROPOUT = 0.25


def load_rbf_model(path=MODEL_PATH):
    ckpt = torch.load(path, map_location=device, weights_only=False)

    init_dict = dict(
        active_dims=np.arange(0, GP_INPUT_DIM),
        lengthscales_init=np.ones(GP_INPUT_DIM),
        flg_train_lengthscales=True,
        lambda_init=np.ones(1), flg_train_lambda=True,
        sigma_n_init=1e-2 * np.ones(1), sigma_n_num=1e-4, flg_train_sigma_n=True,
        dtype=dtype, device=device,
    )
    model = ML.Model_learning_RBF(
        num_gp=STATE_DIM,
        init_dict_list=[dict(init_dict) for _ in range(STATE_DIM)],
        approximation_mode=None,
        dtype=dtype, device=device, flg_norm=False,
    )
    model.load_state_dict(ckpt["state_dict"])

    model.gp_inputs = ckpt["gp_inputs"]
    model.gp_output_list = ckpt["gp_output_list"]
    model.alpha_list = ckpt["alpha_list"]
    model.m_X_list = ckpt["m_X_list"]
    model.K_X_inv_list = ckpt["K_X_inv_list"]
    model.gp_inputs_tr_list = ckpt["gp_inputs_tr_list"]
    model.num_samples = ckpt["gp_inputs"].shape[0]
    model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM
    model.norm_list = [1.0] * STATE_DIM

    stats = {
        "std_obs_mu": np.asarray(ckpt["std_obs_mu"]), "std_obs_sd": np.asarray(ckpt["std_obs_sd"]),
        "std_act_mu": np.asarray(ckpt["std_act_mu"]), "std_act_sd": np.asarray(ckpt["std_act_sd"]),
    }
    return model, stats, ckpt


model, stats, ckpt = load_rbf_model()

model.set_eval_mode()
print(f"loaded frozen model: num_gp={model.num_gp}, "
      f"train points={model.gp_inputs.shape[0]}, gp input dim={model.gp_inputs.shape[1]}")


num_basis = 200
policy_par = dict(
    state_dim=STATE_DIM,
    input_dim=INPUT_DIM,
    num_basis=num_basis,
    u_max=3.0,
    flg_squash=True,
    flg_drop=True,
    centers_init=np.random.randn(num_basis, STATE_DIM),
    lengthscales_init=np.ones(STATE_DIM),
    weight_init=0.1 * np.random.randn(INPUT_DIM, num_basis),
    dtype=dtype, device=device,
)
policy = Policy.Sum_of_gaussians(**policy_par)
print(f"policy: {type(policy).__name__}  trainable params="
      f"{sum(p.numel() for p in policy.parameters() if p.requires_grad)}")


state_pool = model.gp_inputs[:, :STATE_DIM]


optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
rng = np.random.default_rng(0)

for it in range(N_ITERS):
    s0 = sample_initial_particles(state_pool, NUM_PARTICLES, generator=rng,
                                  dtype=dtype, device=device)

    S, A, Dmean, Dvar = gp_rollout(
        model=model,
        policy=policy,
        s0=s0,
        T=T_ROLLOUT,
        p_dropout=P_DROPOUT,
        particle_pred=True,
    )

    state_cost = ((S[:, 1:, :] - S[:, :1, :]) ** 2).sum(dim=2)
    action_cost = 1e-3 * (A ** 2).sum(dim=2)
    per_particle_cost = (state_cost + action_cost).sum(dim=1)
    loss = per_particle_cost.mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if it % 20 == 0:
        with torch.no_grad():
            gnorm = torch.sqrt(sum((p.grad ** 2).sum() for p in policy.parameters()
                                   if p.grad is not None))
        _, _, pred_std = rollout_step_stats(Dmean, Dvar)
        print(f"iter {it:4d}  loss={loss.item():.6e}  |grad|={gnorm.item():.3e}  "
              f"cost std={per_particle_cost.std().item():.3e}")
        print(f"           GP predictive std: step0 mean={pred_std[0].mean().item():.3e}  "
              f"step{T_ROLLOUT-1} mean={pred_std[-1].mean().item():.3e}  "
              f"(growth => leaving training distribution)")

print("\ndone.")
torch.save({"policy_state_dict": policy.state_dict(), "policy_par":
            {k: (v.tolist() if isinstance(v, np.ndarray) else v)
             for k, v in policy_par.items() if k not in ("dtype", "device")}},
           "results_pensim/policy.pt")
print("saved -> results_pensim/policy.pt")
