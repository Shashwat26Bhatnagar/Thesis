# Copyright (C) 2020, 2023 Mitsubishi Electric Research Laboratories (MERL)
# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Test MC-PILCO on a simulated cart-pole system with the RVGP (Matern-on-manifold) kernel.
Cloned from test_mcpilco_cartpole_rbf_ker.py; only the model-learning block is changed.
"""
import argparse
import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import torch
import os

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode

# ---- seed / device ----
p = argparse.ArgumentParser("test cartpole rvgp")
p.add_argument("-seed", type=int, default=1, help="seed")
locals().update(vars(p.parse_known_args()[0]))
torch.manual_seed(seed)
np.random.seed(seed)
dtype = torch.float64
device = torch.device("cpu")
torch.set_num_threads(1)

print("---- Set environment parameters ----")
num_trials = 5
T_sampling = 0.05
T_exploration = 3.0
T_control = 3.0
state_dim = 4
input_dim = 1
ode_fun = f_ode.cartpole
u_max = 10.0
std_noise = 10 ** (-2)
std_list = [std_noise, std_noise, std_noise, std_noise]

# =====================================================================================
#  MODEL LEARNING  ---  RVGP (Matern-on-manifold), ONE joint GP over the state
# =====================================================================================
print("---- Set model learning parameters (RVGP) ----")
f_model_learning = ML.Model_learning_RVGP

# Matern kernel init (replaces the RBF init_dict). REQUIRES sigma_n_init (low-rank block).
matern_init = {}
matern_init["nu_init"] = 1.5
matern_init["kappa_init"] = 5.0
matern_init["sigma_f_init"] = 1.0
matern_init["flg_train_nu"] = False
matern_init["flg_train_kappa"] = True
matern_init["flg_train_sigma_f"] = True
matern_init["sigma_n_init"] = 1.0 * np.ones(1)
matern_init["sigma_n_num"] = None
matern_init["flg_train_sigma_n"] = True
matern_init["dtype"] = dtype
matern_init["device"] = device

# RVGP manifold build options (all singular-value / Laplacian machinery is hidden here)
rvgp_dict = {}
rvgp_dict["n_neighbors"] = 10
rvgp_dict["explained_variance"] = 0.8
rvgp_dict["n_eigenpairs"] = 50            # k ; must stay well below N * D_e

# Model_learning_RVGP constructor kwargs (NOTE: different signature than the RBF class)
model_learning_par = {}
model_learning_par["init_dict"] = matern_init
model_learning_par["rvgp_dict"] = rvgp_dict
model_learning_par["angle_indices"] = [2]          # pole angle -> (cos, sin)
model_learning_par["not_angle_indices"] = [0, 1, 3]  # cart pos, cart vel, pole angvel
model_learning_par["dtype"] = dtype
model_learning_par["device"] = device
# D_e = 3 + 2*1 = 5 ; manifold ambient p = 5 + 1(action) = 6  (matches RBF gp_input_dim=6)

print("\n---- Set exploration policy ----")
f_rand_exploration_policy = Policy.Random_exploration
rand_exploration_policy_par = {}
rand_exploration_policy_par["state_dim"] = state_dim
rand_exploration_policy_par["input_dim"] = input_dim
rand_exploration_policy_par["u_max"] = u_max
rand_exploration_policy_par["dtype"] = dtype
rand_exploration_policy_par["device"] = device

print("\n---- Set control policy ----")
num_basis = 200
f_control_policy = Policy.Sum_of_gaussians_with_angles
control_policy_par = {}
control_policy_par["state_dim"] = state_dim
control_policy_par["input_dim"] = input_dim
control_policy_par["angle_indices"] = np.array([2])
control_policy_par["non_angle_indices"] = np.array([0, 1, 3])
control_policy_par["u_max"] = u_max
control_policy_par["num_basis"] = num_basis
control_policy_par["dtype"] = dtype
control_policy_par["device"] = device
angle_centers = np.pi * 2 * (np.random.rand(num_basis, 1) - 0.5)
cos_centers = np.cos(angle_centers)
sin_centers = np.sin(angle_centers)
not_angle_centers = np.pi * 2 * (np.random.rand(num_basis, 3) - 0.5)
control_policy_par["centers_init"] = np.concatenate([not_angle_centers, cos_centers, sin_centers], 1)
control_policy_par["lengthscales_init"] = 1 * np.ones(state_dim + 1)
control_policy_par["weight_init"] = u_max * (np.random.rand(input_dim, num_basis) - 0.5)
control_policy_par["flg_squash"] = True
control_policy_par["flg_drop"] = True
policy_reinit_dict = {}
policy_reinit_dict["lenghtscales_par"] = control_policy_par["lengthscales_init"]
policy_reinit_dict["centers_par"] = np.array([np.pi, np.pi, np.pi, 1.0, 1.0])
policy_reinit_dict["weight_par"] = u_max

print("\n---- Set cost function ----")
f_cost_function = Cost_function.Cart_pole_cost
cost_function_par = {}
cost_function_par["pos_index"] = 0
cost_function_par["angle_index"] = 2
cost_function_par["target_state"] = torch.tensor([np.pi, 0.0], dtype=dtype, device=device)
cost_function_par["lengthscales"] = torch.tensor([3.0, 1.0], dtype=dtype, device=device)

print("\n---- Init policy learning object ----")
MC_PILCO_init_dict = {}
MC_PILCO_init_dict["T_sampling"] = T_sampling
MC_PILCO_init_dict["state_dim"] = state_dim
MC_PILCO_init_dict["input_dim"] = input_dim
MC_PILCO_init_dict["f_sim"] = ode_fun
MC_PILCO_init_dict["std_meas_noise"] = np.array(std_list)
MC_PILCO_init_dict["f_model_learning"] = f_model_learning
MC_PILCO_init_dict["model_learning_par"] = model_learning_par
MC_PILCO_init_dict["f_rand_exploration_policy"] = f_rand_exploration_policy
MC_PILCO_init_dict["rand_exploration_policy_par"] = rand_exploration_policy_par
MC_PILCO_init_dict["f_control_policy"] = f_control_policy
MC_PILCO_init_dict["control_policy_par"] = control_policy_par
MC_PILCO_init_dict["f_cost_function"] = f_cost_function
MC_PILCO_init_dict["cost_function_par"] = cost_function_par
MC_PILCO_init_dict["log_path"] = "results_tmp/" + str(seed)
MC_PILCO_init_dict["dtype"] = dtype
MC_PILCO_init_dict["device"] = device
PL_obj = MC_PILCO.MC_PILCO(**MC_PILCO_init_dict)

print("\n---- Set MC-PILCO options ----")
# Model optimization options  (num_gp = 1 for RVGP -> list of length 1)
model_optimization_opt_dict = {}
model_optimization_opt_dict["f_optimizer"] = "lambda p : torch.optim.Adam(p, lr=0.01)"
model_optimization_opt_dict["criterion"] = Likelihood.Marginal_log_likelihood
model_optimization_opt_dict["N_epoch"] = 1501
model_optimization_opt_dict["N_epoch_print"] = 500
model_optimization_opt_list = [model_optimization_opt_dict]     # length 1 (one joint GP)

# Policy optimization options
policy_optimization_dict = {}
policy_optimization_dict["num_particles"] = 400
policy_optimization_dict["opt_steps_list"] = [2000, 4000, 4000, 4000, 4000]
policy_optimization_dict["lr_list"] = [0.01, 0.01, 0.01, 0.01, 0.01]
policy_optimization_dict["f_optimizer"] = "lambda p, lr : torch.optim.Adam(p, lr)"
policy_optimization_dict["num_step_print"] = 100
policy_optimization_dict["p_dropout_list"] = [0.25, 0.25, 0.25, 0.25, 0.25]
policy_optimization_dict["p_drop_reduction"] = 0.25 / 2
policy_optimization_dict["alpha_diff_cost"] = 0.99
policy_optimization_dict["min_diff_cost"] = 0.08
policy_optimization_dict["num_min_diff_cost"] = 200

# initial state distribution
initial_state = np.array([0.0, 0.0, 0.0, 0.0])
initial_state_var = 1e-4 * np.ones(state_dim)

print("\n---- Run MC-PILCO ----")
# reinforce() runs the whole loop: get_data_from_system -> add_data -> reinforce_model
# (model learning) -> reinforce_policy (policy opt).  For MODEL-LEARNING ONLY, keep the
# first trial and inspect the model before trusting the policy step (see notes).

os.makedirs(MC_PILCO_init_dict["log_path"], exist_ok=True)

PL_obj.reinforce(
    num_trials=num_trials,
    T_exploration=T_exploration,
    T_control=T_control,
    initial_state=initial_state,
    initial_state_var=initial_state_var,
    model_optimization_opt_list=model_optimization_opt_list,
    policy_optimization_dict=policy_optimization_dict,
#    policy_reinit_dict=policy_reinit_dict,
    flg_init_uniform=False,
    flg_init_multi_gauss=False,
)
