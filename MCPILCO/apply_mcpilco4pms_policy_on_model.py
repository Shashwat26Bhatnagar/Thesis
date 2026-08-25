"""
File to load from the logs the final policy obtained with MC-PILCO4PMS and test it in model learned.
"""

import copy
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import signal

import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood
import gpr_lib.Utils.Parameters_covariance_functions as cov_func
import model_learning.Model_learning as ML
import policy_learning.Cost_function as Cost_function
import policy_learning.MC_PILCO as MC_PILCO
import policy_learning.Policy as Policy
import simulation_class.ode_systems as f_ode

torch.set_num_threads(1)
dtype = torch.float64
device = torch.device("cpu")

seed = 1
folder_path = "results_tmp/" + str(seed) + "/"
config_file_path = folder_path + "/config_log.pkl"
saving_path = folder_path + "/reproduce_policy_log.pkl"

num_particles = 50

num_trial = 5

config_dict = pkl.load(open(config_file_path, "rb"))
PL_obj = MC_PILCO.MC_PILCO4PMS(**config_dict["MC_PILCO_init_dict"])
T_control = config_dict["reinforce_param_dict"]["T_control"]
initial_state = config_dict["reinforce_param_dict"]["initial_state"]
initial_state_var = config_dict["reinforce_param_dict"]["initial_state_var"]

particles_initial_state_mean = torch.tensor(initial_state, dtype=dtype, device=device)
particles_initial_state_var = torch.tensor(initial_state_var, dtype=dtype, device=device)

PL_obj.load_policy_from_log(num_trial, folder=folder_path)

PL_obj.load_model_from_log(num_trial, folder=folder_path)

particles_states, particles_inputs = PL_obj.apply_policy(
    particles_initial_state_mean=particles_initial_state_mean,
    particles_initial_state_var=particles_initial_state_var,
    flg_particles_init_uniform=False,
    particles_init_up_bound=None,
    particles_init_low_bound=None,
    flg_particles_init_multi_gauss=False,
    num_particles=num_particles,
    T_control=int(T_control / PL_obj.T_sampling),
    p_dropout=0.0,
)

particles_states = particles_states.detach().numpy()
particles_inputs = particles_inputs.detach().numpy()

plt.figure()
plt.subplot(3, 1, 1)
plt.grid()
plt.ylabel("POLE")
plt.plot(np.pi * np.ones(len(particles_states[:, :, 2])), "r--")
plt.plot(-np.pi * np.ones(len(particles_states[:, :, 2])), "r--")
plt.plot(particles_states[:, :, 2])
plt.subplot(3, 1, 2)
plt.grid()
plt.ylabel("CART")
plt.plot(np.zeros(len(particles_states[:, :, 0])), "r--")
plt.plot(particles_states[:, :, 0])
plt.subplot(3, 1, 3)
plt.grid()
plt.ylabel("FORCE")
plt.plot(particles_inputs[:, :, 0])
plt.show()
