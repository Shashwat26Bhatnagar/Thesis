"""Train a standard RBF Gaussian-process world model on the PenSim offline data.

Uses BOTH batch CSVs. Model learning only; saves the fitted model + standardizer
statistics so it can be reloaded as a frozen one-step predictor.
"""
import os, json
import numpy as np
import torch

import model_learning.Model_learning as ML
import model_learning.pensim_data as pdata
import gpr_lib.Likelihood.Gaussian_likelihood as Likelihood

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

N_KEEP = 300
N_TEST = 200
STATE_DIM = pdata.OBS_DIM          # 8
INPUT_DIM = pdata.ACT_DIM          # 6
GP_INPUT_DIM = STATE_DIM + INPUT_DIM
SAVE_DIR = "results_pensim"
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------------------------------------------------------- data ----
obs, act, nobs, std_obs, std_act = pdata.load_offline(max_transitions=None)
N = obs.shape[0]
print(f"total transitions loaded: {N}  (expect 1150 per batch CSV)")

# RANDOM held-out split across the whole dataset, not the tail of one batch
rng = np.random.default_rng(0)
perm = rng.permutation(N)
te_idx, tr_idx = perm[:N_TEST], perm[N_TEST:]
obs_te, act_te, nobs_te = obs[te_idx], act[te_idx], nobs[te_idx]
obs_tr, act_tr, nobs_tr = obs[tr_idx], act[tr_idx], nobs[tr_idx]

obs_s, act_s, nobs_s = pdata.subsample(obs_tr, act_tr, nobs_tr, n_keep=N_KEEP)
print(f"train N={obs_s.shape[0]}   test N={obs_te.shape[0]}   "
      f"gp input dim={GP_INPUT_DIM}   num_gp={STATE_DIM}")

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
model.gp_inputs = X
model.gp_output_list = [DY[:, i].reshape(-1, 1) for i in range(STATE_DIM)]
model.num_samples = X.shape[0]
model.dim_state, model.dim_input = STATE_DIM, INPUT_DIM

# ---------------------------------------------------------------- train ----
opt = dict(f_optimizer="lambda p : torch.optim.Adam(p, lr=0.01)",
           criterion=Likelihood.Marginal_log_likelihood,
           N_epoch=501, N_epoch_print=250)
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

print("\nheld-out NEXT-STATE MSE (standardized units):")
for i, nm in enumerate(pdata.OBS_NAMES):
    tag = "OK" if mse[i] < base[i] else "WORSE"
    print(f"  {i} {nm:6s} model={mse[i].item():.6e}  persistence={base[i].item():.6e}  {tag}")
print(f"\nmean MSE {mse.mean().item():.6e}   persistence {base.mean().item():.6e}")
print("model beats persistence:", bool(mse.mean() < base.mean()))

# ----------------------------------------------------------------- save ----
# torch.save handles tensors natively; standardizer stats stored as plain lists
# to avoid the numpy-identity pickling error.
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
    "gp_input_dim": GP_INPUT_DIM, "n_keep": N_KEEP,
    "std_obs_mu": std_obs.mu.tolist(), "std_obs_sd": std_obs.sd.tolist(),
    "std_act_mu": std_act.mu.tolist(), "std_act_sd": std_act.sd.tolist(),
    "obs_names": pdata.OBS_NAMES, "act_names": pdata.ACT_NAMES,
    "mse_per_dim": mse.tolist(), "persistence_per_dim": base.tolist(),
}, os.path.join(SAVE_DIR, "rbf_model.pt"))

with open(os.path.join(SAVE_DIR, "rbf_metrics.json"), "w") as f:
    json.dump({
        "n_train": int(obs_s.shape[0]), "n_test": int(obs_te.shape[0]),
        "n_total_transitions": int(N),
        "mse_per_dim":        {n: float(mse[i])  for i, n in enumerate(pdata.OBS_NAMES)},
        "persistence_per_dim":{n: float(base[i]) for i, n in enumerate(pdata.OBS_NAMES)},
        "mean_mse": float(mse.mean()), "mean_persistence": float(base.mean()),
        "std_obs_mu": std_obs.mu.tolist(), "std_obs_sd": std_obs.sd.tolist(),
        "std_act_mu": std_act.mu.tolist(), "std_act_sd": std_act.sd.tolist(),
    }, f, indent=2)

print(f"\nsaved -> {SAVE_DIR}/rbf_model.pt")
print(f"saved -> {SAVE_DIR}/rbf_metrics.json")
