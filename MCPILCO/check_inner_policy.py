#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
for _c in (os.path.expanduser("~/Thesis/penicillin-dcfba"),
           os.path.expanduser("~/penicillin-dcfba")):
    if os.path.isdir(_c) and _c not in sys.path:
        sys.path.insert(0, _c); break

import model_learning.Model_learning as ML
import model_learning.pensim_dataset as pdata
from policy_learning.gp_particle_rollout import gp_rollout, sample_initial_particles
from policy_learning.policy_variants import build_policy
from policy_learning.wasserstein_loss import w2_cross_dim_torch, TRACE_NORMALIZE
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")
np.random.seed(0); torch.manual_seed(0)

SD, ID = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = SD + ID
NUM_STATES, K_ACTIONS = 100, 5
STEPS = 5
PREFIX = os.environ.get("PREFIX", "results_pensim/rbf_model_bnd_rbf_iter0")
K = int(os.environ.get("K", "15"))
LR = float(os.environ.get("LR", "0.02"))
TIMES = [10.0, 30.0, 50.0, 75.0, 100.0, 130.0, 150.0]

print(f"trace-normalised W2: {TRACE_NORMALIZE}   inner k={K}  alpha={LR}")


def load(path):
    ck = torch.load(path, map_location=device, weights_only=False)
    init = dict(active_dims=np.arange(0, GP_IN), lengthscales_init=np.ones(GP_IN),
                flg_train_lengthscales=True, lambda_init=np.ones(1),
                flg_train_lambda=True, sigma_n_init=1e-2 * np.ones(1),
                sigma_n_num=1e-4, flg_train_sigma_n=True, dtype=dtype, device=device)
    m = ML.Model_learning_RBF(num_gp=SD, init_dict_list=[dict(init) for _ in range(SD)],
                              approximation_mode=None, dtype=dtype, device=device,
                              flg_norm=False)
    m.load_state_dict(ck["state_dict"])
    for k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
              "K_X_inv_list", "gp_inputs_tr_list"):
        setattr(m, k, ck[k])
    m.num_samples = ck["gp_inputs"].shape[0]
    m.dim_state, m.dim_input = SD, ID
    m.norm_list = [1.0] * SD
    m.set_eval_mode()
    return m, ck


MODELS, CKS = {}, {}
for p in (0, 1, 2):
    MODELS[p], CKS[p] = load(f"{PREFIX}_phase{p}.pt")
POOLS = {p: MODELS[p].gp_inputs[:, :SD] for p in (0, 1, 2)}
POOL = torch.cat([POOLS[p] for p in (0, 1, 2)], 0)
stats = {k: np.asarray(CKS[0][k]) for k in ("std_act_mu", "std_act_sd")}


def phase_of(t):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t < hi:
            return p
    return 2


q = PFQuery(verbose=False)
EIG = {t: torch.tensor(np.linalg.eigvalsh(np.asarray(
        q.next_state_distribution(t=t, source="traj")["cov_n"])).tolist(), dtype=dtype)
       for t in TIMES}

policy0, meta = build_policy("rbf", SD, ID, u_max=3.0, dtype=dtype, device=device,
                             rng=np.random.default_rng(0), num_basis=200,
                             s_lo=POOL.min(0).values.tolist(),
                             s_hi=POOL.max(0).values.tolist(), center_range_pad=1.10)
THETA0 = {k: v.detach().clone() for k, v in policy0.state_dict().items()}

_acc = {"var": None, "t0": 0}
_eig = None


def wloss(t, s, a, mu, cov, s_next):
    global _acc
    if _acc["var"] is None:
        _acc = {"var": torch.zeros_like(cov), "t0": t}
    _acc["var"] = _acc["var"] + cov
    if (t - _acc["t0"] + 1) < STEPS:
        return torch.zeros((), dtype=cov.dtype, device=cov.device)
    v = _acc["var"]; _acc = {"var": None, "t0": 0}
    return w2_cross_dim_torch(v, _eig).view(NUM_STATES, K_ACTIONS).mean(1).mean()


print(f"\n{'t [h]':>6}{'W2 before':>11}{'W2 after':>10}{'drop':>8}   "
      f"phi action (z), mean over states")
A_PHI, W_B, W_A = [], [], []
for t in TIMES:
    policy0.load_state_dict(THETA0)
    opt = torch.optim.Adam(policy0.parameters(), lr=LR)
    ph = phase_of(t)
    _eig = EIG[t]
    gen = np.random.default_rng(int(t))
    w_before = w_after = None
    for step in range(K):
        st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=gen,
                                      dtype=dtype, device=device)
        s0 = st.repeat_interleave(K_ACTIONS, dim=0)
        out = gp_rollout(model=MODELS[ph], policy=policy0, s0=s0, T=STEPS,
                         p_dropout=0.25, particle_pred=True, loss_fn=wloss,
                         graph_mode="full")
        loss = out["loss_total"]
        if step == 0:
            w_before = float(loss.detach())
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy0.parameters(), 10.0)
        opt.step()
        w_after = float(loss.detach())
    with torch.no_grad():
        a = policy0(states=POOLS[ph][:64], t=0, p_dropout=0.0).mean(0)
    A_PHI.append(a.numpy()); W_B.append(w_before); W_A.append(w_after)
    print(f"{t:6.0f}{w_before:11.5f}{w_after:10.5f}"
          f"{100*(w_before-w_after)/max(w_before,1e-9):7.1f}%   {np.round(a.numpy(), 3)}")

A = np.array(A_PHI)
print(f"\n{'channel':14s}{'std ACROSS phi_h':>18}{'as % of gpei':>15}")
SDA = stats["std_act_sd"]; SPA = pdata.MAX_ACT - pdata.MIN_ACT
GPEI = np.array([858.26, 24.98, 5.10, 8.90, 0.145, 151.21])
for i, n in enumerate(pdata.ACT_NAMES):
    sd = A[:, i].std()
    print(f"{n:14s}{sd:18.4f}{100*sd*SDA[i]*SPA[i]/2/GPEI[i]:14.2f}%")

print("\n  If these spreads are LARGE, per-hour solutions exist and gradient descent")
print("  finds them -- the collapse is then the OUTER average over windows, i.e. the")
print("  deployed policy is a barycenter of the phi_h.")
print("  If they are SMALL, the collapse happens inside one window and the outer")
print("  average is not the cause.")

print(f"\nfor reference, the OUTER-averaged theta after one Reptile step:")
new = {k: THETA0[k] for k in THETA0}
print("  (this run inspects phi only; the averaged theta is what deployment uses)")
