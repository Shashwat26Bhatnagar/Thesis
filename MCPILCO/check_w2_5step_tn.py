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
from policy_learning.wasserstein_loss_tn import w2_cross_dim_torch
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype = torch.float64
SD, ID = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = SD + ID
STEPS = 5
PREFIX = os.environ.get("PREFIX", "results_pensim/rbf_model_bnd_rbf_iter0")
T_H = float(os.environ.get("T_H", "75"))


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    init = dict(active_dims=np.arange(0, GP_IN), lengthscales_init=np.ones(GP_IN),
                flg_train_lengthscales=True, lambda_init=np.ones(1),
                flg_train_lambda=True, sigma_n_init=1e-2 * np.ones(1),
                sigma_n_num=1e-4, flg_train_sigma_n=True, dtype=dtype, device="cpu")
    m = ML.Model_learning_RBF(num_gp=SD, init_dict_list=[dict(init) for _ in range(SD)],
                              approximation_mode=None, dtype=dtype, device="cpu",
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


ph = 2 if T_H >= 72.5 else (1 if T_H >= 47.5 else 0)
mdl, ck = load(f"{PREFIX}_phase{ph}.pt")
q = PFQuery(verbose=False)
_d = q.next_state_distribution(t=T_H, source="traj")
eig = torch.tensor(np.linalg.eigvalsh(np.asarray(_d["cov_n"])).tolist(), dtype=dtype)
print(f"expert @ t={T_H:.0f} h (phase {ph}): eigenvalues "
      f"{np.round(eig.numpy(), 6)}   trace={float(eig.sum()):.6f}\n")


def roll(s0, a, n=STEPS):
    """5 GP steps at a constant action, summing the per-step covariances -- the same
    accumulation window_loss performs."""
    s = s0.clone()
    tot = torch.zeros(1, SD, dtype=dtype)
    for _ in range(n):
        with torch.no_grad():
            mu, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a]).reshape(1, -1),
                                        gp_index_list=list(range(SD)))
        cov = torch.cat([v[i].reshape(1, 1) for i in range(SD)], dim=1)
        tot = tot + cov
        s = s + torch.cat([mu[i].reshape(1) for i in range(SD)])
    return tot, s


s0 = ck["gp_inputs"][0, :SD]
AS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)

print("W2 with the SUMMED 5-step covariance (what training uses)")
print(f"{'a':>6}" + "".join(f"{n[:9]:>11}" for n in pdata.ACT_NAMES))
W = np.zeros((len(AS), ID))
for r, av in enumerate(AS):
    for j in range(ID):
        a = torch.zeros(ID, dtype=dtype); a[j] = av
        tot, _ = roll(s0, a)
        W[r, j] = float(w2_cross_dim_torch(tot, eig).reshape(-1)[0])
    print(f"{av:6.1f}" + "".join(f"{W[r, j]:11.5f}" for j in range(ID)))

print(f"\n{'channel':13s}{'min':>10}{'max':>10}{'range':>10}{'# of 9 == 0':>13}")
for j, n in enumerate(pdata.ACT_NAMES):
    nz = int((W[:, j] == 0).sum())
    print(f"{n:13s}{W[:, j].min():10.5f}{W[:, j].max():10.5f}"
          f"{W[:, j].max()-W[:, j].min():10.5f}{nz:9d} / 9")

print("\nWHERE THE EXPERT'S EIGENVALUES SIT RELATIVE TO THE CLAMP BAND")
print("(inside [lo, hi] -> cost 0 and NO gradient for that eigenvalue)")
for av in (0.0, 3.0):
    a = torch.zeros(ID, dtype=dtype)
    tot, _ = roll(s0, a) if av == 0.0 else (None, None)
    if av != 0.0:
        a[0] = av
        tot, _ = roll(s0, a)
    lam, _ = torch.sort(tot, dim=1, descending=True)
    m = eig.shape[0]
    lo = lam[0, SD - m:].flip(0)
    hi = lam[0, :m]
    g, _ = torch.sort(eig, descending=True)
    print(f"\n  action a=0 on all channels" if av == 0.0
          else f"\n  discharge a={av}")
    print(f"    model diag sorted: {np.round(lam[0].numpy(), 6)}")
    for i in range(m):
        inside = bool(lo[i] <= g[i] <= hi[i])
        print(f"    gamma{i+1}={float(g[i]):.6f}  band=[{float(lo[i]):.6f}, "
              f"{float(hi[i]):.6f}]  inside={inside}"
              + ("   -> contributes NOTHING" if inside else "   -> contributes"))
