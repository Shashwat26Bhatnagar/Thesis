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
from policy_learning.wasserstein_loss import w2_cross_dim_torch
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype = torch.float64
STATE_DIM, INPUT_DIM = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = STATE_DIM + INPUT_DIM
PREFIX = os.environ.get("PREFIX", "results_pensim/rbf_model_bnd_rbf_iter0")

print("=" * 78)
print("1. DOES THE EXPERT TARGET VARY WITH TIME?")
print("=" * 78)
q = PFQuery(verbose=False)
TS = np.arange(1.0, 150.0 + 1e-9, 1.0)
EIG, COV = {}, {}
for t in TS:
    d = q.next_state_distribution(t=float(t), source="traj")
    c = np.asarray(d["cov_n"])
    COV[t] = c
    EIG[t] = np.linalg.eigvalsh(c)

E = np.array([EIG[t] for t in TS])
print(f"  {'t [h]':>7}{'eig1':>12}{'eig2':>12}{'eig3':>12}{'trace':>12}")
for t in (1.0, 10.0, 50.0, 75.0, 100.0, 130.0, 150.0):
    e = EIG[t]
    print(f"  {t:7.0f}{e[0]:12.3e}{e[1]:12.3e}{e[2]:12.3e}{e.sum():12.3e}")
print(f"\n  across all {len(TS)} hours:")
for j in range(3):
    print(f"    eig{j+1}: min={E[:,j].min():.3e} max={E[:,j].max():.3e} "
          f"ratio={E[:,j].max()/max(E[:,j].min(),1e-30):8.1f}x")
tr = E.sum(1)
print(f"    trace: min={tr.min():.3e} max={tr.max():.3e} "
      f"ratio={tr.max()/max(tr.min(),1e-30):.1f}x   "
      f"rel std={tr.std()/tr.mean():.3f}")
print("  -> a ratio near 1.0 would mean the target is effectively constant in time")

print()
print("=" * 78)
print("2. IS THE TARGET CONDITIONED ON STATE, OR ONLY ON TIME?")
print("=" * 78)
d1 = q.next_state_distribution(t=75.0, source="traj")
d2 = q.next_state_distribution(t=75.0, source="traj")
same = np.allclose(np.asarray(d1["cov_n"]), np.asarray(d2["cov_n"]))
print(f"  two calls at t=75 h give identical covariance: {same}")
print("  the API signature takes t; the conditioning state is interpolated from the")
print("  dcFBA trajectory. So EVERY particle at hour t -- wherever it actually is --")
print("  receives the SAME target. The loss cannot distinguish particle states.")
print(f"  conditioning state at t=75: {np.round(np.asarray(d1.get('b', [])), 4)}")

print()
print("=" * 78)
print("3. DOES THE GP's PREDICTIVE COVARIANCE RESPOND TO THE ACTION?")
print("=" * 78)


def load(path, n_gp):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    init = dict(active_dims=np.arange(0, GP_IN), lengthscales_init=np.ones(GP_IN),
                flg_train_lengthscales=True, lambda_init=np.ones(1),
                flg_train_lambda=True, sigma_n_init=1e-2 * np.ones(1),
                sigma_n_num=1e-4, flg_train_sigma_n=True, dtype=dtype, device="cpu")
    m = ML.Model_learning_RBF(num_gp=n_gp,
                              init_dict_list=[dict(init) for _ in range(n_gp)],
                              approximation_mode=None, dtype=dtype, device="cpu",
                              flg_norm=False)
    m.load_state_dict(ck["state_dict"])
    for k in ("gp_inputs", "gp_output_list", "alpha_list", "m_X_list",
              "K_X_inv_list", "gp_inputs_tr_list"):
        setattr(m, k, ck[k])
    m.num_samples = ck["gp_inputs"].shape[0]
    m.dim_state, m.dim_input = STATE_DIM, INPUT_DIM
    m.norm_list = [1.0] * n_gp
    m.set_eval_mode()
    return m, ck


mdl, ck = load(f"{PREFIX}_phase2.pt", STATE_DIM)
s = ck["gp_inputs"][0, :STATE_DIM]

print(f"  {'channel':13s}{'a=-3':>12}{'a=0':>12}{'a=+3':>12}{'rel range':>12}")
for j, nm in enumerate(pdata.ACT_NAMES):
    trs = []
    for av in (-3.0, 0.0, 3.0):
        a = torch.zeros(INPUT_DIM, dtype=dtype)
        a[j] = av
        with torch.no_grad():
            _, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a]).reshape(1, -1),
                                       gp_index_list=list(range(STATE_DIM)))
        trs.append(float(sum(v[i].reshape(-1)[0] for i in range(STATE_DIM))))
    trs = np.array(trs)
    rel = (trs.max() - trs.min()) / max(trs.mean(), 1e-12)
    flag = "  <-- essentially flat" if rel < 0.05 else ""
    print(f"  {nm:13s}{trs[0]:12.5f}{trs[1]:12.5f}{trs[2]:12.5f}{rel:12.4f}{flag}")

print()
print("=" * 78)
print("4. AND WHAT DOES W2 ITSELF DO AS THE ACTION VARIES?")
print("=" * 78)
eig75 = torch.tensor(EIG[75.0].tolist(), dtype=dtype)
print(f"  {'channel':13s}{'W2(a=-3)':>12}{'W2(a=0)':>12}{'W2(a=+3)':>12}{'rel range':>12}")
for j, nm in enumerate(pdata.ACT_NAMES):
    ws = []
    for av in (-3.0, 0.0, 3.0):
        a = torch.zeros(INPUT_DIM, dtype=dtype)
        a[j] = av
        with torch.no_grad():
            _, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a]).reshape(1, -1),
                                       gp_index_list=list(range(STATE_DIM)))
            cov = torch.cat([v[i].reshape(1, 1) for i in range(STATE_DIM)], dim=1)
            ws.append(float(w2_cross_dim_torch(cov * 5.0, eig75).reshape(-1)[0]))
    ws = np.array(ws)
    rel = (ws.max() - ws.min()) / max(ws.mean(), 1e-12)
    flag = "  <-- W2 cannot see this channel" if rel < 0.05 else ""
    print(f"  {nm:13s}{ws[0]:12.5f}{ws[1]:12.5f}{ws[2]:12.5f}{rel:12.4f}{flag}")

print()
print("=" * 78)
print("5. HOW MUCH DOES W2 VARY ACROSS TIME vs ACROSS ACTIONS?")
print("=" * 78)
a0 = torch.zeros(INPUT_DIM, dtype=dtype)
with torch.no_grad():
    _, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a0]).reshape(1, -1),
                               gp_index_list=list(range(STATE_DIM)))
    cov0 = torch.cat([v[i].reshape(1, 1) for i in range(STATE_DIM)], dim=1) * 5.0
w_t = [float(w2_cross_dim_torch(
    cov0, torch.tensor(EIG[t].tolist(), dtype=dtype)).reshape(-1)[0]) for t in TS]
w_t = np.array(w_t)
print(f"  W2 across the 150 expert hours, action fixed at z=0:")
print(f"    min={w_t.min():.5f}  max={w_t.max():.5f}  "
      f"rel range={(w_t.max()-w_t.min())/max(w_t.mean(),1e-12):.4f}")
print()
print("  If W2 varies far more across TIME than across ACTIONS, the policy is being")
print("  told 'the target changed' rather than 'your action was wrong' -- and a")
print("  constant action is the best single answer to a target it cannot influence.")
