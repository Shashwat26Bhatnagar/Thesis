#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_w2_per_hour.py   (repo root)

Per-hour W2 for a trained policy, against two references at every hour:
    - the MEAN action (z = 0), which is what the collapsed policies emit
    - the best of N random actions, i.e. roughly what is achievable

    python check_w2_per_hour.py -policy results_tracenorm/tn_lam0.pt

WHY PER HOUR RATHER THAN THE MEAN OVER HOURS. The training log reports one averaged
W2, which cannot distinguish "uniformly mediocre everywhere" from "good in some hours,
bad in others". Those call for different responses: the first is an optimisation
problem, the second means the single time-invariant policy is being pulled between
incompatible per-hour optima -- and a random search already showed the per-hour
optima ARE incompatible, with discharge wanting +0.76 at t=75 and -1.59 at t=130.

The gap column is the one to read: policy_W2 - best_W2. If it is large and roughly
CONSTANT across hours, the policy is uniformly short of what is reachable. If it is
small at some hours and large at others, the policy has specialised -- which would be
the first evidence of genuine time-dependence rather than a compromise.
"""
import argparse
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
from policy_learning.policy_variants import rebuild_policy
from policy_learning.wasserstein_loss import (w2_cross_dim_torch, TRACE_NORMALIZE,
                                              SMOOTH_EPS)
from dcfba_pen.flgfn.pf_query import PFQuery

torch.set_num_threads(1)
dtype, device = torch.float64, torch.device("cpu")

SD, ID = pdata.OBS_DIM, pdata.ACT_DIM
GP_IN = SD + ID
STEPS, NUM_STATES, K_ACTIONS = 5, 100, 5

ap = argparse.ArgumentParser()
ap.add_argument("-policy", required=True)
ap.add_argument("-phase_prefix", default="results_pensim/rbf_model_bnd_rbf_iter0")
ap.add_argument("-n_random", type=int, default=200)
ap.add_argument("-last_step", action="store_true",
                help="compare the 5th step's covariance alone instead of the sum")
ap.add_argument("-every", type=int, default=10, help="report every Nth hour")
args = ap.parse_args()

print(f"trace_norm={TRACE_NORMALIZE}  smooth_eps={SMOOTH_EPS}  "
      f"cov={'5th step' if args.last_step else 'sum of 5'}")


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
    MODELS[p], CKS[p] = load(f"{args.phase_prefix}_phase{p}.pt")
POOLS = {p: MODELS[p].gp_inputs[:, :SD] for p in (0, 1, 2)}


def phase_of(t):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t < hi:
            return p
    return 2


pk = torch.load(args.policy, map_location=device, weights_only=False)
pol = rebuild_policy(pk["policy_meta"], dtype=dtype, device=device)
pol.load_state_dict(pk["policy_state_dict"])
pol.eval()
print(f"policy: {os.path.basename(args.policy)}  lam={pk.get('lam')}\n")

q = PFQuery(verbose=False)
TIMES = np.arange(1.0, 150.0 + 1e-9, 1.0)
REPORT = TIMES[::args.every]

_acc = {"var": None, "t0": 0, "last": None}
_eig = None
_out = []


def wloss(t, s, a, mu, cov, s_next):
    global _acc
    if _acc["var"] is None:
        _acc = {"var": torch.zeros_like(cov), "t0": t, "last": None}
    _acc["var"] = _acc["var"] + cov
    _acc["last"] = cov
    if (t - _acc["t0"] + 1) < STEPS:
        return torch.zeros((), dtype=cov.dtype, device=cov.device)
    v = _acc["last"] if args.last_step else _acc["var"]
    _acc = {"var": None, "t0": 0, "last": None}
    _out.append(float(w2_cross_dim_torch(v, _eig)
                      .view(NUM_STATES, K_ACTIONS).mean(1).mean()))
    return torch.zeros((), dtype=cov.dtype, device=cov.device)


def roll_const(a_val, mdl, s0):
    """5 steps at a FIXED action, same accumulation."""
    s = s0.clone()
    tot = torch.zeros(s0.shape[0], SD, dtype=dtype)
    last = None
    a = a_val.unsqueeze(0).expand(s0.shape[0], -1)
    for _ in range(STEPS):
        with torch.no_grad():
            mu, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a], dim=1),
                                        gp_index_list=list(range(SD)))
        cov = torch.cat([v[i].reshape(-1, 1) for i in range(SD)], dim=1)
        tot = tot + cov; last = cov
        s = s + torch.cat([mu[i].reshape(-1, 1) for i in range(SD)], dim=1)
    return last if args.last_step else tot


rng = np.random.default_rng(0)
print(f"{'t [h]':>6}{'policy W2':>11}{'mean-act':>10}{'best rnd':>10}"
      f"{'gap':>9}{'vs mean':>9}")
rows = []
for t in REPORT:
    t = float(t)
    ph = phase_of(t)
    _eig = torch.tensor(np.linalg.eigvalsh(np.asarray(
        q.next_state_distribution(t=t, source="traj")["cov_n"])).tolist(), dtype=dtype)
    gen = np.random.default_rng(int(t))
    st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=gen,
                                  dtype=dtype, device=device)
    s0 = st.repeat_interleave(K_ACTIONS, dim=0)

    _out.clear()
    with torch.no_grad():
        gp_rollout(model=MODELS[ph], policy=pol, s0=s0, T=STEPS, p_dropout=0.0,
                   particle_pred=False, loss_fn=wloss, graph_mode="full")
    w_pol = _out[-1] if _out else float("nan")

    w_mean = float(w2_cross_dim_torch(
        roll_const(torch.zeros(ID, dtype=dtype), MODELS[ph], s0), _eig)
        .view(NUM_STATES, K_ACTIONS).mean(1).mean())

    best = w_mean
    for _ in range(args.n_random):
        a = torch.tensor(rng.uniform(-2, 2, ID), dtype=dtype)
        v = float(w2_cross_dim_torch(roll_const(a, MODELS[ph], s0), _eig)
                  .view(NUM_STATES, K_ACTIONS).mean(1).mean())
        best = min(best, v)

    rows.append((t, w_pol, w_mean, best))
    print(f"{t:6.0f}{w_pol:11.5f}{w_mean:10.5f}{best:10.5f}"
          f"{w_pol-best:9.5f}{100*(w_mean-w_pol)/max(w_mean,1e-9):8.1f}%")

R = np.array(rows)
print(f"\npolicy W2 : mean={R[:,1].mean():.5f}  std={R[:,1].std():.5f}  "
      f"min={R[:,1].min():.5f}  max={R[:,1].max():.5f}")
print(f"gap to best: mean={(R[:,1]-R[:,3]).mean():.5f}  "
      f"std={(R[:,1]-R[:,3]).std():.5f}")
print(f"policy beats the mean action at {int((R[:,1] < R[:,2]).sum())}/{len(R)} hours")
print("\n  a gap that is large and FLAT across hours -> uniformly short of reachable")
print("  a gap that VARIES -> the policy has specialised to some hours over others")
