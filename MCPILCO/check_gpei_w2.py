#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_gpei_w2.py   (repo root)

Does the REFERENCE CONTROLLER score well under our objective?

At each hour, replay gpei's own recorded action through the world model and compute W2
against the expert, alongside three baselines:

    gpei action      what the reference controller actually did
    mean action      z = 0, the dataset-mean action, which collapsed policies emit
    best of N random the reachable-optimum proxy
    trained policy   optional, via -policy

THIS IS THE VALIDATION TEST FOR THE OBJECTIVE ITSELF. gpei reaches 3835 yield against
our 2728-3486; it is the behaviour being imitated. If gpei's actions do NOT beat the
mean action under this W2, then the objective does not measure imitation, and every
result derived from it -- the collapse, the lambda sweeps, the phase split -- is
measuring something else.

    gpei WINS  -> the objective is sound; the policy simply is not reaching gpei's
                  actions, and the problem is optimisation or representation.
    gpei TIES  -> the objective cannot distinguish good control from no control.
    gpei LOSES -> the objective actively prefers the mean action over the reference,
                  and minimising it will never produce reference-like behaviour.

    python check_gpei_w2.py
    python check_gpei_w2.py -policy results_phase/phase_ph0.pt -every 10
"""
import argparse
import csv
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
from policy_learning.gp_particle_rollout import sample_initial_particles
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
ap.add_argument("-gpei_csv",
                default=os.path.expanduser(
                    "~/Thesis/deps/smpl/smpl/configdata/pensimenv/gpei_batch_0.csv"))
ap.add_argument("-phase_prefix", default="results_pensim/rbf_model_bnd_rbf_iter0")
ap.add_argument("-policy", default=None, help="optional trained policy to include")
ap.add_argument("-n_random", type=int, default=200)
ap.add_argument("-every", type=int, default=10)
ap.add_argument("-last_step", action="store_true")
args = ap.parse_args()

print(f"trace_norm={TRACE_NORMALIZE}  smooth_eps={SMOOTH_EPS}  "
      f"cov={'5th step' if args.last_step else 'sum of 5'}")

# ------------------------------------------------------ gpei's recorded actions ----
_h = [c.strip() for c in next(csv.reader(open(args.gpei_csv)))]
_d = np.genfromtxt(args.gpei_csv, delimiter=",", skip_header=1)
_d = _d[~np.isnan(_d).any(axis=1)]
GT, GA = _d[:, 0], _d[:, 1:7]
print(f"gpei: {os.path.basename(args.gpei_csv)}  {len(GT)} steps  "
      f"t={GT[0]:.1f}..{GT[-1]:.1f} h  yield={_d[:, -1].sum():.1f}")

MU_A = None  # filled after the models load


def gpei_action_z(t_h):
    """gpei's PHYSICAL action at the nearest recorded time, converted to z."""
    k = int(np.argmin(np.abs(GT - t_h)))
    a_phys = GA[k]
    smpl = 2.0 * (a_phys - pdata.MIN_ACT) / (pdata.MAX_ACT - pdata.MIN_ACT) - 1.0
    return torch.tensor((smpl - MU_A) / SD_A, dtype=dtype), a_phys


# ------------------------------------------------------------------- the models ---
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
MU_A = np.asarray(CKS[0]["std_act_mu"])
SD_A = np.asarray(CKS[0]["std_act_sd"])


def phase_of(t):
    for p in (0, 1, 2):
        lo, hi = pdata.PHASES[p]
        if lo <= t < hi:
            return p
    return 2


POL = None
if args.policy:
    pk = torch.load(args.policy, map_location=device, weights_only=False)
    POL = rebuild_policy(pk["policy_meta"], dtype=dtype, device=device)
    POL.load_state_dict(pk["policy_state_dict"])
    POL.eval()
    print(f"policy: {os.path.basename(args.policy)}")

q = PFQuery(verbose=False)


def w2_of(a_vec, mdl, s0, eig):
    """5 steps at a fixed action, same accumulation the trainer uses."""
    s = s0.clone()
    tot = torch.zeros(s0.shape[0], SD, dtype=dtype)
    last = None
    a = a_vec.unsqueeze(0).expand(s0.shape[0], -1)
    for _ in range(STEPS):
        with torch.no_grad():
            mu, v = mdl.get_gp_estimate(gp_inputs=torch.cat([s, a], dim=1),
                                        gp_index_list=list(range(SD)))
        cov = torch.cat([v[i].reshape(-1, 1) for i in range(SD)], dim=1)
        tot = tot + cov; last = cov
        s = s + torch.cat([mu[i].reshape(-1, 1) for i in range(SD)], dim=1)
    v = last if args.last_step else tot
    return float(w2_cross_dim_torch(v, eig).view(NUM_STATES, K_ACTIONS).mean(1).mean())


TIMES = np.arange(1.0, 150.0 + 1e-9, float(args.every))
rng = np.random.default_rng(0)
print(f"\n{'t [h]':>6}{'gpei W2':>10}{'mean W2':>10}{'best rnd':>10}"
      + (f"{'policy':>10}" if POL else "") + f"{'gpei vs mean':>14}")
rows = []
for t in TIMES:
    t = float(t)
    ph = phase_of(t)
    eig = torch.tensor(np.linalg.eigvalsh(np.asarray(
        q.next_state_distribution(t=t, source="traj")["cov_n"])).tolist(), dtype=dtype)
    gen = np.random.default_rng(int(t))
    st = sample_initial_particles(POOLS[ph], NUM_STATES, generator=gen,
                                 dtype=dtype, device=device)
    s0 = st.repeat_interleave(K_ACTIONS, dim=0)

    a_g, a_g_phys = gpei_action_z(t)
    w_g = w2_of(a_g, MODELS[ph], s0, eig)
    w_m = w2_of(torch.zeros(ID, dtype=dtype), MODELS[ph], s0, eig)
    w_b = w_m
    for _ in range(args.n_random):
        w_b = min(w_b, w2_of(torch.tensor(rng.uniform(-2, 2, ID), dtype=dtype),
                             MODELS[ph], s0, eig))
    w_p = None
    if POL:
        with torch.no_grad():
            a_p = POL(states=s0, t=0, p_dropout=0.0).mean(0)
        w_p = w2_of(a_p, MODELS[ph], s0, eig)

    rows.append((t, w_g, w_m, w_b, w_p if w_p is not None else np.nan))
    print(f"{t:6.0f}{w_g:10.5f}{w_m:10.5f}{w_b:10.5f}"
          + (f"{w_p:10.5f}" if POL else "")
          + f"{100*(w_m-w_g)/max(w_m,1e-9):13.1f}%")

R = np.array(rows)
n_win = int((R[:, 1] < R[:, 2]).sum())
print(f"\ngpei beats the mean action at {n_win}/{len(R)} hours")
print(f"  gpei W2  mean={R[:,1].mean():.5f}")
print(f"  mean-act mean={R[:,2].mean():.5f}")
print(f"  best rnd mean={R[:,3].mean():.5f}")
if POL:
    print(f"  policy   mean={np.nanmean(R[:,4]):.5f}")
print()
if n_win > 0.7 * len(R):
    print("  -> gpei WINS: the objective does prefer the reference's actions. The")
    print("     policy is not reaching them, so the problem is optimisation or")
    print("     representation, not the objective.")
elif n_win < 0.3 * len(R):
    print("  -> gpei LOSES: the objective prefers the MEAN ACTION over the controller")
    print("     that achieves 3835 yield. Minimising it cannot produce reference-like")
    print("     behaviour, and the observed collapse is the objective working as")
    print("     specified rather than a failure to optimise it.")
else:
    print("  -> MIXED: the objective distinguishes them at some hours and not others.")
    print("     Worth reading which hours, against the phase boundaries.")
