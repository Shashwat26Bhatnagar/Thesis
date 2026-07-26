#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/gp_particle_rollout.py

Particle rollout through a FROZEN GP world model.
Not PenSim-specific: works with any Model_learning + Policy pair.

MAXIMUM REUSE: the entire per-step transition -- GP input assembly, GP inference,
the reparameterization trick, and the state update -- is delegated to the existing
MC-PILCO method Model_learning.get_next_state(). Nothing is reimplemented here.

    get_next_state(current_state, current_input, particle_pred)
      -> get_one_step_gp_out()
           -> data_to_gp_input()            # torch.cat([s, a], dim=1)
           -> get_gp_estimate()             # -> get_exact_gp_estimate()
                -> gp.get_estimate_from_alpha()   # differentiable in X_test
      -> get_next_state_from_gp_output()
           delta_mean = cat(gp_output_mean_list, 1)
           delta_var  = cat(gp_output_var_list, 1)
           d = Normal(delta_mean, sqrt(clamp(delta_var))).rsample()   # REPARAM TRICK
           next_states = current_state + d
      -> (next_states, delta_mean, delta_var)

TRANSITION DISTRIBUTION  P(s' | s, a)
-------------------------------------
Derived from those returns without extra inference:

    mu(s,a)  = s + delta_mean      # adding s shifts the mean only
    cov(s,a) = delta_var           # DIAGONAL: the model is num_gp independent
                                   # scalar GPs, so there is no cross-channel term

Both carry grad_fn back to the policy parameters.

CDIL SUPPORT (per-step loss)
----------------------------
Standard MC-PILCO accumulates a single terminal cost. Cross-Domain Imitation
Learning needs a loss AT EACH STEP, which changes how the autograd graph is
managed. Two optional hooks:

    loss_fn(t, s, a, mu, cov, s_next) -> scalar tensor
        called every step; the returned tensors are collected and also returned.

    flush_fn(loss_segment) -> None
        called at graph boundaries (see graph_mode). The DRIVER supplies this and
        does backward()/clip/step inside it, so optimizer logic stays out of here.
        After a flush the state is detached, starting a fresh graph segment.

graph_mode:
    "full"      accumulate every step into ONE graph; the driver backprops once
                after the rollout. Exact BPTT through the whole horizon. Default.
    "truncated" flush every `truncate_every` steps, then detach. Bounded memory;
                gradients do not cross a flush boundary.
    "per_step"  flush and detach every step. Myopic gradients; debugging only.
"""

import torch


def gp_rollout(
    model,
    policy,
    s0,
    T,
    p_dropout=0.0,
    particle_pred=True,
    loss_fn=None,
    flush_fn=None,
    graph_mode="full",
    truncate_every=10,
):
    """Roll `s0` particles forward `T` steps through the frozen GP model.

    Returns
    -------
    out : dict
        S       (P, T+1, ds)  states, S[:, 0] == s0
        A       (P, T,   da)  actions
        Mu      (P, T,   ds)  MEAN of P(s'|s,a) at each step
        Cov     (P, T,   ds)  DIAGONAL covariance of P(s'|s,a) at each step
        Dmean   (P, T,   ds)  delta mean  (Mu = S[:, :-1] + Dmean)
        Dvar    (P, T,   ds)  delta variance (== Cov)
        step_losses  list[scalar tensor]  per-step losses if loss_fn was given
        loss_total   scalar tensor or None -- sum of the losses still in the graph

    In "full" mode every tensor above shares one graph, so a single
    loss_total.backward() in the driver gives exact gradients through the horizon.
    """
    if graph_mode not in ("full", "truncated", "per_step"):
        raise ValueError("graph_mode must be 'full', 'truncated' or 'per_step'")

    s = s0
    traj_s, traj_a = [s], []
    traj_mu, traj_cov, traj_dmean, traj_dvar = [], [], [], []
    step_losses = []
    segment = []          # losses since the last flush

    def _flush():
        """Close the current graph segment: hand it to the driver, then detach."""
        nonlocal s, segment
        if flush_fn is not None and segment:
            flush_fn(torch.stack(segment).sum())
        segment = []
        s = s.detach()

    for t in range(T):
        # ---- 1. POLICY: action distribution / action given the current state ----
        a = policy(states=s, t=t, p_dropout=p_dropout)

        # ---- 2. GP TRANSITION (reparameterization trick, reused) ----
        s_next, delta_mean, delta_var = model.get_next_state(
            current_state=s, current_input=a, particle_pred=particle_pred
        )

        # ---- P(s' | s, a): mean and (diagonal) covariance, differentiable ----
        mu_next = s + delta_mean          # (P, ds)
        cov_next = delta_var              # (P, ds) diagonal

        # ---- 3. PER-STEP LOSS (supplied by the driver) ----
        if loss_fn is not None:
            loss_t = loss_fn(t=t, s=s, a=a, mu=mu_next, cov=cov_next, s_next=s_next)
            step_losses.append(loss_t)
            segment.append(loss_t)

        traj_a.append(a)
        traj_mu.append(mu_next)
        traj_cov.append(cov_next)
        traj_dmean.append(delta_mean)
        traj_dvar.append(delta_var)

        s = s_next
        traj_s.append(s)

        # ---- 4. GRAPH MANAGEMENT ----
        if graph_mode == "per_step" or (
            graph_mode == "truncated" and (t + 1) % truncate_every == 0
        ):
            _flush()
            traj_s[-1] = s      # keep the stored state consistent with the detached one

    loss_total = torch.stack(segment).sum() if segment else None

    return {
        "S": torch.stack(traj_s, dim=1),
        "A": torch.stack(traj_a, dim=1),
        "Mu": torch.stack(traj_mu, dim=1),
        "Cov": torch.stack(traj_cov, dim=1),
        "Dmean": torch.stack(traj_dmean, dim=1),
        "Dvar": torch.stack(traj_dvar, dim=1),
        "step_losses": step_losses,
        "loss_total": loss_total,
    }


def cov_full(cov_diag):
    """(P, ds) diagonal -> (P, ds, ds) full matrix, for losses that need one
    (e.g. a Gaussian KL against the expert's FULL 3x3 covariance)."""
    return torch.diag_embed(cov_diag)


def rollout_step_stats(Dmean, Dvar):
    """Per-timestep summary: (mean_over_particles, std_over_particles,
    mean_predictive_std), each (T, ds). The third is the GP's own uncertainty --
    if it grows along the horizon the rollout has left the training distribution."""
    with torch.no_grad():
        return (Dmean.mean(dim=0), Dmean.std(dim=0), torch.sqrt(Dvar).mean(dim=0))


def sample_initial_particles(states_pool, num_particles, generator=None,
                             dtype=torch.float64, device=torch.device("cpu")):
    """Draw initial particles by sampling rows from a pool of real states, keeping
    the rollout inside the region the GP was trained on."""
    import numpy as np

    pool = states_pool.detach().cpu().numpy() if torch.is_tensor(states_pool) \
        else np.asarray(states_pool)
    rng = generator or np.random.default_rng(0)
    idx = rng.integers(0, pool.shape[0], size=num_particles)
    return torch.tensor(pool[idx], dtype=dtype, device=device)
