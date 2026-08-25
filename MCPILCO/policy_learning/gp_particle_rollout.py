#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
    segment = []

    def _flush():
        """Close the current graph segment: hand it to the driver, then detach."""
        nonlocal s, segment
        if flush_fn is not None and segment:
            flush_fn(torch.stack(segment).sum())
        segment = []
        s = s.detach()

    for t in range(T):
        a = policy(states=s, t=t, p_dropout=p_dropout)

        s_next, delta_mean, delta_var = model.get_next_state(
            current_state=s, current_input=a, particle_pred=particle_pred
        )

        mu_next = s + delta_mean
        cov_next = delta_var

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

        if graph_mode == "per_step" or (
            graph_mode == "truncated" and (t + 1) % truncate_every == 0
        ):
            _flush()
            traj_s[-1] = s

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
