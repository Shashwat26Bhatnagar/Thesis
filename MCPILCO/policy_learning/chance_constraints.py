#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/chance_constraints.py

Soft chance constraints, following Tan et al., "Iterative Model-Learning Scheme via
GP-NMPC with Chance Constraints", Eqs. (8)-(9).

FORMULATION (paper Eq. 8)
    A probabilistic linear inequality  Pr(h^T x <= b) >= eps  is reformulated in terms
    of the first two moments as a DETERMINISTIC tightening:

        h^T mu  <=  b - Phi^-1(eps) * sqrt(h^T Sigma h)

    where Phi^-1 is the inverse standard-Gaussian CDF. The second term is the
    "back-off": the larger the predictive uncertainty, the further inside the bound
    the mean must sit. That is the whole point -- it makes the controller cautious
    exactly where the GP is unsure, which is where over-extrapolation happens.

SOFT IMPLEMENTATION (paper Eq. 9)
    Hard constraints make the problem infeasible on sparse data, so the paper adds a
    slack s >= 0 penalised by alpha in the objective:

        min  E[l(x,u)] + alpha * s
        s.t. s >= h^T mu - b + Phi^-1(1-eps) * sqrt(h^T Sigma h),   s >= 0

    With s at its lower bound this is exactly  alpha * relu(violation), which is what
    is implemented here -- differentiable, and zero when the constraint holds.
    Paper settings: eps = 0.95, alpha = 1000.

APPLIED TO ACTIONS (this project)
    The paper constrains STATES. Here the same machinery is applied per ACTION channel,
    because the CDIL objective is covariance-only and therefore says nothing about what
    the actions do -- the policy discharged ~200 L/h for the whole batch while the
    reference recipe discharges 0 for ~97% of it. Nothing in the loss penalised that.

    For a two-sided box lo_i <= a_i <= hi_i the two rows of h are +e_i and -e_i:

        upper:  mu_i + z * sigma_i  <=  hi_i
        lower:  lo_i  <=  mu_i - z * sigma_i          z = Phi^-1(eps)

    sigma_i is the spread of the action across the K policy samples drawn for the same
    state (dropout), i.e. the policy's own stochasticity -- the natural analogue of the
    GP predictive variance the paper uses.
"""
import math

import torch


def phi_inv(eps):
    """Inverse standard-Gaussian CDF. eps=0.95 -> 1.6449."""
    return math.sqrt(2.0) * torch.erfinv(torch.tensor(2.0 * eps - 1.0,
                                                      dtype=torch.float64)).item()


def action_chance_penalty(a, lo, hi, num_states=None, k_actions=None,
                          eps=0.95, alpha=1000.0, sigma_floor=1e-8,
                          return_parts=False):
    """Soft chance-constraint penalty on actions (paper Eq. 9).

    a        : (P, da) actions from the rollout, differentiable
    lo, hi   : (da,) or (P, da) box bounds in the SAME units as `a`
    num_states, k_actions : if given, a is reshaped (S, K, da) and sigma is the spread
               ACROSS THE K SAMPLES for each state -- the policy's own uncertainty.
               If omitted, sigma is taken across the whole batch (cruder).
    eps      : confidence, paper uses 0.95
    alpha    : soft-constraint weight, paper uses 1000
    """
    z = phi_inv(eps)
    lo = lo if torch.is_tensor(lo) else torch.as_tensor(lo, dtype=a.dtype, device=a.device)
    hi = hi if torch.is_tensor(hi) else torch.as_tensor(hi, dtype=a.dtype, device=a.device)

    if num_states is not None and k_actions is not None and k_actions > 1:
        g = a.view(num_states, k_actions, a.shape[-1])
        mu = g.mean(dim=1)                              # (S, da)
        sigma = g.std(dim=1, unbiased=False).clamp_min(sigma_floor)
    else:
        mu = a
        sigma = a.std(dim=0, keepdim=True).clamp_min(sigma_floor).expand_as(a)

    # Eq. 8 tightening, both sides of the box. Positive => violated.
    v_hi = (mu + z * sigma) - hi                        # want <= 0
    v_lo = lo - (mu - z * sigma)                        # want <= 0

    # Eq. 9 with the slack at its lower bound: s = max(0, violation)
    s = torch.relu(v_hi) + torch.relu(v_lo)             # (S, da) or (P, da)
    penalty = alpha * s.mean()

    if return_parts:
        with torch.no_grad():
            frac = ((v_hi > 0) | (v_lo > 0)).to(a.dtype).mean(dim=0)   # per channel
        return penalty, {"violation_frac_per_channel": frac,
                         "mean_slack": s.mean().detach(),
                         "max_slack": s.max().detach(),
                         "z": z}
    return penalty


def recipe_band(profiles, t_hours, frac=0.10, floor_abs=None):
    """+/- frac band around the TIME-VARYING recipe profile.

    profiles : (da, T) array of the recipe setpoints, or a callable t -> (da,)
    t_hours  : scalar time
    floor_abs: minimum half-width, so a setpoint of 0 does not give a zero-width band
               (the reference recipe discharges EXACTLY 0 for most of the batch, and a
               zero-width band would be unsatisfiable)

    NOTE: a band centred on the DATASET MEAN action instead of the profile is wrong.
    The recipe discharges 0 for ~220 of 230 h, so a fixed [0.9*200, 1.1*200] band
    forces >=180 L/h throughout -- penalising the policy for doing the right thing.
    """
    import numpy as np
    r = profiles(t_hours) if callable(profiles) else np.asarray(profiles)
    r = np.asarray(r, dtype=np.float64).reshape(-1)
    half = frac * np.abs(r)
    if floor_abs is not None:
        half = np.maximum(half, np.asarray(floor_abs, dtype=np.float64))
    return r - half, r + half


if __name__ == "__main__":
    torch.manual_seed(0)
    S, K, da = 100, 5, 6
    lo = torch.tensor([0., 7., 21., 29., .5, 0.], dtype=torch.float64)
    hi = torch.tensor([4100., 151., 36., 76., 1.2, 510.], dtype=torch.float64)

    print(f"Phi^-1(0.95) = {phi_inv(0.95):.4f}   (paper's eps = 95%)\n")
    for name, centre in (("inside the box", (lo + hi) / 2),
                         ("on the upper bound", hi.clone()),
                         ("far outside", hi * 1.5)):
        a = centre.expand(S * K, da).clone() + 0.01 * torch.randn(S * K, da,
                                                                  dtype=torch.float64)
        a.requires_grad_(True)
        pen, info = action_chance_penalty(a, lo, hi, S, K, return_parts=True)
        pen.backward()
        print(f"{name:22s} penalty={pen.item():12.4f}  "
              f"max slack={info['max_slack'].item():10.4f}  "
              f"|grad|={a.grad.abs().max().item():.3e}")
    print("\nnote: 'on the upper bound' is already penalised -- the back-off term")
    print("      z*sigma pushes the required mean strictly INSIDE the bound.")
