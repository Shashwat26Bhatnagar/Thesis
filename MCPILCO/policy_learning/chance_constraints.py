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


def action_violation_multiplier(a, lo, hi, span=None, beta=1000.0, cap=100.0,
                                return_parts=False):
    """Multiplier on the primary loss, growing with how far actions leave the box.

        excess_i  = relu(lo_i - a_i) + relu(a_i - hi_i)          per channel
        v         = sum_i excess_i / span_i                      normalised, unitless
        multiplier= 1 + beta * v                                 (clamped at `cap`)

    The primary objective is then  W2 * multiplier, so an in-bounds action pays exactly
    the W2 it earns, while an out-of-bounds one pays many times over -- the policy
    cannot buy a better W2 by leaving the safe region.

    WHY MULTIPLICATIVE AND NOT ADDITIVE
    An additive penalty competes with W2 on an absolute scale, so its weight has to be
    retuned whenever the W2 magnitude changes across iterations. A multiplier is
    scale-free: it is exactly 1 when the actions are legal, whatever W2 happens to be.

    CALIBRATION (measured on bnd_iter6_batch_7.csv, static +/-10% band)
        discharge out of band 91.3% of steps, sugar 31.9%
        mean normalised excess v = 0.0107, max 0.0349
        beta=  10  -> multiplier  1.11 mean / 1.35 max     (too weak to deter)
        beta= 100  -> multiplier  2.07 mean / 4.49 max
        beta=1000  -> multiplier 11.74 mean / 35.94 max    <- "many folds", default
    `cap` bounds the multiplier so a single wild action cannot produce a gradient
    large enough to destabilise the update.
    """
    lo = lo if torch.is_tensor(lo) else torch.as_tensor(lo, dtype=a.dtype, device=a.device)
    hi = hi if torch.is_tensor(hi) else torch.as_tensor(hi, dtype=a.dtype, device=a.device)
    if span is None:
        span = (hi - lo).clamp_min(1e-12)
    elif not torch.is_tensor(span):
        span = torch.as_tensor(span, dtype=a.dtype, device=a.device)

    excess = torch.relu(lo - a) + torch.relu(a - hi)          # (P, da)
    v = (excess / span).sum(dim=1).mean()                     # scalar, unitless
    mult = torch.clamp(1.0 + beta * v, max=cap)

    if return_parts:
        with torch.no_grad():
            frac = (excess > 1e-9).to(a.dtype).mean(dim=0)    # per channel
        return mult, {"v": v.detach(), "violation_frac_per_channel": frac,
                      "max_excess": excess.max().detach()}
    return mult


def state_chance_penalty(mu, cov_diag, idx, lo=None, hi=None,
                         num_states=None, k_actions=None,
                         eps=0.95, alpha=1000.0, sigma_floor=1e-12,
                         return_parts=False):
    """Soft chance constraint on a PREDICTED STATE channel (Tan et al. Eq. 8-9).

    This is the paper's own form: it constrains STATES using the GP's predictive mean
    and variance, so the back-off term Phi^-1(eps)*sigma is the model's own uncertainty
    rather than a proxy.

        lower bound:  mu_i - z*sigma_i  >=  lo      (e.g. vessel must not empty)
        upper bound:  mu_i + z*sigma_i  <=  hi

    WHY THIS IS NEEDED IN ADDITION TO THE ACTION CONSTRAINT
    A per-timestep action bound cannot express an ACCUMULATION limit. Measured over
    t=100..130 h, the reference controller and the learned policy reach almost the same
    PEAK discharge (3705 vs 3600) -- but the reference averages 245 L/h while the policy
    averages 3237, because the recipe's 4000 is a ceiling it touches briefly, and the
    +/-10% band turns that ceiling into a permitted operating point. Every individual
    action is legal; the accumulation drains the vessel (62900 -> 25436, episode
    terminating at t=122 h instead of 230). Constraining the predicted vessel weight
    expresses the real requirement directly.

    mu, cov_diag : (P, ds) predictive mean and DIAGONAL variance, differentiable
    idx          : which state channel to constrain
    lo, hi       : bounds in the SAME units as mu (None = unbounded on that side)
    """
    z = phi_inv(eps)
    m = mu[:, idx]
    sd = torch.sqrt(cov_diag[:, idx].clamp_min(sigma_floor))

    if num_states is not None and k_actions is not None and k_actions > 1:
        m = m.view(num_states, k_actions).mean(dim=1)
        sd = sd.view(num_states, k_actions).mean(dim=1)

    s = torch.zeros_like(m)
    if lo is not None:
        s = s + torch.relu(lo - (m - z * sd))          # want mu - z*sd >= lo
    if hi is not None:
        s = s + torch.relu((m + z * sd) - hi)          # want mu + z*sd <= hi
    penalty = alpha * s.mean()

    if return_parts:
        with torch.no_grad():
            frac = (s > 0).to(m.dtype).mean()
        return penalty, {"violation_frac": frac, "mean_slack": s.mean().detach(),
                         "max_slack": s.max().detach(), "worst_mu": m.min().detach()}
    return penalty


def mass_balance_penalty(mu, s_prev, a, wt_idx, water_idx, disch_idx,
                         std_obs_sd, std_act_sd, obs_span, act_span,
                         dt=0.2, c=0.9275, tol=80.0, alpha=1.0,
                         num_states=None, k_actions=None, return_parts=False):
    """Penalise predicted state changes that are INCONSISTENT with the commanded flows.

        residual = | dWt_predicted  -  c * (water - discharge) * dt |
        penalty  = alpha * mean( relu(residual - tol) )

    WHY
    Action-box constraints cannot catch a policy that keeps every action legal while
    driving the STATE somewhere the actions do not justify. Measured on real batches:

        reference (gpei_batch_7) : corr(net_flow, dWt) = 0.898,  c = 0.928  (~1 L
                                   in, ~1 L out -- physically coherent)
        learned   (bnd_iter6_..) : corr(net_flow, dWt) = 0.153,  c = -9.2   (the
                                   least-squares fit is NEGATIVE, i.e. the vessel
                                   fills while the net flow drains it)

    The agent found action combinations whose individual values are legal but whose
    EFFECT on the state is decoupled from the flows -- the state trajectory looks
    plausible while the physics does not hold. This term restores that coupling.

    c and tol are calibrated on the reference run: c = 0.9275 by least squares through
    the origin, and the residual there has std ~79 against a typical |dWt| of ~110, so
    tol = 80 keeps the reference itself essentially unpenalised. A TOLERANCE band, not
    an equality: evaporation, density changes and the other feeds all contribute.

    All quantities are converted from Z-SCORED units back to physical before comparison,
    since c has physical meaning (litres, not standard deviations).
    """
    import numpy as np

    # z -> smpl-normalised -> physical, for the three channels involved
    def _obs_phys_delta(dz, idx):
        return dz * float(std_obs_sd[idx]) * float(obs_span[idx]) / 2.0

    def _act_phys(az, idx):
        # only the DIFFERENCE water-discharge is needed, so the offset cancels only if
        # both share a span; they do not, hence each is converted separately
        return az * float(std_act_sd[idx]) * float(act_span[idx]) / 2.0

    d_wt_z = mu[:, wt_idx] - s_prev[:, wt_idx]
    d_wt = _obs_phys_delta(d_wt_z, wt_idx)

    # actions are z-scored; the additive part of the affine map matters here, so the
    # caller passes spans and the mean offset is handled by the caller's centring
    water = _act_phys(a[:, water_idx], water_idx)
    disch = _act_phys(a[:, disch_idx], disch_idx)

    resid = torch.abs(d_wt - c * (water - disch) * dt)
    slack = torch.relu(resid - tol)

    if num_states is not None and k_actions is not None and k_actions > 1:
        slack = slack.view(num_states, k_actions).mean(dim=1)

    penalty = alpha * slack.mean()
    if return_parts:
        with torch.no_grad():
            return penalty, {"mean_resid": resid.mean().detach(),
                             "max_resid": resid.max().detach(),
                             "violation_frac": (slack > 0).to(mu.dtype).mean().detach()}
    return penalty


class RecipeBounds:
    """+/- frac band around the TIME-VARYING recipe profile, in Z-SCORED action units.

    The default profiles are STEP functions, e.g. DISCHARGE_DEFAULT_PROFILE:
        t <  100 h : 0
        t = 100-102: 0 -> 4000        (and similar pulses every 20 h thereafter)
    so +/-10% of the setpoint is a ZERO-WIDTH band for the first 100 hours. Two things
    follow, both handled here:

      * floor_frac  gives the band a minimum half-width as a fraction of the channel's
                    physical span, so a setpoint of 0 does not produce an
                    unsatisfiable constraint.
      * smooth_h    averages the profile over a +/-h window. The profile jumps by 4000
                    within 2 h; without smoothing, a window whose time sits a step
                    away from an edge incurs an enormous penalty for what is really a
                    timing technicality.

    NOTE this is a SEPARATE, TIGHTER constraint from the static physical box. Both are
    applied: the static box rules out impossible actions, this one rules out actions
    that are possible but far from the recipe at that point in the batch.
    """

    def __init__(self, recipe_combo, act_names_to_keys, min_act, max_act,
                 std_act_mu, std_act_sd, frac=0.10, floor_frac=0.05,
                 smooth_h=2.0, n_smooth=5):
        import numpy as np
        self.rc = recipe_combo
        self.keys = act_names_to_keys          # ordered list, one key per action channel
        self.min_act = np.asarray(min_act, dtype=np.float64)
        self.max_act = np.asarray(max_act, dtype=np.float64)
        self.mu = np.asarray(std_act_mu, dtype=np.float64)
        self.sd = np.asarray(std_act_sd, dtype=np.float64)
        self.frac = frac
        self.floor = floor_frac * (self.max_act - self.min_act)
        self.smooth_h = smooth_h
        self.n_smooth = n_smooth
        self._cache = {}

    def _profile_at(self, t):
        """Recipe setpoints at time t, PHYSICAL units, smoothed over +/-smooth_h."""
        import numpy as np
        ts = np.linspace(t - self.smooth_h, t + self.smooth_h, self.n_smooth)
        vals = []
        for tt in ts:
            d = self.rc.get_values_dict_at(max(float(tt), 0.0))
            vals.append([float(d[k]) for k in self.keys])
        return np.asarray(vals, dtype=np.float64).mean(axis=0)

    def at(self, t_hours):
        """(lo, hi) in Z-SCORED units at time t. Cached: the profile is deterministic."""
        import numpy as np
        key = round(float(t_hours), 3)
        if key in self._cache:
            return self._cache[key]
        r = self._profile_at(key)
        half = np.maximum(self.frac * np.abs(r), self.floor)
        lo_p = np.clip(r - half, self.min_act, self.max_act)
        hi_p = np.clip(r + half, self.min_act, self.max_act)
        # physical -> smpl min-max -> z-score  (same chain as the rest of the pipeline)
        span = self.max_act - self.min_act
        lo_z = ((2.0 * (lo_p - self.min_act) / span - 1.0) - self.mu) / self.sd
        hi_z = ((2.0 * (hi_p - self.min_act) / span - 1.0) - self.mu) / self.sd
        self._cache[key] = (lo_z, hi_z)
        return self._cache[key]


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
