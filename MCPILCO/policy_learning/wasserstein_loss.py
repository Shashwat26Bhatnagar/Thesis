#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/wasserstein_loss.py

2-Wasserstein distance between Gaussians, in two forms:

  * wasserstein_gaussian(...)        numpy  -- reference implementation, used for
                                    testing / analysis. NOT differentiable.
  * w2_gaussian_torch(...)           torch  -- differentiable, batched over
                                    particles. Used inside the CDIL rollout.

EQUAL DIMENSION (Bures-Wasserstein):
    W2^2 = ||mu1-mu2||^2 + tr(S1) + tr(S2) - 2 tr((S1^1/2 S2 S1^1/2)^1/2)

DIFFERENT DIMENSION (Cai-Lim projection distance):
    gamma = eigenvalues of the SMALLER cov (desc), lambda = of the LARGER (desc)
    s_i*  = clamp(gamma_i, lambda_{n-m+i}, lambda_i)
    W2^2  = sum_i (sqrt(gamma_i) - sqrt(s_i*))^2

*** IMPORTANT: in the cross-dimensional branch the MEANS DROP OUT. ***
The Cai-Lim distance absorbs the mean difference into a free translation, so the
cost depends ONLY on the covariance spectra. For CDIL this means a pure cross-dim
loss gives the policy NO signal about where the trajectory goes -- only about the
shape of its uncertainty. See `mode` in cdil_w2_loss() for the alternatives.

DIFFERENTIABILITY NOTES (torch version)
    - Our GP covariance is DIAGONAL (num_gp independent scalar GPs), so its
      eigenvalues are just the sorted diagonal: torch.sort, no eigendecomposition,
      no unstable eigh backward. This is why the cross-dim branch is cheap and safe.
    - The expert covariance is a CONSTANT target; its eigenvalues are computed once
      and detached, so no gradient passes through the expert at all.
    - The equal-dim branch needs a matrix square root -> eigh on a symmetric PSD
      matrix. eigh's backward contains 1/(lambda_i - lambda_j) terms and can blow up
      when eigenvalues are nearly degenerate; `jitter` guards this.
"""
import numpy as np
import torch


# =====================================================================================
# NUMPY REFERENCE  (unchanged behaviour -- for tests and offline analysis)
# =====================================================================================
def _sym_sqrt(S):
    U, s, _ = np.linalg.svd(S)
    return (U * np.sqrt(np.clip(s, 0, None))) @ U.T


def wasserstein_gaussian(mu1, mu2, sigma1, sigma2):
    """2-Wasserstein distance between N(mu1, sigma1) and N(mu2, sigma2).
    Equal dim -> Bures-Wasserstein; different dim -> Cai-Lim (means drop out)."""
    mu1 = np.atleast_1d(np.asarray(mu1, float))
    mu2 = np.atleast_1d(np.asarray(mu2, float))
    S1 = np.atleast_2d(np.asarray(sigma1, float))
    S2 = np.atleast_2d(np.asarray(sigma2, float))
    m, n = S1.shape[0], S2.shape[0]
    if m == n:
        s1h = _sym_sqrt(S1)
        cross = _sym_sqrt(s1h @ S2 @ s1h)
        cov = np.trace(S1) + np.trace(S2) - 2.0 * np.trace(cross)
        mean = float((mu1 - mu2) @ (mu1 - mu2))
        return np.sqrt(max(mean + cov, 0.0))
    g = np.sort(np.linalg.svd(S1, compute_uv=False))[::-1]
    l = np.sort(np.linalg.svd(S2, compute_uv=False))[::-1]
    if m > n:
        g, l, m, n = l, g, n, m
    i = np.arange(m)
    s_star = np.clip(g, l[n - m + i], l[i])
    cost = np.sum((np.sqrt(g) - np.sqrt(s_star)) ** 2)
    return np.sqrt(max(cost, 0.0))


# =====================================================================================
# TORCH  -- differentiable, batched over particles
# =====================================================================================
def _sym_sqrt_torch(S, eps=1e-12):
    """Matrix square root of a symmetric PSD matrix (..., d, d) via eigh."""
    w, V = torch.linalg.eigh(S)
    w = torch.clamp(w, min=eps)
    return (V * torch.sqrt(w).unsqueeze(-2)) @ V.transpose(-1, -2)


def w2_cross_dim_torch(cov_big_diag, eig_small, eps=1e-12):
    """Cai-Lim projection distance, LARGE side diagonal (our GP), SMALL side given
    by its eigenvalues (the expert).

    cov_big_diag : (P, n) diagonal covariance of the larger-dimensional Gaussian
                   (differentiable)
    eig_small    : (m,)   eigenvalues of the smaller covariance, ANY order
                   (constant target; detached)
    returns      : (P,) W2 distances.   NOTE: means play no role here.
    """
    P, n = cov_big_diag.shape
    m = eig_small.shape[0]
    if m > n:
        raise ValueError("eig_small must be the smaller dimension")

    # eigenvalues of a diagonal matrix are its diagonal -> just sort (differentiable)
    lam, _ = torch.sort(cov_big_diag, dim=1, descending=True)      # (P, n)
    gam, _ = torch.sort(eig_small.detach(), descending=True)       # (m,)

    idx = torch.arange(m, device=cov_big_diag.device)
    lo = lam[:, n - m + idx]                                       # (P, m) lower band
    hi = lam[:, idx]                                               # (P, m) upper band
    g = gam.unsqueeze(0).expand(P, -1)                             # (P, m)

    # s* = clamp(gamma, lo, hi) -- torch.clamp with tensor bounds is differentiable
    # w.r.t. whichever argument is selected (subgradient at the boundaries).
    s_star = torch.minimum(torch.maximum(g, lo), hi)

    cost = ((torch.sqrt(torch.clamp(g, min=eps))
             - torch.sqrt(torch.clamp(s_star, min=eps))) ** 2).sum(dim=1)
    return torch.sqrt(torch.clamp(cost, min=0.0) + eps)


def w2_equal_dim_torch(mu1, S1, mu2, S2, eps=1e-12, jitter=1e-8):
    """Bures-Wasserstein between N(mu1,S1) and N(mu2,S2), same dimension.

    mu1 (P, d), S1 (P, d, d)   -- model side, differentiable
    mu2 (d,) or (P, d), S2 (d, d) or (P, d, d) -- expert side (broadcast, detached)
    returns (P,)
    """
    if mu2.dim() == 1:
        mu2 = mu2.unsqueeze(0).expand_as(mu1)
    if S2.dim() == 2:
        S2 = S2.unsqueeze(0).expand_as(S1)
    d = S1.shape[-1]
    I = torch.eye(d, dtype=S1.dtype, device=S1.device).expand_as(S1)
    S1 = S1 + jitter * I                    # guard eigh backward near degeneracy
    S2 = S2 + jitter * I

    s1h = _sym_sqrt_torch(S1, eps)
    cross = _sym_sqrt_torch(s1h @ S2 @ s1h, eps)
    tr = (torch.diagonal(S1, dim1=-2, dim2=-1).sum(-1)
          + torch.diagonal(S2, dim1=-2, dim2=-1).sum(-1)
          - 2.0 * torch.diagonal(cross, dim1=-2, dim2=-1).sum(-1))
    mean = ((mu1 - mu2) ** 2).sum(-1)
    return torch.sqrt(torch.clamp(mean + tr, min=0.0) + eps)


# =====================================================================================
# CDIL entry point
# =====================================================================================
def cdil_w2_loss(mu_model, cov_model_diag, expert_b, expert_cov,
                 mode="cross_dim", project_mu=None, project_cov=None,
                 mean_weight=1.0):
    """W2 loss between the GP transition Gaussian and the expert Gaussian.

    mu_model       (P, ds) mean of P(s'|s,a)                 -- differentiable
    cov_model_diag (P, ds) DIAGONAL covariance               -- differentiable
    expert_b       (de,)   expert mean                        -- constant
    expert_cov     (de,de) expert FULL covariance             -- constant

    mode:
      "cross_dim"  Cai-Lim, ds != de. NO projection needed, but the MEANS DROP OUT
                   -- covariance-spectrum matching only.
      "projected"  project the model to the expert space, then Bures-Wasserstein.
                   Means DO matter. Requires project_mu / project_cov.
      "hybrid"     cross-dim covariance term + an explicit projected mean term,
                   weighted by mean_weight. Keeps the projection-free covariance
                   comparison but restores a trajectory-tracking signal.
    """
    eig_e = torch.linalg.eigvalsh(expert_cov.detach())        # (de,) ascending

    if mode == "cross_dim":
        return w2_cross_dim_torch(cov_model_diag, eig_e).mean()

    if mode == "projected":
        if project_mu is None or project_cov is None:
            raise ValueError("mode='projected' needs project_mu and project_cov")
        mu_p = project_mu(mu_model)                            # (P, de)
        cov_p = project_cov(cov_model_diag)                    # (P, de, de)
        return w2_equal_dim_torch(mu_p, cov_p, expert_b.detach(),
                                  expert_cov.detach()).mean()

    if mode == "hybrid":
        cov_term = w2_cross_dim_torch(cov_model_diag, eig_e).mean()
        if project_mu is None:
            raise ValueError("mode='hybrid' needs project_mu")
        mu_p = project_mu(mu_model)
        mean_term = ((mu_p - expert_b.detach().unsqueeze(0)) ** 2).sum(dim=1).mean()
        return cov_term + mean_weight * mean_term

    raise ValueError("mode must be 'cross_dim', 'projected' or 'hybrid'")


# =====================================================================================
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print("--- numpy reference sanity checks ---")
    A = rng.standard_normal((4, 4)); A = A @ A.T
    print("identical:", wasserstein_gaussian([1, 2, 3, 4], [1, 2, 3, 4], A, A))

    S1 = np.diag([4.0, 1.0]); S2 = np.diag([9.0, 16.0])
    analytic = np.sqrt(9 + (2 - 3) ** 2 + (1 - 4) ** 2)
    print("diag equal-dim:", wasserstein_gaussian([0., 0.], [3., 0.], S1, S2),
          "vs", analytic)

    lam = np.diag([9.0, 4.0, 1.0])
    for var in [0.25, 2.0, 25.0]:
        sig = np.sqrt(var)
        exp = 1 - sig if sig < 1 else (0.0 if sig <= 3 else sig - 3)
        got = wasserstein_gaussian([0.0], [0, 0, 0], [[var]], lam)
        print(f"cross-dim var={var}: got {got:.4f} exp {exp:.4f}")

    print("\n--- torch vs numpy agreement (cross-dim, 8-D model vs 3-D expert) ---")
    P, ds, de = 4, 8, 3
    cov_d = torch.rand(P, ds, dtype=torch.float64) + 0.1
    E = torch.randn(de, de, dtype=torch.float64); E = E @ E.T + 0.5 * torch.eye(de, dtype=torch.float64)
    eig_e = torch.linalg.eigvalsh(E)
    got = w2_cross_dim_torch(cov_d, eig_e)
    for p in range(P):
        ref = wasserstein_gaussian(np.zeros(ds), np.zeros(de),
                                   np.diag(cov_d[p].numpy()), E.numpy())
        print(f"  particle {p}: torch={got[p].item():.8f}  numpy={ref:.8f}  "
              f"diff={abs(got[p].item()-ref):.2e}")

    print("\n--- gradient flows through the torch version ---")
    cov_g = (torch.rand(P, ds, dtype=torch.float64) + 0.1).requires_grad_(True)
    w2_cross_dim_torch(cov_g, eig_e).mean().backward()
    print(f"  d(W2)/d(cov) max|grad| = {cov_g.grad.abs().max().item():.6e}")
