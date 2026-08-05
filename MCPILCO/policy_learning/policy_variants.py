#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/policy_variants.py

Three interchangeable policy architectures for the CDIL / Dyna pipeline:

    "rbf"  Policy.Sum_of_gaussians          (existing MC-PILCO policy, unchanged)
    "mlp"  plain feed-forward network
    "kan"  Kolmogorov-Arnold network with RADIAL BASIS (Gaussian) edge functions

All three satisfy the SAME contract, so nothing downstream changes:

    policy(states=(P,ds), t=int, p_dropout=float) -> (P,da) in [-u_max, u_max]
    policy.state_dim, policy.input_dim
    policy.parameters()                    for Adam + clip_grad_norm_
    build_policy(...) -> (policy, meta)    meta is saved into the checkpoint so a
                                           warm start can rebuild the same object

ARCHITECTURAL DIFFERENCE (the point of the ablation)

    RBF   u = W . phi(||s - c_k||)      basis is JOINT over all input dims: one
                                        Gaussian per centre in R^ds. Local: a state
                                        far from every centre gives phi ~ 0, hence
                                        u ~ 0 AND a vanishing gradient (this is the
                                        "centre abandonment" failure seen when
                                        rollouts diverged to |s| ~ 140).

    MLP   u = W2 . tanh(W1 . s + b1)    activations fixed, weights learned. Global:
                                        no dead regions, but no locality either.

    KAN   u_j = sum_i psi_ij(s_i)       activations LEARNED, on edges, and each is
          psi(x) = sum_g w_g exp(...)   UNIVARIATE. Per-dimension grids mean each
                                        input is resolved independently -- useful
                                        here because the 8 channels have very
                                        different dynamics (pH vs vessel weight).

PARAMETER BUDGET -- matched to the existing RBF policy (2808 params for ds=8, da=6):
    rbf : 200 centres x 8 + 6 x 200 + 8            = 2808
    mlp : 8x64 + 64 + 64x64 + 64 + 64x6 + 6        = 5062   (hidden=48 -> 3126)
    kan : (8x10 + 10x6) x grid(16) + biases        = 2246   (grid=20 -> 2806)
Defaults below are chosen to land near 2808; print_param_count() reports the actual.

DROPOUT. The nested objective E_s(E_{a|s}) needs the K replicas of a state to draw
DIFFERENT actions. The existing RBF policy's dropout gives a spread of only ~1e-5,
i.e. E_{a|s} is nearly degenerate. MLP and KAN here draw a PER-ROW mask, so replicas
genuinely differ. This is a deliberate behavioural difference -- flag it when
comparing, since it changes what the inner expectation actually estimates.
"""
import numpy as np
import torch
import torch.nn as nn


# =====================================================================================
def _squash(x, u_max):
    """Bound to [-u_max, u_max]. tanh, matching the smooth saturating form used by
    Sum_of_gaussians (MC-PILCO's own squash is the periodic 9sin+sin3 variant; tanh
    is used here because it is monotone, which keeps the action ordering meaningful)."""
    return u_max * torch.tanh(x)


# =====================================================================================
class MLPPolicy(nn.Module):
    """Plain feed-forward policy: fixed activations, learned weights."""

    def __init__(self, state_dim, input_dim, hidden=(48, 48), u_max=3.0,
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.state_dim, self.input_dim, self.u_max = state_dim, input_dim, u_max
        dims = [state_dim] + list(hidden) + [input_dim]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1], dtype=dtype, device=device))
        self.layers = nn.ModuleList(layers)
        for l in self.layers:                       # small init -> starts near u=0
            nn.init.normal_(l.weight, std=0.1)
            nn.init.zeros_(l.bias)

    def forward(self, states, t=None, p_dropout=0.0):
        x = states
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = torch.tanh(x)
                if p_dropout > 0:
                    # PER-ROW mask: replicas of one state must get different actions
                    m = (torch.rand_like(x) > p_dropout).to(x.dtype)
                    x = x * m / max(1.0 - p_dropout, 1e-6)
        return _squash(x, self.u_max)


# =====================================================================================
class RBFKANLayer(nn.Module):
    """One KAN layer with radial-basis (Gaussian) edge functions.

        y_j = sum_i psi_ij(x_i),   psi_ij(x) = sum_g w_ijg * exp(-((x - mu_g)/h)^2)

    Each INPUT DIMENSION has its own 1-D grid of Gaussians, so the learned activation
    on every edge is univariate -- the Kolmogorov-Arnold structure. Contrast with
    Sum_of_gaussians, where a single Gaussian is evaluated on the JOINT distance
    ||s - c|| and therefore mixes all dimensions into one bump.
    """

    def __init__(self, in_dim, out_dim, grid_size=20, grid_range=(-3.0, 3.0),
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.in_dim, self.out_dim, self.grid_size = in_dim, out_dim, grid_size
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size,
                              dtype=dtype, device=device)
        self.register_buffer("grid", grid)                       # (G,)
        h = (grid_range[1] - grid_range[0]) / max(grid_size - 1, 1)
        self.register_buffer("h", torch.tensor(h, dtype=dtype, device=device))
        # spline coefficients: one weight per (input, output, grid point)
        self.coef = nn.Parameter(
            0.1 * torch.randn(in_dim, out_dim, grid_size, dtype=dtype, device=device))
        # a linear residual path keeps gradients alive when x falls off the grid
        # (the RBF policy's failure mode: basis -> 0 => gradient -> 0)
        self.res = nn.Parameter(
            0.1 * torch.randn(in_dim, out_dim, dtype=dtype, device=device))
        self.bias = nn.Parameter(torch.zeros(out_dim, dtype=dtype, device=device))

    def forward(self, x):                                        # x: (P, in_dim)
        # basis: (P, in_dim, G)
        z = (x.unsqueeze(-1) - self.grid) / self.h
        b = torch.exp(-z * z)
        y = torch.einsum("pig,iog->po", b, self.coef)            # spline part
        y = y + x @ self.res                                     # residual part
        return y + self.bias


class KANPolicy(nn.Module):
    """Two-layer RBF-KAN policy."""

    def __init__(self, state_dim, input_dim, hidden=10, grid_size=20,
                 grid_range=(-3.0, 3.0), u_max=3.0,
                 dtype=torch.float64, device=torch.device("cpu")):
        super().__init__()
        self.state_dim, self.input_dim, self.u_max = state_dim, input_dim, u_max
        self.grid_range = grid_range
        self.l1 = RBFKANLayer(state_dim, hidden, grid_size, grid_range, dtype, device)
        self.l2 = RBFKANLayer(hidden, input_dim, grid_size, grid_range, dtype, device)

    def forward(self, states, t=None, p_dropout=0.0):
        h = self.l1(states)
        if p_dropout > 0:
            m = (torch.rand_like(h) > p_dropout).to(h.dtype)     # per-row mask
            h = h * m / max(1.0 - p_dropout, 1e-6)
        return _squash(self.l2(h), self.u_max)


# =====================================================================================
def build_policy(kind, state_dim, input_dim, u_max=3.0, dtype=torch.float64,
                 device=torch.device("cpu"), rng=None,
                 # --- rbf ---
                 num_basis=200, centers_init=None, lengthscales_init=None,
                 s_lo=None, s_hi=None, center_range_pad=1.10,
                 # --- mlp ---
                 mlp_hidden=(48, 48),
                 # --- kan ---
                 kan_hidden=10, kan_grid=20, kan_range=(-3.0, 3.0)):
    """Construct a policy of the requested kind.

    Returns (policy, meta). `meta` records everything needed to rebuild the same
    architecture on warm start, and is written into the policy checkpoint.

    For "rbf", centres are drawn over the OBSERVED state range (s_lo/s_hi) rather
    than randn's [-3,3]: the training states span far wider than that, and centres
    that under-cover the data are exactly what caused basis abandonment.
    """
    rng = rng or np.random.default_rng(0)
    meta = {"kind": kind, "state_dim": state_dim, "input_dim": input_dim,
            "u_max": u_max}

    if kind == "rbf":
        import policy_learning.Policy as Policy
        if centers_init is None:
            lo = np.array([float(v) * center_range_pad for v in s_lo], dtype=np.float64)
            hi = np.array([float(v) * center_range_pad for v in s_hi], dtype=np.float64)
            span = hi - lo
            centers_init = np.array(
                [[lo[j] + span[j] * float(rng.random()) for j in range(state_dim)]
                 for _ in range(num_basis)], dtype=np.float64)
            lengthscales_init = np.array([float(v) / 4.0 for v in span],
                                         dtype=np.float64)
        policy = Policy.Sum_of_gaussians(
            state_dim=state_dim, input_dim=input_dim, num_basis=num_basis,
            u_max=u_max, flg_squash=True, flg_drop=True,
            centers_init=centers_init, lengthscales_init=lengthscales_init,
            weight_init=0.1 * rng.standard_normal((input_dim, num_basis)),
            dtype=dtype, device=device)
        meta.update(num_basis=num_basis,
                    centers_init=np.asarray(centers_init).tolist(),
                    lengthscales_init=np.asarray(lengthscales_init).tolist())

    elif kind == "mlp":
        policy = MLPPolicy(state_dim, input_dim, hidden=tuple(mlp_hidden),
                           u_max=u_max, dtype=dtype, device=device)
        meta.update(mlp_hidden=list(mlp_hidden))

    elif kind == "kan":
        policy = KANPolicy(state_dim, input_dim, hidden=kan_hidden,
                           grid_size=kan_grid, grid_range=tuple(kan_range),
                           u_max=u_max, dtype=dtype, device=device)
        meta.update(kan_hidden=kan_hidden, kan_grid=kan_grid,
                    kan_range=list(kan_range))

    else:
        raise ValueError(f"kind must be 'rbf', 'mlp' or 'kan', got {kind!r}")

    meta["n_params"] = int(sum(p.numel() for p in policy.parameters()
                               if p.requires_grad))
    return policy, meta


def rebuild_policy(meta, dtype=torch.float64, device=torch.device("cpu")):
    """Rebuild a policy from saved meta (for warm start / evaluation)."""
    kind = meta["kind"]
    kw = dict(state_dim=meta["state_dim"], input_dim=meta["input_dim"],
              u_max=meta.get("u_max", 3.0), dtype=dtype, device=device)
    if kind == "rbf":
        return build_policy("rbf",
                            centers_init=np.asarray(meta["centers_init"]),
                            lengthscales_init=np.asarray(meta["lengthscales_init"]),
                            num_basis=meta["num_basis"], **kw)[0]
    if kind == "mlp":
        return build_policy("mlp", mlp_hidden=meta["mlp_hidden"], **kw)[0]
    if kind == "kan":
        return build_policy("kan", kan_hidden=meta["kan_hidden"],
                            kan_grid=meta["kan_grid"],
                            kan_range=meta["kan_range"], **kw)[0]
    raise ValueError(kind)


# =====================================================================================
if __name__ == "__main__":
    ds, da, P, K = 8, 6, 20, 5
    lo, hi = -np.ones(ds) * 10, np.ones(ds) * 10
    print(f"{'kind':6s}{'params':>9}  {'out range':>22}  {'replica spread (E_a|s)':>24}")
    print("-" * 68)
    for kind in ("rbf", "mlp", "kan"):
        pol, meta = build_policy(kind, ds, da, s_lo=lo, s_hi=hi)
        s = torch.randn(P, ds, dtype=torch.float64)
        with torch.no_grad():
            u = pol(states=s, t=0, p_dropout=0.0)
            rep = pol(states=s[:1].expand(K, -1).contiguous(), t=0, p_dropout=0.25)
        print(f"{kind:6s}{meta['n_params']:>9}  "
              f"[{u.min():8.3f}, {u.max():8.3f}]  {rep.std(0).mean():>24.3e}")
        # gradient sanity
        s.requires_grad_(True)
        pol(states=s, t=0).sum().backward()
        gp = max(p.grad.abs().max().item() for p in pol.parameters()
                 if p.grad is not None)
        print(f"      d(u)/d(params) max = {gp:.3e}   "
              f"d(u)/d(state) max = {s.grad.abs().max().item():.3e}")
