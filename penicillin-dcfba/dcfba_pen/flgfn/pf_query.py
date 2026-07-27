"""
pf_query.py
===========
Query the FL-GFN forward policy as a DISTRIBUTION OVER THE NEXT STATE, given a TIME t.

    P_F( s' | s(t) )  =  Normal( mean = b, cov = Sigma )   # FULL 3x3 Sigma
    with   a = s(t)                (state at time t)
           b = a + drift(a)        (predicted next state == CENTRE of the Gaussian)

So the returned Gaussian is centred at b, exactly as requested.

------------------------------------------------------------------------------
TIME CONVENTIONS (verified against the codebase, not memory)
------------------------------------------------------------------------------
    DT      = 1.0 h    <- INTEGRATION STEP SIZE (time advanced per FL-GFN step)
    NSTEP   = 150      <- number of steps
    horizon = DT*NSTEP = 150.0 h      <- FL-GFN training horizon (FULL fermentation)

    DT is NOT the horizon and NOT the cultivation length.
    dcFBA fed-batch run   T = 150 h   (capped_traj.npz spans 0..150 h)
    PenSim/IndPenSim      ~230 h      (a DIFFERENT simulator; not used here)

Two sources are available for a = s(t):
    source="traj"  -> interpolate the realistic fed-batch dFBA trajectory
                      (capped_traj.npz), valid t in [0, 150] h.
    source="flgfn" -> roll the FL-GFN forward k = round(t/DT) steps from the
                      initial Gaussian; valid t in [0, 150] h (training horizon).

------------------------------------------------------------------------------
USAGE
------------------------------------------------------------------------------
    from pf_query import PFQuery
    q = PFQuery()                       # loads (or trains once and caches) the net

    d = q.next_state_distribution(t=3.0)
    d["a"]        # state at t          (S g/L, X g, V L)
    d["b"]        # CENTRE of Gaussian  (S, X, V)  == predicted next state
    d["cov"]      # FULL 3x3 covariance, physical units
    d["corr"]     # 3x3 correlation matrix
    d["sample"](n)      # draw n next-states
    d["pdf"]([S, X])    # density at a query point (per (g/L)^2)

    q.next_state_distribution(t=3.0, source="traj")   # use the 150 h fed-batch run
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

# ----------------------------------------------------------------------------
# constants -- MUST match flgfn_corrected.py exactly
# ----------------------------------------------------------------------------
MW_GLC = 0.18016
KD     = 0.02          # biomass death / autolysis rate [1/h]
VMAX   = 2.0           # vessel limit [L]; V is now a STATE (3-D)
cG     = 500.0 / MW_GLC                 # feed glucose conc [mmol/L]
FMAX, KP = 0.05, 0.6                    # max feed [L/h], controller gain [1/h]
S_HI, S_LO, X_SW, W = 10.0, 0.15, 25.0, 8.0   # growth->production setpoint switch

DT     = 1.0           # <-- INTEGRATION STEP [h]  (NOT the horizon)
NSTEP  = 150           # number of steps
HORIZON = DT * NSTEP   # = 150.0 h -> now the FULL fermentation

MU0  = np.array([15.0, 0.5, 0.5])       # inoculum (S0=15 g/L, X0=0.5 g, V0=0.5 L)
SIG0 = np.diag([2.0, 0.15, 0.03]) ** 2  # initial cloud covariance

SC  = np.array([20.0, 80.0, 2.5])       # normalisation scale (S, X, V)
OFF = np.array([0.0, 0.0, 0.0])         # normalisation offset

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT  = os.path.join(_HERE, "flgfn_corrected.pt")
TABLES = os.path.join(_HERE, "field_tables.npz")
TRAJ   = os.path.join(_HERE, os.pardir, "capped_traj.npz")   # canonical copy written by capped_dfba.py


def _norm(s):
    return (np.asarray(s, dtype=float) - OFF) / SC


def _denorm(n):
    return np.asarray(n, dtype=float) * SC + OFF


# ----------------------------------------------------------------------------
# corrected dcFBA FED-BATCH field (state-feedback feed + death) -- same as training
# ----------------------------------------------------------------------------
_d   = np.load(TABLES)
_Sg, _mu_t, _vg_t = _d["Sgrid"], _d["mu_tab"], _d["vg_tab"]


def _mu(S):
    return np.interp(S, _Sg, _mu_t)


def _vg(S):
    return np.interp(S, _Sg, _vg_t)


def _dvg(S):
    return (np.interp(S + 1e-3, _Sg, _vg_t) - np.interp(S - 1e-3, _Sg, _vg_t)) / 2e-3


def _sset(X):
    """setpoint: high S while biomass small (growth), low S later (production)."""
    return S_LO + (S_HI - S_LO) / (1.0 + np.exp((X - X_SW) / W))


def _feed(S, X, V):
    """consumption-matching P controller, gated off when the vessel is full."""
    raw = (-_vg(S) * X + KP * (_sset(X) - S) * V / MW_GLC) / cG
    return float(np.clip(raw, 0.0, FMAX) / (1.0 + np.exp((V - VMAX) / 0.02)))


def field(s):
    """(dS/dt, dX/dt, dV/dt) for the 3-D fed-batch field (state-feedback feed)."""
    S, X, V = s
    S = max(S, 0.0); V = max(V, 1e-3)
    F = _feed(S, X, V)
    return np.array([(F * cG + _vg(S) * X) * MW_GLC / V - S * F / V,
                     (_mu(S) - KD) * X,
                     F])


def divergence(s, h=1e-4):
    """numerical: the feed law breaks the old analytic form."""
    out = 0.0
    for i in range(3):
        sp = list(s); sm = list(s); sp[i] += h; sm[i] -= h
        out += (field(sp)[i] - field(sm)[i]) / (2 * h)
    return float(out)


# ----------------------------------------------------------------------------
# FL-GFN network -- architecture identical to flgfn_corrected.py
# ----------------------------------------------------------------------------
def _mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(),
                         nn.Linear(h, h), nn.SiLU(),
                         nn.Linear(h, o))


D = 3
NTRI = D * (D + 1) // 2                 # 3 diagonal + 3 off-diagonal = 6
_DI = torch.arange(D)
_TI = torch.tril_indices(D, D, offset=-1)


def _tril(raw):
    """raw (...,6) -> lower-triangular L (...,3,3), positive diagonal; Sigma = L L^T."""
    L = torch.zeros(*raw.shape[:-1], D, D, dtype=raw.dtype, device=raw.device)
    L[..., _DI, _DI] = nn.functional.softplus(raw[..., :D] - 3.0) + 1e-3
    L[..., _TI[0], _TI[1]] = raw[..., D:]
    return L


class FLGFN(nn.Module):
    def __init__(s):
        super().__init__()
        s.fdrift = _mlp(D, D)
        s.bdrift = _mlp(D, D)
        s.logF   = _mlp(D, 1)
        s.fcov   = _mlp(D, NTRI)        # state-dependent FULL covariance factor
        s.bcov   = _mlp(D, NTRI)

    def distF(s, a):
        return MultivariateNormal(a + s.fdrift(a), scale_tril=_tril(s.fcov(a)))

    def distB(s, b):
        return MultivariateNormal(b + s.bdrift(b), scale_tril=_tril(s.bcov(b)))

    def logpF(s, a, b):
        return s.distF(a).log_prob(b)

    def logpB(s, a, b):
        return s.distB(b).log_prob(a)


def _make_traj(m, rng):
    S0 = rng.multivariate_normal(MU0, SIG0, size=m)
    traj = np.zeros((m, NSTEP + 1, D))
    en   = np.zeros((m, NSTEP))
    traj[:, 0] = S0
    for k in range(NSTEP):
        for i in range(m):
            s = traj[i, k]
            en[i, k] = divergence(s) * DT
            traj[i, k + 1] = np.clip(s + field(s) * DT,
                                     [0.0, 1e-6, 1e-3], [np.inf, np.inf, VMAX])
    return traj, en


def _train(iters=4500, verbose=True):
    """Train FL-GFN with the FL-DB loss (same recipe as flgfn_corrected.py)."""
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    TR, EN = _make_traj(256, rng)
    net = FLGFN()
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    NS = torch.tensor(_norm(TR), dtype=torch.float32)
    E  = torch.tensor(EN, dtype=torch.float32)
    m  = NS.shape[0]
    for it in range(iters):
        bi = torch.randint(0, m, (128,))
        ki = torch.randint(0, NSTEP, (128,))
        s, s2, e = NS[bi, ki], NS[bi, ki + 1], E[bi, ki]
        res = (net.logF(s).squeeze(-1) + net.logpF(s, s2)
               - net.logF(s2).squeeze(-1) - net.logpB(s, s2) + e)
        fm = (((net.fdrift(s) - (s2 - s)) ** 2).sum(-1).mean()
              + ((net.bdrift(s2) - (s - s2)) ** 2).sum(-1).mean())
        tan = (net.logF(NS[bi, NSTEP]).squeeze(-1) ** 2).mean()
        loss = (res ** 2).mean() + 25 * fm + tan
        opt.zero_grad(); loss.backward(); opt.step()
        if verbose and it % 1500 == 0:
            print(f"  [train] it {it:5d}  FL-DB={(res**2).mean().item():.5f}  fm={fm.item():.6f}")
    return net


# ----------------------------------------------------------------------------
# main API
# ----------------------------------------------------------------------------
class PFQuery:
    """Query P_F(s'|s(t)) as a Gaussian centred at b = a + drift(a)."""

    def __init__(self, ckpt=CKPT, train_if_missing=True, verbose=True):
        self.net = FLGFN()
        if os.path.exists(ckpt):
            self.net.load_state_dict(torch.load(ckpt, map_location="cpu"))
            if verbose:
                print(f"[PFQuery] loaded checkpoint {os.path.basename(ckpt)}")
        elif train_if_missing:
            if verbose:
                print("[PFQuery] no checkpoint -> training FL-GFN once ...")
            self.net = _train(verbose=verbose)
            torch.save(self.net.state_dict(), ckpt)
            if verbose:
                print(f"[PFQuery] saved checkpoint {os.path.basename(ckpt)}")
        else:
            raise FileNotFoundError(ckpt)
        self.net.eval()

        # realistic 150 h fed-batch trajectory (optional source for a = s(t))
        self._traj = np.load(TRAJ) if os.path.exists(TRAJ) else None

    # ---------------- state at a given time -------------------------------
    def state_at(self, t, source="flgfn", n_cloud=400, seed=0):
        """Return a = s(t) as (S g/L, X g, V L).

        source="flgfn": roll FL-GFN forward k = round(t/DT) steps  (t in [0, 150] h)
        source="traj" : interpolate the 150 h fed-batch dFBA run   (t in [0, 150] h)
        """
        if source == "traj":
            if self._traj is None:
                raise FileNotFoundError("capped_traj.npz not found")
            tt = self._traj["t"]
            if not (tt[0] - 1e-9 <= t <= tt[-1] + 1e-9):
                raise ValueError(f"t={t} outside fed-batch run [0, {tt[-1]:.0f}] h")
            return np.array([np.interp(t, tt, self._traj["S"]),
                             np.interp(t, tt, self._traj["X"]),
                             np.interp(t, tt, self._traj["V"])])

        if source == "flgfn":
            k = int(round(t / DT))
            if k < 0 or k > NSTEP:
                raise ValueError(
                    f"t={t} h -> step {k}; FL-GFN horizon is "
                    f"{HORIZON} h (DT={DT} h x NSTEP={NSTEP}). "
                    f"Use source='traj' for the 150 h fed-batch run.")
            rng = np.random.default_rng(seed)
            ns = torch.tensor(_norm(rng.multivariate_normal(MU0, SIG0, n_cloud)),
                              dtype=torch.float32)
            with torch.no_grad():
                for _ in range(k):
                    ns = ns + self.net.fdrift(ns)          # deterministic drift step
            return _denorm(np.array(ns.tolist())).mean(axis=0)

        raise ValueError("source must be 'flgfn' or 'traj'")

    # ---------------- the requested function ------------------------------
    def next_state_distribution(self, t, source="flgfn", state=None):
        """Distribution over the NEXT state, centred at b.

        Parameters
        ----------
        t      : time [h]. Converted to step index k = round(t/DT), DT = 1.0 h.
        source : "flgfn" (roll the sampler, t in [0, 150] h) or
                 "traj"  (interpolate the 150 h fed-batch run).
        state  : optionally supply a = (S, X, V) directly and skip the time lookup.

        Returns
        -------
        dict with:
          t, k, dt, t_next : timing info
          a        : current state (S, X, V)  [g/L, g, L]
          b        : CENTRE of the Gaussian = a + drift(a) = predicted next state
          sigma    : marginal std per axis = sqrt(diag(cov))
          cov      : FULL 3x3 covariance (physical units); corr : 3x3 correlation
          drift    : b - a
          sample(n): draw n next-states  -> (n, 3), uses the FULL covariance
          pdf(pt)  : density of P_F at pt, per (g/L)^2  (physical units)
          logpdf(pt)
          field    : true dcFBA field velocity at a  (for comparison)
          b_true   : a + field(a)*DT  (the deterministic ground-truth next state)
        """
        a = np.asarray(state, dtype=float) if state is not None \
            else self.state_at(t, source=source)

        na = torch.tensor(_norm(a)[None, :], dtype=torch.float32)
        with torch.no_grad():
            nb = na + self.net.fdrift(na)                  # normalised centre
            L  = _tril(self.net.fcov(na))[0]               # Cholesky factor (3x3)
            cov_n = np.array((L @ L.T).tolist())           # FULL normalised covariance (no ABI bridge)
        b = _denorm(np.array(nb.tolist())[0])

        cov   = cov_n * np.outer(SC, SC)                    # -> physical units (3x3, full)
        sigma = np.sqrt(np.diag(cov))                       # marginal std per axis
        sd    = np.sqrt(np.diag(cov))
        corr  = cov / np.outer(sd, sd)                      # correlation matrix

        # density computed ANALYTICALLY from (b, cov) -- P_F is exactly N(b, cov).
        # Pure numpy: no torch call, no numpy-ABI bridge, no closure indirection.
        _Lc     = np.linalg.cholesky(cov)                   # physical-space Cholesky
        _logdet = 2.0 * float(np.sum(np.log(np.diag(_Lc))))
        _DIM    = cov.shape[0]

        def logpdf(pt):
            x = np.atleast_2d(np.asarray(pt, dtype=float)) - b      # (n, D)
            z = np.linalg.solve(_Lc, x.T)                           # (D, n)
            return -0.5 * (z ** 2).sum(0) - 0.5 * (_logdet + _DIM * np.log(2 * np.pi))

        def pdf(pt):
            return np.exp(logpdf(pt))

        def sample(n=1, seed=None):
            rng = np.random.default_rng(seed)
            return rng.multivariate_normal(b, cov, size=n)   # uses FULL covariance

        f = field(a)
        return {
            "t": float(t), "k": int(round(t / DT)), "dt": DT,
            "t_next": float(t) + DT,
            "a": a, "b": b, "drift": b - a,
            "sigma": sigma, "cov": cov, "corr": corr,
	    "cov_n": cov_n,
            "sample": sample, "pdf": pdf, "logpdf": logpdf,
            "field": f, "b_true": a + f * DT,
            "source": "given" if state is not None else source,
        }


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"DT = {DT} h  (integration step)   NSTEP = {NSTEP}   "
          f"horizon = {HORIZON} h")
    print("dcFBA fed-batch run = 150 h ; PenSim ~230 h (different simulator)\n")

    q = PFQuery()

    for t in (0.0, 50.0, 100.0, 150.0):
        d = q.next_state_distribution(t)
        print(f"t={d['t']:4.1f} h (step {d['k']:2d}) -> t'={d['t_next']:4.1f} h")
        print(f"   a  (state at t)      S={d['a'][0]:7.3f} g/L   X={d['a'][1]:7.3f} g")
        print(f"   b  (Gaussian centre) S={d['b'][0]:7.3f} g/L   X={d['b'][1]:7.3f} g")
        print(f"   b_true (field*DT)    S={d['b_true'][0]:7.3f} g/L   X={d['b_true'][1]:7.3f} g")
        print(f"   sigma                S={d['sigma'][0]:7.4f}       X={d['sigma'][1]:7.4f}")
        print(f"   pdf at b             {d['pdf'](d['b'])[0]:.4g}  per (g/L)^2")
        print(f"   3 samples            {np.round(d['sample'](3, seed=1), 3).tolist()}")
        print()

    d = q.next_state_distribution(t=75.0, source="traj")
    print(f"[source='traj'] t=75 h  a={np.round(d['a'],3)}  b={np.round(d['b'],3)}")
