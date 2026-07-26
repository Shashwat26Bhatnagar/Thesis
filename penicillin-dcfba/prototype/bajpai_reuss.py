"""
Bajpai-Reuss (1980) rate laws -> time-varying c(t) = qp/mu, pre-computed into a
state-keyed dictionary for use as the FBA constraint  v_pen = c * v_biomass.

Penicillin is GROWTH-DECOUPLED:
  mu(S,X) = mu_x * S / (K_x*X + S)              # Contois growth (Eq 16)
  qp(S)   = mu_P * S / (K_p + S*(1 + S/K_I))    # Haldane substrate-inhibition (Eq 17)
  c(S,X)  = qp/mu = (mu_P/mu_x)*(K_x*X + S)/(K_p + S*(1 + S/K_I))

O2 is assumed non-limiting -> Kox = Kop = 0, which is exactly Bajpai-Reuss's own
M4 synthetic-medium case, so c depends only on (S, X).

Parameters: Pirt & Righelato M4 fit (Bajpai & Reuss 1980, Table p.338).
Rescale MU_P / yields to a modern strain or refit to IndPenSim as needed.
"""
from __future__ import annotations
import numpy as np

# --- Bajpai-Reuss M4 parameters ------------------------------------------------
MU_X = 0.092     # max specific growth rate           [1/h]
K_X  = 0.15      # Contois saturation constant        [g substrate / g DW]
MU_P = 0.005     # max specific production rate        [1/h]  (rescale to strain)
K_P  = 2.0e-4    # Monod constant, product formation   [g/dm^3]
K_I  = 0.1       # substrate-inhibition constant       [g/dm^3]
C_CAP = 1.0      # ceiling on penicillin-per-biomass [g/g]; caps the qp/mu ratio
                 # as mu -> 0 (encodes the physical production limit / the
                 # critical-growth-rate decay of qp).


def mu_of(S, X):
    """Specific growth rate, Contois (Eq 16), O2 non-limiting."""
    S = np.maximum(S, 0.0)
    return MU_X * S / (K_X * np.maximum(X, 1e-9) + S + 1e-12)


def qp_of(S):
    """Specific penicillin production rate, Haldane (Eq 17), O2 non-limiting."""
    S = np.maximum(S, 0.0)
    return MU_P * S / (K_P + S * (1.0 + S / K_I) + 1e-12)


def c_of(S, X):
    """c = qp/mu  (penicillin per unit biomass, g/g). Decoupled => rises as S falls.
    Capped at C_CAP to keep penicillin draw physical as mu -> 0."""
    mu = mu_of(S, X)
    c = np.where(mu > 1e-9, qp_of(S) / np.maximum(mu, 1e-9), C_CAP)
    return np.minimum(c, C_CAP)


# --- pre-computed state-keyed dictionary --------------------------------------
class CDict:
    """Grid lookup of c over (S, X). Keyed by state, NOT time, so it stays valid
    when the outer optimiser changes the feed profile."""

    def __init__(self, S_grid, X_grid, table):
        self.S_grid = S_grid
        self.X_grid = X_grid
        self.table = table          # shape (len(S_grid), len(X_grid))

    def __call__(self, S, X):
        i = int(np.clip(np.searchsorted(self.S_grid, S) - 1, 0, len(self.S_grid) - 1))
        j = int(np.clip(np.searchsorted(self.X_grid, X) - 1, 0, len(self.X_grid) - 1))
        return float(self.table[i, j])


def build_c_dict(S_max=40.0, X_max=50.0, nS=400, nX=200):
    """Sweep the (S, X) grid and tabulate c = qp/mu."""
    S_grid = np.linspace(1e-4, S_max, nS)
    X_grid = np.linspace(1e-3, X_max, nX)
    SS, XX = np.meshgrid(S_grid, X_grid, indexing="ij")
    table = c_of(SS, XX)
    return CDict(S_grid, X_grid, table)


if __name__ == "__main__":
    cd = build_c_dict()
    print("Bajpai-Reuss c = qp/mu  (penicillin per biomass)")
    print(f"{'S(g/L)':>8}{'X(g/L)':>8}{'mu(1/h)':>10}{'qp(1/h)':>10}{'c=qp/mu':>10}")
    for S in (20.0, 5.0, 1.0, 0.2, 0.05):
        X = 10.0
        print(f"{S:8.2f}{X:8.1f}{mu_of(S,X):10.4f}{qp_of(S):10.5f}{cd(S,X):10.4f}")
