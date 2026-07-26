"""
dFBA by the direct (static-optimisation) approach for penicillin fed-batch.

Per step: kinetics -> FBA_pen (adds v_pen = c*v_biomass) -> rates -> integrate the
total-amount ODEs (eqs 1a-1d) over dt -> repeat.  Mirrors the E. coli Julia script
( mi,vac,vg,vo = FBA(vg,vo) ), with FBA now returning the penicillin rate pi.
"""
from __future__ import annotations
import numpy as np
import cobra
from core_model import build_core_model, RXN
from bajpai_reuss import build_c_dict

# --- glucose uptake kinetics (Monod + mild substrate inhibition) --------------
VG_MAX = 0.35   # g glc / gDW / h
KG     = 0.5    # g/L
KI_UP  = 200.0  # g/L


def vg_uptake(S):
    S = max(S, 0.0)
    return VG_MAX * S / (KG + S + S * S / KI_UP)


# --- FBA layer: returns (mu, pi, gamma) ---------------------------------------
def FBA_pen(model, vg_ub, c):
    """Constrain glucose uptake, add v_pen = c*v_biomass, maximise biomass.
    Returns mu = v_bio [1/h], pi = v_pen [g/gDW/h], gamma = v_glc [g/gDW/h]."""
    with model:
        model.reactions.get_by_id(RXN["glc"]).upper_bound = max(vg_ub, 0.0)
        vbio = model.reactions.get_by_id(RXN["bio"]).flux_expression
        vpen = model.reactions.get_by_id(RXN["pen"]).flux_expression
        cons = model.problem.Constraint(vpen - c * vbio, lb=0, ub=0, name="c_couple")
        model.add_cons_vars([cons])
        model.objective = RXN["bio"]
        s = model.optimize()
        if s.status != "optimal":
            return 0.0, 0.0, 0.0
        return (s.fluxes[RXN["bio"]], s.fluxes[RXN["pen"]], s.fluxes[RXN["glc"]])


# --- forward simulation (direct-approach dFBA) --------------------------------
def simulate(phi, T=150.0, nsteps=600, X0=0.5, G0=10.0, V0=1.0,
             gphi=150.0, model=None, cdict=None, record_rates=True, **_):
    """
    phi : array of feed rates [L/h], one per control interval (piecewise constant).
    States are TOTAL amounts: X [gDW], P [g], G [g], V [L]; titre = P/V.
    Integrated with a stiff adaptive solver (LSODA), FBA evaluated in the RHS.
    """
    from scipy.integrate import solve_ivp
    model = model or build_core_model()
    cdict = cdict or build_c_dict()
    phi = np.atleast_1d(phi).astype(float)
    nfe = len(phi)

    def rhs(t, y):
        X, P, G, V = y
        X = max(X, 1e-9); V = max(V, 1e-6)
        F = phi[min(int(t / T * nfe), nfe - 1)]
        S = max(G, 0.0) / V
        Xc = X / V
        mu, pi, gamma = FBA_pen(model, vg_uptake(S), cdict(S, Xc))
        return [mu * X, pi * X, F * gphi - gamma * X, F]

    teval = np.linspace(0, T, nsteps + 1)
    sol = solve_ivp(rhs, (0, T), [X0, 0.0, G0, V0], method="LSODA",
                    t_eval=teval, rtol=1e-5, atol=1e-7, max_step=2.0)
    X, P, G, V = sol.y
    # recover rates along the trajectory for plotting
    MU, PI, GA, CC = (np.zeros_like(sol.t) for _ in range(4))
    if record_rates:
        for k in range(len(sol.t)):
            S = max(G[k], 0.0) / max(V[k], 1e-6); Xc = max(X[k], 1e-9) / max(V[k], 1e-6)
            c = cdict(S, Xc); mu, pi, ga = FBA_pen(model, vg_uptake(S), c)
            MU[k], PI[k], GA[k], CC[k] = mu, pi, ga, c
    titer = P[-1] / V[-1]
    return dict(t=sol.t, X=X, P=P, G=np.maximum(G, 0.0), V=V,
                mu=MU, pi=PI, gamma=GA, c=CC, titer=titer, total_pen=P[-1])


if __name__ == "__main__":
    # constant modest feed as a sanity check
    import numpy as np
    out = simulate(phi=np.full(10, 0.02))
    print(f"final: X={out['X'][-1]:.2f} gDW  P={out['P'][-1]:.3f} g  "
          f"V={out['V'][-1]:.2f} L  titre={out['titer']:.4f} g/L")
    print(f"c rose {out['c'][1]:.4f} -> {out['c'][-1]:.4f}  "
          f"(mu {out['mu'][1]:.3f} -> {out['mu'][-1]:.3f} 1/h)")
