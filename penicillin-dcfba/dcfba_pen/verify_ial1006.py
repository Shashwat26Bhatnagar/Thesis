"""
verify_ial1006.py  --  run the explicit dFBA on the REAL condensed iAL1006 CSVs
(00_files/S.csv, LB.csv, UB.csv) with the Bajpai-Reuss c(t) coupling, to confirm
the genome-scale matrices + indices + unit conversions work end-to-end.

LP per step (HiGHS):  max v[bio]  s.t.  S v = 0,  LB<=v<=UB,
   v[glu] in [-vg_BR(S)/MW_GLC, 0]   (kinetic glucose uptake, mmol/gDW/h)
   v[pro] = c(S,X) * v[bio] / MW_PENG  (penicillin coupling, mmol/gDW/h)
States (total amounts): G[mmol], X[g], P[mmol], V[L];  titre = P/V * MW_PENG [g/L].
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

# --- Bajpai-Reuss (same as bajpai_reuss.py / the Julia) -----------------------
MU_X,K_X,MU_P,K_P,K_I = 0.092,0.15,0.005,2.0e-4,0.1
# time-varying experimental P/X ratio, one value per finite element (matches Julia C_PX)
C_PX = [0.0007, 0.0222, 0.2255, 0.6403, 1.0178, 1.2504, 1.3700, 1.4268, 1.4527, 1.4643]
VGMAX,KG,KIUP = 0.35,0.5,200.0
MW_GLC,MW_PENG = 0.18016,0.33439
mu_BR=lambda S,X: MU_X*S/(K_X*X+S+1e-9)
qp_BR=lambda S:   MU_P*S/(K_P+S*(1+S/K_I)+1e-12)
vg_BR=lambda S:   VGMAX*S/(KG+S+S*S/KIUP)
def c_of_t(t, T):                                # interpolate c(t) from the per-FE array
    nfe=len(C_PX); mids=[T/nfe*(i-0.5) for i in range(1,nfe+1)]
    return float(np.interp(t, mids, C_PX))

# --- load REAL condensed iAL1006 ----------------------------------------------
S  = np.loadtxt("00_files/S.csv")          # (nM, nR) = (1044, 1295)
LB = np.loadtxt("00_files/LB.csv")
UB = np.loadtxt("00_files/UB.csv")
nM,nR = S.shape
Ssp = sp.csr_matrix(S)
glu,xxx,pro,atp = 2-1, 532-1, 3-1, 553-1     # 0-based (index_map.json)
print(f"iAL1006 condensed: {nM} mets x {nR} rxns")

def FBA_pen(vg_ub_mmol, c):
    """max biomass s.t. Sv=0, bounds, glucose uptake<=kinetic, v_pro=c*v_bio/MW."""
    bounds = list(zip(LB, UB))
    bounds[glu] = (-vg_ub_mmol, 0.0)                 # uptake (negative)
    # equality: S v = 0  AND  v_pro - (c/MW)*v_bio = 0
    row = sp.lil_matrix((1, nR)); row[0, pro] = 1.0; row[0, xxx] = -c/MW_PENG
    A_eq = sp.vstack([Ssp, row]).tocsr()
    b_eq = np.zeros(nM + 1)
    obj = np.zeros(nR); obj[xxx] = -1.0              # maximize biomass
    r = linprog(obj, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not r.success: return 0.0, 0.0, 0.0
    return r.x[xxx], r.x[pro], r.x[glu]              # mu[1/h], pi[mmol/gDW/h], v_glu

def rhs(t, y, phi, T, nfe, cG):
    G,P,X,V = y; X=max(X,1e-9); V=max(V,1e-6)
    F = phi[min(int(t/T*nfe), nfe-1)]
    S_gL = max(G,0.0)/V * MW_GLC
    c = c_of_t(t, T)                                 # time-varying experimental ratio
    mu, pi, vglu = FBA_pen(vg_BR(S_gL)/MW_GLC, c)
    return [F*cG + vglu*X, pi*X, mu*X, F]            # dG,dP,dX,dV  (vglu negative)

def simulate(phi, T=150.0, nsteps=200, X0=0.5, S0_gL=15.0, V0=0.5):
    cG = 500.0/MW_GLC                                # feed glucose mmol/L
    G0 = S0_gL/MW_GLC*V0
    sol = solve_ivp(rhs, (0,T), [G0,0.0,X0,V0], args=(np.atleast_1d(phi),T,len(np.atleast_1d(phi)),cG),
                    method="LSODA", t_eval=np.linspace(0,T,nsteps+1), rtol=1e-5, atol=1e-7, max_step=2.0)
    G,P,X,V = sol.y
    return dict(t=sol.t, G=G, P=P, X=X, V=V, titer=P[-1]/V[-1]*MW_PENG, total_pen_g=P[-1]*MW_PENG)

if __name__ == "__main__":
    out = simulate(np.full(8, 0.012))
    print(f"final: X={out['X'][-1]:.1f} g  P={out['total_pen_g']:.2f} g  "
          f"V={out['V'][-1]:.2f} L  titre={out['titer']:.2f} g/L")
    S_gL = out['G']/out['V']*MW_GLC
    print(f"glucose S: {S_gL[0]:.1f} -> {S_gL[-1]:.3f} g/L   biomass {out['X'][1]:.2f} -> {out['X'][-1]:.1f} g")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig,ax=plt.subplots(1,3,figsize=(13,4))
        ax[0].plot(out['t'],out['X'],lw=2);                ax[0].set(title="Biomass X [g]",xlabel="t [h]")
        ax[1].plot(out['t'],out['P']*MW_PENG,lw=2,c="tab:red"); ax[1].set(title="Penicillin P [g]",xlabel="t [h]")
        ax[2].plot(out['t'],S_gL,lw=2,c="tab:green");      ax[2].set(title="Glucose S [g/L]",xlabel="t [h]")
        for a in ax: a.grid(alpha=.3)
        fig.suptitle(f"REAL iAL1006 dFBA + Bajpai-Reuss c(t)  |  titre {out['titer']:.1f} g/L",fontweight="bold")
        fig.tight_layout(); fig.savefig("verify_ial1006.png",dpi=140); print("wrote verify_ial1006.png")
    except ImportError: pass
