"""
phase_portrait_SX.py -- 2D phase portrait in CONCENTRATION space (S g/L vs X g).
Drops the trivial V-axis (dV/dt=F carries no dynamics) and uses substrate
CONCENTRATION so the real curvature of the dynamics is visible.
Also reconstructs a feed-CAPPED trajectory (S<=40 g/L) as a preview of what the
glucose-constrained re-solve would give.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016; cG=500.0/MW_GLC
mu=lambda S:np.interp(S,Sg,mu_t); vglu=lambda S:np.interp(S,Sg,vg_t)

# field in (S,X) at fixed V, given feed F:  dS/dt = MW/V*(F*cG + vglu*X), dX/dt = mu*X
def field_SX(S,X,F,V):
    dS=MW_GLC/V*(F*cG + vglu(S)*X)
    dX=mu(S)*X
    return dS,dX

# ---- current optimal trajectory in concentration space ----
tr=np.load("traj_real.npz"); G,X,V,t=tr["G"],tr["X"],tr["V"],tr["t"]
S_opt=G/np.maximum(V,1e-6)*MW_GLC

# ---- feed-capped trajectory: throttle feed so S stays <= S_CAP (heuristic preview) ----
FEED=[0.05,0.0338,0,0,0,0,0,0,0,0.0274]; C_PX=[0.0002,0.2299,0.5212,0.7523,0.9400,1.1007,1.2879,1.4236,1.4892,1.5062]
T=150.0; nFE=10; S_CAP=40.0
def F_of_t(tt): return FEED[min(int(tt/T*nFE),nFE-1)]
def rhs(tt,y):
    Gg,Xg,Vg=y; Vg=max(Vg,1e-6); Xg=max(Xg,1e-9)
    S=max(Gg,0)/Vg*MW_GLC
    F=F_of_t(tt) if Vg<2.0 else 0.0
    # throttle feed to keep S<=S_CAP: if S near cap, only feed what uptake removes
    if S>=S_CAP:
        uptake=-vglu(S)*Xg           # mmol/h consumed (vglu negative)
        F=min(F, max(uptake/cG,0.0)) # feed only enough to replace consumption
    return [F*cG+vglu(S)*Xg, mu(S)*Xg, F]
X0,S0,V0=0.5,15.0,0.5; G0=S0/MW_GLC*V0
sol=solve_ivp(rhs,(0,T),[G0,X0,V0],t_eval=np.linspace(0,T,151),method="LSODA",rtol=1e-5,atol=1e-7,max_step=2.0)
Gc,Xc,Vc=sol.y; S_cap=Gc/np.maximum(Vc,1e-6)*MW_GLC
print(f"capped trajectory: S max {S_cap.max():.1f} g/L (cap {S_CAP}), X end {Xc[-1]:.1f} g")

# ============================ figure ============================
fig,(axL,axR)=plt.subplots(1,2,figsize=(15,6.2))

# ---- LEFT: full S range, current optimal trajectory + field streamplot (V=1, F=0.05) ----
Sg2=np.linspace(1,360,40); Xg2=np.linspace(1,150,40)
SS,XX=np.meshgrid(Sg2,Xg2)
dS,dX=field_SX(SS,XX,F=0.05,V=1.0)
spd=np.sqrt(dS**2+dX**2)
axL.streamplot(SS,XX,dS,dX,color=np.log10(spd+1),cmap="viridis",density=1.1,linewidth=0.8,arrowsize=0.9)
pts=axL.scatter(S_opt,X,c=t,cmap="autumn",s=14,zorder=5)
axL.plot(S_opt,X,color="crimson",lw=1,alpha=0.5,zorder=4)
axL.scatter([S_opt[0]],[X[0]],color="k",s=70,zorder=6,label="start")
axL.scatter([S_opt[-1]],[X[-1]],color="crimson",marker="*",s=160,zorder=6,label="end")
axL.axvspan(40,360,color="red",alpha=0.06)
axL.text(200,8,"osmotically\nlethal zone\n(S > 40 g/L)",color="darkred",fontsize=9,ha="center")
axL.set_xlabel("S  substrate concentration [g/L]");axL.set_ylabel("X  biomass [g]")
axL.set_title("CURRENT optimum (uncapped)\nglucose balloons to 358 g/L",fontweight="bold",fontsize=11)
axL.legend(loc="center right",fontsize=9);axL.set_xlim(0,360);axL.set_ylim(0,155)
cb=fig.colorbar(pts,ax=axL,shrink=0.8,pad=0.02);cb.set_label("time [h]")

# ---- RIGHT: zoomed low-S, capped trajectory + field ----
Sg3=np.linspace(0.2,45,40); Xg3=np.linspace(1,150,40)
SS3,XX3=np.meshgrid(Sg3,Xg3)
dS3,dX3=field_SX(SS3,XX3,F=0.02,V=1.0)
spd3=np.sqrt(dS3**2+dX3**2)
axR.streamplot(SS3,XX3,dS3,dX3,color=np.log10(spd3+1),cmap="viridis",density=1.1,linewidth=0.8,arrowsize=0.9)
pts3=axR.scatter(S_cap,Xc,c=sol.t,cmap="autumn",s=14,zorder=5)
axR.plot(S_cap,Xc,color="crimson",lw=1,alpha=0.5,zorder=4)
axR.scatter([S_cap[0]],[Xc[0]],color="k",s=70,zorder=6,label="start")
axR.scatter([S_cap[-1]],[Xc[-1]],color="crimson",marker="*",s=160,zorder=6,label="end")
axR.axvline(S_CAP,color="green",ls="--",lw=1.5,label=f"S cap = {S_CAP:.0f} g/L")
axR.set_xlabel("S  substrate concentration [g/L]");axR.set_ylabel("X  biomass [g]")
axR.set_title(f"FEED-CAPPED preview (S $\\leq$ {S_CAP:.0f} g/L)\nrealistic: S stays low, X reaches {Xc[-1]:.0f} g",fontweight="bold",fontsize=11)
axR.legend(loc="center right",fontsize=9);axR.set_xlim(0,45);axR.set_ylim(0,155)
cb3=fig.colorbar(pts3,ax=axR,shrink=0.8,pad=0.02);cb3.set_label("time [h]")

fig.suptitle("dcFBA dynamics in CONCENTRATION space  (S–X phase portrait, V-axis dropped)\n"
             "streamlines = flow field;  the trajectory is a clear curve, not a straight line",
             fontweight="bold",fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.93]);fig.savefig("phase_portrait_SX.png",dpi=140,bbox_inches="tight")
print("wrote phase_portrait_SX.png")
