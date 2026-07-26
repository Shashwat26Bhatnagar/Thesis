"""
vector_field.py -- G-X-V vector field of the dcFBA state dynamics, with the
OPTIMAL trajectory (from the solved Julia run) overlaid.

States (total amounts):  G [mmol glucose], X [g biomass], V [L volume].
Dynamics (same inner-FBA as the Julia / verify_ial1006.py):
    dG/dt = F*cG + vglu(S,X)*X      (feed in  -  uptake)
    dX/dt = mu(S,X)*X               (growth)
    dV/dt = F                       (feed; state-independent)
  where S_gL = G/V*MW_GLC, and mu, vglu come from  max-biomass FBA on iAL1006.

The system is NON-autonomous: F (feed) and c (P/X ratio) are functions of time,
so a static field is a SNAPSHOT at a fixed (F, c).  We draw two snapshots:
  - FEEDING  (F=0.05, c=early): the front-loaded growth phase
  - COASTING (F=0,    c=late):  the glucose-drawdown/production phase
Note dV/dt=F is the same at every grid point, so the V-arrows are uniform within
a snapshot -- the real structure is in the G-X plane.
"""
import numpy as np, scipy.sparse as sp
from scipy.optimize import linprog
from scipy.integrate import solve_ivp

# ---- REAL solved values from the Julia run ----------------------------------
FEED   = [0.05, 0.0338, 0,0,0,0,0,0,0, 0.0274]   # optimal feed [L/h], per FE
C_PX   = [0.0002, 0.2299, 0.5212, 0.7523, 0.9400, 1.1007, 1.2879, 1.4236, 1.4892, 1.5062]
T      = 150.0
nFE    = len(FEED)
MW_GLC, MW_PENG = 0.18016, 0.33439
VGMAX, KG, KIUP = 0.35, 0.5, 200.0
vg_BR = lambda S: VGMAX*S/(KG+S+S*S/KIUP)
cG    = 500.0/MW_GLC

# ---- core iAL1006 -----------------------------------------------------------
S  = np.loadtxt("00_files/S.csv"); LB = np.loadtxt("00_files/LB.csv"); UB = np.loadtxt("00_files/UB.csv")
nM,nR = S.shape; Ssp = sp.csr_matrix(S)
glu,xxx,pro = 2-1, 532-1, 3-1
print(f"core iAL1006: {nM} x {nR}")

def FBA(vg_ub, c):
    bounds=list(zip(LB,UB)); bounds[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR)); row[0,pro]=1.0; row[0,xxx]=-c/MW_PENG
    A=sp.vstack([Ssp,row]).tocsr(); b=np.zeros(nM+1)
    obj=np.zeros(nR); obj[xxx]=-1.0
    r=linprog(obj,A_eq=A,b_eq=b,bounds=bounds,method="highs")
    if not r.success: return 0.0,0.0
    return r.x[xxx], r.x[glu]                 # mu, vglu(negative)

def F_of_t(t): return FEED[min(int(t/T*nFE), nFE-1)]
def c_of_t(t):
    mids=[T/nFE*(i-0.5) for i in range(1,nFE+1)]
    return float(np.interp(t,mids,C_PX))

# ---- reconstruct the optimal trajectory -------------------------------------
def rhs(t,y):
    G,X,V=y; X=max(X,1e-9); V=max(V,1e-6)
    S_gL=max(G,0.0)/V*MW_GLC
    mu,vglu=FBA(vg_BR(S_gL)/MW_GLC, c_of_t(t))
    F = 0.0 if V>=2.0 else F_of_t(t)        # respect reactor V_max=2.0 cap
    return [F*cG+vglu*X, mu*X, F]

X0,S0,V0=0.5,15.0,0.5; G0=S0/MW_GLC*V0
print("integrating optimal trajectory...")
sol=solve_ivp(rhs,(0,T),[G0,X0,V0],method="LSODA",
              t_eval=np.linspace(0,T,151),rtol=1e-5,atol=1e-7,max_step=2.0)
Gtr,Xtr,Vtr=sol.y
print(f"  traj: G {Gtr[0]:.0f}->{Gtr[-1]:.0f} mmol | X {Xtr[0]:.1f}->{Xtr[-1]:.1f} g | V {Vtr[0]:.2f}->{Vtr[-1]:.2f} L")
np.savez("traj_real.npz",t=sol.t,G=Gtr,X=Xtr,V=Vtr)


# ============================ FIELD + 3D PLOT ============================
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
GMAX,XMAX,V0,V1=3600.,150.,0.5,2.05
nG=lambda g:g/GMAX; nX=lambda x:x/XMAX; nV=lambda v:(v-V0)/(V1-V0)
def field_cube(gG,gX,gV,F,c):
    Gg,Xg,Vg=np.meshgrid(gG,gX,gV,indexing="ij")
    dG=np.zeros_like(Gg);dX=np.zeros_like(Gg)
    for idx in np.ndindex(Gg.shape):
        G,X,V=Gg[idx],Xg[idx],Vg[idx]
        S_gL=max(G,0.)/max(V,1e-6)*MW_GLC
        mu,vglu=FBA(vg_BR(S_gL)/MW_GLC,c)
        dG[idx]=F*cG+vglu*X; dX[idx]=mu*X
    dV=np.full_like(Gg,F)
    u,v,w=dG/GMAX,dX/XMAX,dV/(V1-V0)
    mag=np.sqrt(u*u+v*v+w*w);safe=np.maximum(mag,1e-12);L=0.085
    return (nG(Gg),nX(Xg),nV(Vg),u/safe*L,v/safe*L,w/safe*L,mag/mag.max())

gG=np.array([100,700,1600,2700,3450.]);gX=np.linspace(10,140,5);gV=np.array([0.7,1.35,1.95])
fig=plt.figure(figsize=(15,7))
for k,(F,c,ttl) in enumerate([(0.05,0.23,"FEEDING  (F=0.05 L/h, c=0.23)  early growth"),
                              (0.0,1.40,"COASTING  (F=0, c=1.40)  production / drawdown")]):
    Xn,Yn,Zn,U,Vv,W,rel=field_cube(gG,gX,gV,F,c)
    ax=fig.add_subplot(1,2,k+1,projection="3d");cmap=plt.cm.viridis
    ax.quiver(Xn,Yn,Zn,U,Vv,W,colors=cmap(rel.ravel()),lw=1.6,arrow_length_ratio=0.45,normalize=False)
    ax.scatter(nG(Gtr),nX(Xtr),nV(Vtr),c=sol.t,cmap="autumn",s=12,zorder=10)
    ax.plot(nG(Gtr),nX(Xtr),nV(Vtr),color="crimson",lw=1.5,alpha=0.55,zorder=9)
    ax.scatter([nG(Gtr[0])],[nX(Xtr[0])],[nV(Vtr[0])],color="k",s=70,zorder=11,label="start t=0")
    ax.scatter([nG(Gtr[-1])],[nX(Xtr[-1])],[nV(Vtr[-1])],color="crimson",marker="*",s=150,zorder=11,label="end t=150 h")
    ax.set_xticks([nG(x) for x in [0,1000,2000,3000]]);ax.set_xticklabels(["0","1000","2000","3000"])
    ax.set_yticks([nX(x) for x in [0,50,100,150]]);ax.set_yticklabels(["0","50","100","150"])
    ax.set_zticks([nV(x) for x in [0.5,1.0,1.5,2.0]]);ax.set_zticklabels(["0.5","1.0","1.5","2.0"])
    ax.set_xlabel("G  glucose [mmol]",labelpad=10);ax.set_ylabel("X  biomass [g]",labelpad=10);ax.set_zlabel("V  vol [L]",labelpad=2)
    ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_zlim(0,1)
    ax.set_title(ttl,fontweight="bold",fontsize=10.5);ax.view_init(elev=20,azim=-62)
    if k==1:ax.legend(loc="upper left",fontsize=9)
sm=plt.cm.ScalarMappable(cmap="autumn",norm=plt.Normalize(0,150))
cb=fig.colorbar(sm,ax=fig.axes,shrink=0.5,pad=0.02,location="bottom");cb.set_label("trajectory time [h]")
fig.suptitle("dcFBA state-space flow  $(\\dot G,\\dot X,\\dot V)$  with optimal trajectory\n"
             "arrows: flow direction, colour = relative speed  |  $\\dot V=F$ (state-independent)",fontweight="bold",fontsize=12.5)
fig.savefig("vector_field_GXV.png",dpi=140,bbox_inches="tight");print("wrote vector_field_GXV.png")
