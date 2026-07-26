import numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog

# ---- fast field (same as vector_field.py) ----
MW_GLC,MW_PENG=0.18016,0.33439; VGMAX,KG,KIUP=0.35,0.5,200.0
vg_BR=lambda S:VGMAX*S/(KG+S+S*S/KIUP); cG=500.0/MW_GLC
S=np.loadtxt("00_files/S.csv");LB=np.loadtxt("00_files/LB.csv");UB=np.loadtxt("00_files/UB.csv")
nM,nR=S.shape;Ssp=sp.csr_matrix(S);glu,xxx,pro=1,531,2
def FBA(vg_ub,c):
    b=list(zip(LB,UB));b[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR));row[0,pro]=1.0;row[0,xxx]=-c/MW_PENG
    A=sp.vstack([Ssp,row]).tocsr()
    r=linprog(np.eye(1,nR,xxx).ravel()*-1,A_eq=A,b_eq=np.zeros(nM+1),bounds=b,method="highs")
    return (r.x[xxx],r.x[glu]) if r.success else (0.0,0.0)

# ---- load denoiser rollout ----
roll = np.load("denoiser_rollout.npz")["roll"]   # (25,64,3)
nstep,npart=roll.shape[0]-1, roll.shape[1]

# ---- normalization: match rollout's actual range ----
GMAX=roll[:,:,0].max()*1.05; XMAX=roll[:,:,1].max()*1.10
V0=roll[:,:,2].min()*0.97;   V1=roll[:,:,2].max()*1.03
nG=lambda g:g/GMAX; nX=lambda x:x/XMAX; nV=lambda v:(v-V0)/(V1-V0)

# ---- field arrows: 5x5x3 grid over the rollout's range, feeding snapshot ----
F_SNAP,C_SNAP=0.05,0.23
gG=np.linspace(50,GMAX*0.92,5); gX=np.linspace(0.5,XMAX*0.92,5)
gV=np.array([V0+(V1-V0)*f for f in [0.1,0.5,0.9]])
Gg,Xg,Vg=np.meshgrid(gG,gX,gV,indexing="ij")
dG=np.zeros_like(Gg); dX=np.zeros_like(Gg)
for idx in np.ndindex(Gg.shape):
    G,X,V=Gg[idx],Xg[idx],Vg[idx]; V=max(V,1e-6)
    S_gL=max(G,0)/V*MW_GLC
    mu,vglu=FBA(vg_BR(S_gL)/MW_GLC,C_SNAP)
    dG[idx]=F_SNAP*cG+vglu*X; dX[idx]=mu*X
dV=np.full_like(Gg,F_SNAP)
u,v,w=dG/GMAX,dX/XMAX,dV/(V1-V0)
mag=np.sqrt(u**2+v**2+w**2); safe=np.maximum(mag,1e-12); L=0.07
u,v,w=u/safe*L,v/safe*L,w/safe*L; rel=mag/mag.max()

# ---- plot ----
fig=plt.figure(figsize=(13,7))
ax=fig.add_subplot(111,projection="3d"); cmap=plt.cm.viridis

# arrows
ax.quiver(nG(Gg),nX(Xg),nV(Vg),u,v,w,
          colors=cmap(rel.ravel()),lw=1.4,arrow_length_ratio=0.4,alpha=0.8)

# denoiser forward trajectories: colour = time step
t_norm=np.linspace(0,1,nstep+1)
tcmap=plt.cm.plasma
for j in range(npart):
    gn=nG(roll[:,j,0]); xn=nX(roll[:,j,1]); vn=nV(roll[:,j,2])
    for k in range(nstep):
        ax.plot(gn[k:k+2],xn[k:k+2],vn[k:k+2],
                color=tcmap(t_norm[k]),lw=1.0,alpha=0.55)

# initial cloud (blue) and final cloud (red)
ax.scatter(nG(roll[0,:,0]),nX(roll[0,:,1]),nV(roll[0,:,2]),
           c="royalblue",s=28,depthshade=False,zorder=10,label="initial Gaussian  $p(s,0)$")
ax.scatter(nG(roll[-1,:,0]),nX(roll[-1,:,1]),nV(roll[-1,:,2]),
           c="crimson",s=35,marker="*",depthshade=False,zorder=11,label="denoised cloud  $p(s,T)$")

# real-value tick labels
ax.set_xticks([nG(x) for x in [0,500,1000,1500]])
ax.set_xticklabels(["0","500","1000","1500"])
ax.set_yticks([nX(x) for x in [0,3,6,9]])
ax.set_yticklabels(["0","3","6","9"])
vlab=[V0+(V1-V0)*f for f in [0,0.33,0.67,1.0]]
ax.set_zticks([nV(v) for v in vlab])
ax.set_zticklabels([f"{v:.2f}" for v in vlab])
ax.set_xlabel("G  glucose [mmol]",labelpad=10)
ax.set_ylabel("X  biomass [g]",labelpad=10)
ax.set_zlabel("V  volume [L]",labelpad=4)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_zlim(0,1)
ax.view_init(elev=20,azim=-62)

# colourbar for trajectory time
sm=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
cb=fig.colorbar(sm,ax=ax,shrink=0.5,pad=0.02,location="right")
cb.set_label("denoiser step  (0 → T)",fontsize=9)

ax.legend(loc="upper left",fontsize=9)
ax.set_title("Forward denoiser trajectories projected onto the dcFBA vector field\n"
             "arrows = field flow direction  |  paths = $P_F$ forward diffusion  |  "
             "colour = time (plasma)",
             fontweight="bold",fontsize=11)
fig.tight_layout()
fig.savefig("field_with_denoiser.png",dpi=140,bbox_inches="tight")
print("wrote field_with_denoiser.png")
