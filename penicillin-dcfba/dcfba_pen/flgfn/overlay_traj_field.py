"""
overlay_traj_field.py
Overlay the FL-GFN forward denoiser trajectories on the G-X-V vector field.
The denoiser particles (G:22->1683, X:0.4->9.8) live in a different region
than the optimal-trajectory field (G:0->3600, X:0->150), so we recompute a
field grid matched to the denoiser's region and overlay both together.
"""
import numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.optimize import linprog

MW_GLC,MW_PENG=0.18016,0.33439; VGMAX,KG,KIUP=0.35,0.5,200.0
vg_BR=lambda S:VGMAX*S/(KG+S+S*S/KIUP); cG=500.0/MW_GLC; F_SNAP,C_SNAP=0.05,0.23
S =np.loadtxt("00_files/S.csv"); LB=np.loadtxt("00_files/LB.csv"); UB=np.loadtxt("00_files/UB.csv")
nM,nR=S.shape; Ssp=sp.csr_matrix(S); glu,xxx,pro=1,531,2
def FBA(vg_ub,c):
    b=list(zip(LB,UB)); b[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR)); row[0,pro]=1.0; row[0,xxx]=-c/MW_PENG
    A=sp.vstack([Ssp,row]).tocsr()
    r=linprog(np.eye(1,nR,xxx).ravel()*-1,A_eq=A,b_eq=np.zeros(nM+1),bounds=b,method="highs")
    return (r.x[xxx],r.x[glu]) if r.success else (0.0,0.0)

# load denoiser rollout
roll=np.load("denoiser_rollout.npz")["roll"]   # (25,64,3)
nstep,npart=roll.shape[0]-1,roll.shape[1]
Gr,Xr,Vr=roll[:,:,0],roll[:,:,1],roll[:,:,2]

# build field grid matched to the denoiser region
gG=np.linspace(50, 1700, 5); gX=np.linspace(0.5, 10, 5); gV=np.array([0.6,0.9,1.2])
GMAX,XMAX,V0,V1=1800.,12.,0.5,1.3
nG2=lambda g:g/GMAX; nX2=lambda x:x/XMAX; nV2=lambda v:(v-V0)/(V1-V0)

def field_cube(gG,gX,gV,F,c):
    Gg,Xg,Vg=np.meshgrid(gG,gX,gV,indexing="ij")
    dG=np.zeros_like(Gg); dX=np.zeros_like(Gg)
    for idx in np.ndindex(Gg.shape):
        G,X,V=Gg[idx],Xg[idx],Vg[idx]
        S_gL=max(G,0.)/max(V,1e-6)*MW_GLC
        mu,vglu=FBA(vg_BR(S_gL)/MW_GLC,c)
        dG[idx]=F*cG+vglu*X; dX[idx]=mu*X
    dV=np.full_like(Gg,F)
    u,v,w=dG/GMAX,dX/XMAX,dV/(V1-V0)
    mag=np.sqrt(u*u+v*v+w*w); safe=np.maximum(mag,1e-12); L=0.08
    return nG2(Gg),nX2(Xg),nV2(Vg),u/safe*L,v/safe*L,w/safe*L,mag/mag.max()

print("computing field in denoiser region ..."); Xn,Yn,Zn,U,Vv,W,rel=field_cube(gG,gX,gV,F_SNAP,C_SNAP)

# time colormap for trajectories
t_norm=np.linspace(0,1,nstep+1); cmap_traj=plt.cm.plasma

fig=plt.figure(figsize=(13,7))
ax=fig.add_subplot(111,projection="3d")

# --- vector field arrows ---
ax.quiver(Xn,Yn,Zn,U,Vv,W,colors=plt.cm.viridis(rel.ravel()),
          lw=1.4,arrow_length_ratio=0.4,alpha=0.75)

# --- denoiser forward trajectories (subset for clarity) ---
show=range(0,npart,4)          # every 4th particle = 16 paths
for j in show:
    pts=np.column_stack([nG2(Gr[:,j]),nX2(Xr[:,j]),nV2(Vr[:,j])])
    for k in range(nstep):
        ax.plot(pts[k:k+2,0],pts[k:k+2,1],pts[k:k+2,2],
                color=cmap_traj(t_norm[k]),lw=1.2,alpha=0.7)

# --- initial cloud (t=0) ---
ax.scatter(nG2(Gr[0,:]),nX2(Xr[0,:]),nV2(Vr[0,:]),
           c="royalblue",s=28,depthshade=False,zorder=10,label="initial cloud  $p(s,0)$")
# --- final cloud (t=T) ---
ax.scatter(nG2(Gr[-1,:]),nX2(Xr[-1,:]),nV2(Vr[-1,:]),
           c="crimson",s=28,marker="*",depthshade=False,zorder=11,label="denoised cloud  $p(s,T)$")

# axes ticks in real units
ax.set_xticks([nG2(x) for x in [0,500,1000,1500]]); ax.set_xticklabels(["0","500","1000","1500"])
ax.set_yticks([nX2(x) for x in [0,4,8,12]]);         ax.set_yticklabels(["0","4","8","12"])
ax.set_zticks([nV2(x) for x in [0.5,0.75,1.0,1.25]]);ax.set_zticklabels(["0.5","0.75","1.0","1.25"])
ax.set_xlabel("G  glucose [mmol]",labelpad=10)
ax.set_ylabel("X  biomass [g]",labelpad=10)
ax.set_zlabel("V  volume [L]",labelpad=2)
ax.set_xlim(0,1); ax.set_ylim(0,1); ax.set_zlim(0,1)
ax.view_init(elev=22,azim=-58)

# colorbar for trajectory time
sm=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
cb=fig.colorbar(sm,ax=ax,shrink=0.5,pad=0.02,location="right"); cb.set_label("trajectory step")
ax.legend(loc="upper left",fontsize=9)

# legend patch for arrows
from matplotlib.patches import Patch
ax.add_artist(ax.legend(handles=[
    plt.Line2D([],[],color="royalblue",marker="o",lw=0,label="initial cloud  p(s,0)"),
    plt.Line2D([],[],color="crimson",  marker="*",lw=0,label="denoised cloud  p(s,T)"),
    Patch(fc=plt.cm.viridis(0.7),label="vector field (arrows)"),
],loc="upper left",fontsize=9))

fig.suptitle("FL-GFN forward denoiser trajectories projected onto the dcFBA vector field\n"
             "paths coloured by time (dark→bright);  arrows = $(\\dot G,\\dot X,\\dot V)$ at $F=0.05$",
             fontweight="bold",fontsize=12)
fig.savefig("denoiser_on_field.png",dpi=140,bbox_inches="tight")
print("wrote denoiser_on_field.png")
