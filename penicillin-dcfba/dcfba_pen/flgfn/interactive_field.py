"""
interactive_field.py  --  3D interactive matplotlib plot
Rotate with mouse. Press 'q' to quit.
Shows:
  - dcFBA vector field arrows (viridis, coloured by speed)
  - FL-GFN denoiser forward trajectories (plasma, coloured by time step)
  - initial Gaussian cloud (blue) and final denoised cloud (red stars)
"""
import numpy as np, scipy.sparse as sp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import linprog

MW_GLC=0.18016; VGMAX,KG,KIUP=0.35,0.5,200.0; cG=500.0/0.18016; F_SNAP,C_SNAP=0.05,0.23
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
_vg=lambda S:np.interp(S,Sg,vg_t); _mu=lambda S:np.interp(S,Sg,mu_t)

S=np.loadtxt("00_files/S.csv"); LB=np.loadtxt("00_files/LB.csv"); UB=np.loadtxt("00_files/UB.csv")
nM,nR=S.shape; Ssp=sp.csr_matrix(S); glu,xxx,pro=1,531,2
def FBA(vg_ub,c):
    b=list(zip(LB,UB)); b[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR)); row[0,pro]=1.0; row[0,xxx]=-c/MW_PENG if False else row[0,xxx]
    row[0,xxx]=-c/0.33439
    A=sp.vstack([Ssp,row]).tocsr()
    r=linprog(np.eye(1,nR,xxx).ravel()*-1,A_eq=A,b_eq=np.zeros(nM+1),bounds=b,method="highs")
    return (r.x[xxx],r.x[glu]) if r.success else (0.0,0.0)

# ---- load denoiser rollout ----
roll=np.load("denoiser_rollout.npz")["roll"]   # (25,64,3)
nstep=roll.shape[0]-1; npart=roll.shape[1]
Gr,Xr,Vr=roll[:,:,0],roll[:,:,1],roll[:,:,2]

# ---- field grid in the denoiser region ----
gG=np.linspace(50,1700,5); gX=np.linspace(0.5,10,5); gV=np.array([0.6,0.9,1.2])
print("computing field arrows ..."); arrows=[]
for G in gG:
    for X in gX:
        for V in gV:
            S_gL=max(G,0.)/max(V,1e-6)*MW_GLC
            mu,vglu=FBA(_vg(S_gL)/MW_GLC, C_SNAP)
            dG=F_SNAP*cG+vglu*X; dX=mu*X; dV=F_SNAP
            mag=np.sqrt(dG**2+dX**2+dV**2)+1e-12
            scale=60.0
            arrows.append((G,X,V, dG/mag*scale, dX/mag*scale, dV/mag*scale, mag))
arrows=np.array(arrows)   # (N,7)
mag_n=arrows[:,6]/arrows[:,6].max()

# ---- interactive plot ----
fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111,projection="3d")

# arrows coloured by speed
cmap_f=plt.cm.viridis
for i,row_ in enumerate(arrows):
    G,X,V,u,v,w,_=row_
    ax.quiver(G,X,V,u,v,w,color=cmap_f(mag_n[i]),lw=1.5,arrow_length_ratio=0.35,alpha=0.8)

# trajectories coloured by time step
cmap_t=plt.cm.plasma
t_vals=np.linspace(0,1,nstep+1)
for j in range(0,npart,2):                   # every 2nd particle = 32 paths
    for k in range(nstep):
        ax.plot([Gr[k,j],Gr[k+1,j]],[Xr[k,j],Xr[k+1,j]],[Vr[k,j],Vr[k+1,j]],
                color=cmap_t(t_vals[k]),lw=1.0,alpha=0.65)

# initial + final clouds
ax.scatter(Gr[0,:],Xr[0,:],Vr[0,:],c="royalblue",s=35,depthshade=True,zorder=10,label="p(s,0) initial")
ax.scatter(Gr[-1,:],Xr[-1,:],Vr[-1,:],c="crimson",marker="*",s=60,depthshade=True,zorder=11,label="p(s,T) denoised")

ax.set_xlabel("G  glucose [mmol]",labelpad=12,fontsize=11)
ax.set_ylabel("X  biomass [g]",labelpad=12,fontsize=11)
ax.set_zlabel("V  volume [L]",labelpad=6,fontsize=11)
ax.set_title("FL-GFN denoiser on dcFBA field  —  drag to rotate",fontweight="bold",fontsize=12)
ax.view_init(elev=25,azim=-50)
ax.legend(loc="upper left",fontsize=10)

# colorbars
sm1=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
sm2=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,1))
fig.colorbar(sm1,ax=ax,shrink=0.4,pad=0.0,label="trajectory step",location="left")
fig.colorbar(sm2,ax=ax,shrink=0.4,pad=0.02,label="field speed (relative)",location="right")

plt.tight_layout()
plt.show()
print("window closed.")
