"""
make_gif.py  --  rotating 3D GIF of the FL-GFN denoiser on the dcFBA vector field.
Spins 360 degrees so you can see the 3D structure from every angle.
"""
import numpy as np, scipy.sparse as sp, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import linprog

MW_GLC,MW_PENG=0.18016,0.33439
VGMAX,KG,KIUP=0.35,0.5,200.0; cG=500.0/MW_GLC; F_SNAP,C_SNAP=0.05,0.23
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
_vg=lambda S:np.interp(S,Sg,vg_t)
S=np.loadtxt("00_files/S.csv"); LB=np.loadtxt("00_files/LB.csv"); UB=np.loadtxt("00_files/UB.csv")
nM,nR=S.shape; Ssp=sp.csr_matrix(S); glu,xxx,pro=1,531,2
def FBA(vg_ub,c):
    b=list(zip(LB,UB)); b[glu]=(-vg_ub,0.0)
    row=sp.lil_matrix((1,nR)); row[0,pro]=1.0; row[0,xxx]=-c/MW_PENG
    A=sp.vstack([Ssp,row]).tocsr()
    r=linprog(np.eye(1,nR,xxx).ravel()*-1,A_eq=A,b_eq=np.zeros(nM+1),bounds=b,method="highs")
    return (r.x[xxx],r.x[glu]) if r.success else (0.0,0.0)

# ---- load denoiser rollout ----
roll=np.load("denoiser_rollout.npz")["roll"]   # (25,64,3)
nstep=roll.shape[0]-1; npart=roll.shape[1]
Gr,Xr,Vr=roll[:,:,0],roll[:,:,1],roll[:,:,2]

# ---- compute field arrows ----
gG=np.linspace(50,1700,5); gX=np.linspace(0.5,10,5); gV=np.array([0.6,0.9,1.2])
print("computing field arrows ...")
arrows=[]
for G in gG:
    for X in gX:
        for V in gV:
            S_gL=max(G,0.)/max(V,1e-6)*MW_GLC
            mu,vglu=FBA(_vg(S_gL)/MW_GLC, C_SNAP)
            dG=F_SNAP*cG+vglu*X; dX=mu*X; dV=F_SNAP
            mag=np.sqrt(dG**2+dX**2+dV**2)+1e-12
            sc=60.0
            arrows.append((G,X,V,dG/mag*sc,dX/mag*sc,dV/mag*sc,mag))
arrows=np.array(arrows)
mag_n=arrows[:,6]/arrows[:,6].max()
cmap_f=plt.cm.viridis; cmap_t=plt.cm.plasma
t_vals=np.linspace(0,1,nstep+1)

# ---- build figure ----
fig=plt.figure(figsize=(10,7),facecolor="k")
ax=fig.add_subplot(111,projection="3d",facecolor="k")

def draw(ax):
    # field arrows
    for i,row_ in enumerate(arrows):
        G,X,V,u,v,w,_=row_
        ax.quiver(G,X,V,u,v,w,color=cmap_f(mag_n[i]),lw=1.3,
                  arrow_length_ratio=0.35,alpha=0.75)
    # trajectories (every 2nd particle)
    for j in range(0,npart,2):
        for k in range(nstep):
            ax.plot([Gr[k,j],Gr[k+1,j]],[Xr[k,j],Xr[k+1,j]],[Vr[k,j],Vr[k+1,j]],
                    color=cmap_t(t_vals[k]),lw=0.9,alpha=0.7)
    # clouds
    ax.scatter(Gr[0,:],Xr[0,:],Vr[0,:],c="royalblue",s=30,depthshade=True,label="p(s,0)")
    ax.scatter(Gr[-1,:],Xr[-1,:],Vr[-1,:],c="crimson",marker="*",s=55,depthshade=True,label="p(s,T)")
    ax.set_xlabel("G  [mmol]",color="white",labelpad=8)
    ax.set_ylabel("X  [g]",color="white",labelpad=8)
    ax.set_zlabel("V  [L]",color="white",labelpad=4)
    ax.tick_params(colors="white"); ax.xaxis.pane.fill=False
    ax.yaxis.pane.fill=False;      ax.zaxis.pane.fill=False
    ax.xaxis.pane.set_edgecolor("0.3"); ax.yaxis.pane.set_edgecolor("0.3")
    ax.zaxis.pane.set_edgecolor("0.3")
    ax.grid(color="0.25",lw=0.4)
    leg=ax.legend(loc="upper left",fontsize=9,facecolor="0.15",labelcolor="white")

draw(ax)
ax.set_title("FL-GFN denoiser  ·  dcFBA field  ·  360° rotation",
             color="white",fontweight="bold",fontsize=12,pad=10)

# colorbars
sm1=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
sm2=plt.cm.ScalarMappable(cmap="viridis",norm=plt.Normalize(0,1))
cb1=fig.colorbar(sm1,ax=ax,shrink=0.38,pad=0.0,location="left")
cb2=fig.colorbar(sm2,ax=ax,shrink=0.38,pad=0.02,location="right")
cb1.set_label("trajectory step",color="white"); cb1.ax.yaxis.set_tick_params(color="white")
cb2.set_label("field speed",color="white");     cb2.ax.yaxis.set_tick_params(color="white")
plt.setp(cb1.ax.yaxis.get_ticklabels(),color="white")
plt.setp(cb2.ax.yaxis.get_ticklabels(),color="white")
fig.patch.set_facecolor("k")

# ---- animate: full 360° rotation ----
NFRAMES=72   # 5° per frame
def update(frame):
    ax.view_init(elev=22, azim=-50 + frame*(360/NFRAMES))
    return []

print(f"rendering {NFRAMES} frames ...")
ani=FuncAnimation(fig,update,frames=NFRAMES,interval=80,blit=False)
out="denoiser_field_360.gif"
ani.save(out,writer=PillowWriter(fps=14))
print(f"saved {out}")
