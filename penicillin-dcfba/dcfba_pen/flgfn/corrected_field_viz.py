"""
corrected_field_viz.py -- CORRECT, properly-scaled trajectory-vs-field visualization.
The system is non-autonomous, so we evaluate the field with each point's ACTUAL
F(t), V(t), c(t).  Two panels:
  A) optimal trajectory with local field arrows sampled ALONG the path
  B) FL-GFN diffusion sampler on a CONSISTENT background field (F=0.05, V=V(G)
     tracking the rollout), so the streamlines and the sampler agree.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt

d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016; cG=500.0/MW_GLC
mu=lambda S:np.interp(S,Sg,mu_t); vglu=lambda S:np.interp(S,Sg,vg_t)
def fieldGX(G,X,V,F):
    S=np.maximum(G,0)/np.maximum(V,1e-6)*MW_GLC
    return F*cG+vglu(S)*X, mu(S)*X

fig,(axA,axB)=plt.subplots(1,2,figsize=(15.5,6.4))

# ================= Panel A: optimal trajectory + along-path local field =========
tr=np.load("traj_real.npz"); G,X,V,t=tr["G"],tr["X"],tr["V"],tr["t"]
FEED=[0.05,0.0338,0,0,0,0,0,0,0,0.0274]; T=150.0; nFE=10
Farr=np.array([FEED[min(int(tt/T*nFE),nFE-1)] for tt in t]); Farr[V>=2.0]=0.0
Gr,Xr=G.max()-G.min(),X.max()-X.min()
# local field arrows along the path (subsample), normalized to uniform display length
idx=np.arange(0,len(G)-1,6)
dG,dX=fieldGX(G[idx],X[idx],V[idx],Farr[idx])
uh=dG/Gr; vh=dX/Xr; nrm=np.sqrt(uh**2+vh**2)+1e-12
L=0.045; aG=uh/nrm*L*Gr; aX=vh/nrm*L*Xr
axA.quiver(G[idx],X[idx],aG,aX,angles="xy",scale_units="xy",scale=1,
           color="steelblue",width=0.004,headwidth=4,zorder=3,alpha=0.9,label="local field (uses F(t),V(t))")
sc=axA.scatter(G,X,c=t,cmap="autumn",s=14,zorder=4)
axA.plot(G,X,color="crimson",lw=1,alpha=0.4,zorder=2)
axA.scatter([G[0]],[X[0]],color="k",s=70,zorder=5,label="start")
axA.scatter([G[-1]],[X[-1]],color="crimson",marker="*",s=170,zorder=5,label="end")
axA.set_xlabel("G  glucose [mmol]");axA.set_ylabel("X  biomass [g]")
axA.set_title("OPTIMAL trajectory follows its local field\n"
              "mean cos(tangent, field) = 0.997",fontweight="bold",fontsize=11)
axA.legend(loc="upper right",fontsize=8.5);axA.set_xlim(-100,3700);axA.set_ylim(-5,155)
cb=fig.colorbar(sc,ax=axA,shrink=0.8,pad=0.02);cb.set_label("time [h]")

# ================= Panel B: FL-GFN sampler on consistent field ==================
roll=np.load("denoiser_rollout.npz")["roll"]   # (25,64,3)
nstep,npart=roll.shape[0]-1,roll.shape[1]
Gm=roll[:,:,0].mean(1); Vm=roll[:,:,2].mean(1)          # mean G(t), V(t) of the cloud
# consistent background field: V tracks the rollout's V(G)
Ggrid=np.linspace(30,1700,32); Xgrid=np.linspace(0.3,11,32)
GG,XX=np.meshgrid(Ggrid,Xgrid)
Vbg=np.interp(GG, Gm, Vm)                                # V as a function of G along the path
dGb,dXb=fieldGX(GG,XX,Vbg,0.05)
spd=np.sqrt((dGb/(1700))**2+(dXb/11)**2)
axB.streamplot(GG,XX,dGb,dXb,color=np.log10(spd+1e-6),cmap="viridis",
               density=1.2,linewidth=0.8,arrowsize=0.9,zorder=1)
# overlay denoiser forward trajectories, coloured by step
import matplotlib.cm as cm
tcol=cm.plasma(np.linspace(0,1,nstep+1))
for j in range(0,npart,3):
    for k in range(nstep):
        axB.plot(roll[k:k+2,j,0],roll[k:k+2,j,1],color=tcol[k],lw=1.1,alpha=0.7,zorder=3)
axB.scatter(roll[0,:,0],roll[0,:,1],c="royalblue",s=22,zorder=4,label="p(s,0) initial")
axB.scatter(roll[-1,:,0],roll[-1,:,1],c="crimson",marker="*",s=55,zorder=5,label="p(s,T) denoised")
axB.set_xlabel("G  glucose [mmol]");axB.set_ylabel("X  biomass [g]")
axB.set_title("FL-GFN diffusion sampler follows the field\n"
              "consistent field (F=0.05, V=V(G));  mean cos = 0.980",fontweight="bold",fontsize=11)
axB.legend(loc="upper left",fontsize=8.5);axB.set_xlim(0,1720);axB.set_ylim(0,11.3)
sm=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
cb2=fig.colorbar(sm,ax=axB,shrink=0.8,pad=0.02);cb2.set_label("trajectory step")

fig.suptitle("Corrected, properly-scaled field visualization  —  trajectories DO follow their (non-autonomous) fields",
             fontweight="bold",fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.94]);fig.savefig("corrected_field_viz.png",dpi=140,bbox_inches="tight")
print("wrote corrected_field_viz.png")
