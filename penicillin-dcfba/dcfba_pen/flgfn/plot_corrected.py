"""plot_corrected.py -- biologically-defensible figure in g/L:
  A) realistic fed-batch dFBA trajectory (cap+death+Haldane) in S-X, with local field
  B) FL-GFN diffusion sampler on the corrected batch field (F=0, death), S-X g/L."""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.cm as cm

d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016; cG=500.0/MW_GLC; KD=0.02
_mu=lambda S:np.interp(S,Sg,mu_t); _vg=lambda S:np.interp(S,Sg,vg_t)

fig,(axA,axB)=plt.subplots(1,2,figsize=(15.5,6.4))

# ============ Panel A: realistic fed-batch trajectory + local field ============
tr=np.load("capped_traj.npz"); t,S,X,P,V=tr["t"],tr["S"],tr["X"],tr["P"],tr["V"]
feed=tr["feed"]; T=150.0; nFE=10
def Ft(tt,Vv): return (feed[min(int(tt/T*nFE),nFE-1)] if Vv<2.0 else 0.0)
# local field in (S,X): dS=(F*cG - vglu*X... ) wait use concentration form
# dS/dt = MW/V*(F*cG + vglu(S)*X) ; dX/dt=(mu(S)-KD)*X  (glucose-limited handled in sim)
Sr=S.max()-S.min()+1e-9; Xr=X.max()-X.min()+1e-9
idx=np.arange(0,len(t)-1,40)
dS=np.array([MW_GLC/max(V[i],1e-6)*(Ft(t[i],V[i])*cG+_vg(max(S[i],0))*X[i]) for i in idx])
dX=np.array([(_mu(max(S[i],0))-KD)*X[i] for i in idx])
uh=dS/Sr; vh=dX/Xr; nn_=np.sqrt(uh**2+vh**2)+1e-12; L=0.05
axA.quiver(S[idx],X[idx],uh/nn_*L*Sr,vh/nn_*L*Xr,angles="xy",scale_units="xy",scale=1,
           color="steelblue",width=0.005,headwidth=4,zorder=3,alpha=0.85,label="local field")
sc=axA.scatter(S,X,c=t,cmap="viridis",s=16,zorder=4)
axA.plot(S,X,color="gray",lw=0.8,alpha=0.4,zorder=2)
axA.scatter([S[0]],[X[0]],color="k",s=80,zorder=5,label="start")
axA.scatter([S[-1]],[X[-1]],color="crimson",marker="*",s=200,zorder=5,label="end (autolysis)")
axA.axvspan(35,axA.get_xlim()[1],color="red",alpha=0.05)
axA.set_xlabel("S  substrate [g/L]");axA.set_ylabel("X  biomass [g]")
axA.set_title(f"Realistic fed-batch dFBA  (cap+death+Haldane)\n"
              f"biomass {X[-1]/V[-1]:.0f} g/L, titre {P[-1]/V[-1]:.0f} g/L, S$\\leq$35 g/L",
              fontweight="bold",fontsize=11)
axA.legend(loc="upper right",fontsize=9); axA.set_xlim(-1,38)
cb=fig.colorbar(sc,ax=axA,shrink=0.85,pad=0.02); cb.set_label("time [h]")

# ============ Panel B: FL-GFN sampler on corrected batch field ============
fl=np.load("flgfn_corrected.npz"); roll=fl["roll"]; nstep=roll.shape[0]-1; npart=roll.shape[1]
# batch field (F=0, death) for streamlines
def fieldSX(S,X): return _vg(np.maximum(S,0))*X*MW_GLC, (_mu(np.maximum(S,0))-KD)*X
Sgrid=np.linspace(0.5,45,30); Xgrid=np.linspace(0.5,28,30); SS,XX=np.meshgrid(Sgrid,Xgrid)
dSb,dXb=fieldSX(SS,XX); spd=np.sqrt((dSb/45)**2+(dXb/28)**2)
axB.streamplot(SS,XX,dSb,dXb,color=np.log10(spd+1e-6),cmap="viridis",density=1.2,linewidth=0.8,arrowsize=0.9,zorder=1)
# alignment cosine
cs=[]
for j in range(npart):
    for k in range(nstep):
        fS,fX=fieldSX(roll[k,j,0],roll[k,j,1]); tS,tX=roll[k+1,j]-roll[k,j]
        fv=np.array([fS/45,fX/28]); tv=np.array([tS/45,tX/28])
        if np.linalg.norm(fv)>1e-9 and np.linalg.norm(tv)>1e-9:
            cs.append(np.dot(fv,tv)/(np.linalg.norm(fv)*np.linalg.norm(tv)))
mc=np.mean(cs)
tcol=cm.plasma(np.linspace(0,1,nstep+1))
for j in range(0,npart,3):
    for k in range(nstep):
        axB.plot(roll[k:k+2,j,0],roll[k:k+2,j,1],color=tcol[k],lw=1.0,alpha=0.7,zorder=3)
axB.scatter(roll[0,:,0],roll[0,:,1],c="royalblue",s=24,zorder=4,label="p(s,0) initial")
axB.scatter(roll[-1,:,0],roll[-1,:,1],c="crimson",marker="*",s=55,zorder=5,label="p(s,T) sampled")
axB.set_xlabel("S  substrate [g/L]");axB.set_ylabel("X  biomass [g]")
axB.set_title(f"FL-GFN diffusion sampler on corrected field\n"
              f"batch (F=0) with death;  mean cos(tangent,field)={mc:.3f}",fontweight="bold",fontsize=11)
axB.legend(loc="upper right",fontsize=9); axB.set_xlim(0,45); axB.set_ylim(0,28)
sm=plt.cm.ScalarMappable(cmap="plasma",norm=plt.Normalize(0,nstep))
cb2=fig.colorbar(sm,ax=axB,shrink=0.85,pad=0.02); cb2.set_label("trajectory step")

fig.suptitle("Biologically-corrected dcFBA + FL-GFN  (concentration units, g/L)",
             fontweight="bold",fontsize=13)
fig.tight_layout(rect=[0,0,1,0.94]); fig.savefig("corrected_biological.png",dpi=140,bbox_inches="tight")
print(f"wrote corrected_biological.png   (FL-GFN mean cos={mc:.3f})")
