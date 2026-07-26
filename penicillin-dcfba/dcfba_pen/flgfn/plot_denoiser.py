import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("denoiser_rollout.npz"); roll=d["roll"]; TR=d["data_traj"]   # roll:(NSTEP+1,64,3)
ft=np.load("field_tables.npz");Sg,mu_t,vg_t=ft["Sgrid"],ft["mu_tab"],ft["vg_tab"]
MW_GLC=0.18016
def _vg(S):return np.interp(S,Sg,vg_t)
def _dvg(S):h=1e-3;return (np.interp(S+h,Sg,vg_t)-np.interp(S-h,Sg,vg_t))/(2*h)
def _mu(S):return np.interp(S,Sg,mu_t)
DT=0.5
def div(s):G,X,V=s;V=max(V,1e-6);S=max(G,0)/V*MW_GLC;return X*_dvg(S)*(MW_GLC/V)+_mu(S)

GMAX,XMAX,V0,V1=1900.,12.,0.5,1.25
nG=lambda g:g/GMAX;nX=lambda x:x/XMAX;nV=lambda v:(v-V0)/(V1-V0)
S0=roll[0]; ST=roll[-1]                       # initial Gaussian / final cloud

fig=plt.figure(figsize=(15,6))
# ---- panel A: 3D transport of the Gaussian by the forward denoiser ----
ax=fig.add_subplot(121,projection="3d")
for j in range(roll.shape[1]):
    ax.plot(nG(roll[:,j,0]),nX(roll[:,j,1]),nV(roll[:,j,2]),color="0.6",lw=0.5,alpha=0.5,zorder=1)
ax.scatter(nG(S0[:,0]),nX(S0[:,1]),nV(S0[:,2]),c="royalblue",s=22,depthshade=False,label="initial Gaussian  p(s,0)",zorder=5)
ax.scatter(nG(ST[:,0]),nX(ST[:,1]),nV(ST[:,2]),c="crimson",s=26,marker="*",depthshade=False,label="denoised  p(s,T)",zorder=6)
ax.set_xticks([nG(x) for x in [0,500,1000,1500]]);ax.set_xticklabels(["0","500","1000","1500"])
ax.set_yticks([nX(x) for x in [0,4,8,12]]);ax.set_yticklabels(["0","4","8","12"])
ax.set_zticks([nV(x) for x in [0.5,0.75,1.0,1.25]]);ax.set_zticklabels(["0.5","0.75","1.0","1.25"])
ax.set_xlabel("G glucose [mmol]",labelpad=8);ax.set_ylabel("X biomass [g]",labelpad=8);ax.set_zlabel("V vol [L]",labelpad=2)
ax.set_xlim(0,1);ax.set_ylim(0,1);ax.set_zlim(0,1);ax.view_init(elev=20,azim=-62)
ax.set_title("Forward denoiser transports the Gaussian\nalong the dcFBA field (FL-GFN $P_F$)",fontweight="bold",fontsize=10.5)
ax.legend(loc="upper left",fontsize=8.5)

# ---- panel B: additive FL-GFN credit (local divergence energy) ----
ax2=fig.add_subplot(122)
nrep=12
for j in range(nrep):
    e=np.array([div(roll[k,j])*DT for k in range(roll.shape[0]-1)])  # E(s->s')=div*dt
    eta=-np.concatenate([[0],np.cumsum(e)])                          # eta(t)-eta(0) = -sum E
    ax2.plot(np.arange(len(e)),e,color="teal",alpha=0.35,lw=1)
ax2.plot([],[],color="teal",label=r"per-step energy  $E(s\to s')=\nabla\!\cdot\!f\,\Delta t$")
e0=np.array([div(roll[k,0])*DT for k in range(roll.shape[0]-1)])
ax2b=ax2.twinx()
for j in range(nrep):
    e=np.array([div(roll[k,j])*DT for k in range(roll.shape[0]-1)])
    eta=-np.concatenate([[0],np.cumsum(e)])
    ax2b.plot(np.arange(len(eta)),eta,color="crimson",alpha=0.35,lw=1)
ax2b.plot([],[],color="crimson",label=r"cumulative  $\eta(t)-\eta_0=-\sum E$")
ax2.set_xlabel("trajectory step k");ax2.set_ylabel("per-step energy  $E(s\\to s')$",color="teal")
ax2b.set_ylabel(r"cumulative log-density change  $\Delta\eta$",color="crimson")
ax2.tick_params(axis='y',labelcolor="teal");ax2b.tick_params(axis='y',labelcolor="crimson")
ax2.set_title("Additive energy = the dense FL-GFN credit signal\n(available at every step, not just the terminal reward)",fontweight="bold",fontsize=10.5)
ax2.grid(alpha=0.25)
l1,la1=ax2.get_legend_handles_labels();l2,la2=ax2b.get_legend_handles_labels()
ax2.legend(l1+l2,la1+la2,loc="upper right",fontsize=8.5)
fig.suptitle("FL-GFN over dcFBA state space:  Gaussian $\\to$ denoiser $\\to$ flowed distribution,  "
             "trained with local divergence credit",fontweight="bold",fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.95]);fig.savefig("flgfn_denoiser.png",dpi=140,bbox_inches="tight")
print("wrote flgfn_denoiser.png")
