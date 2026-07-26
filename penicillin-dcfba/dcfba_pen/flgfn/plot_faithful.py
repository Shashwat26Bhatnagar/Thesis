import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
d=np.load("flgfn_faithful.npz")
NG,NX=int(d["NG"]),int(d["NX"]);target=d["target"].reshape(NG,NX)
hFL=d["hFL"].reshape(NG,NX);hDB=d["hDB"].reshape(NG,NX)
cFL,cDB=d["cFL"],d["cDB"];Gmin,Gmax=float(d["Gmin"]),float(d["Gmax"]);Xmin,Xmax=float(d["Xmin"]),float(d["Xmax"])
ext=[Xmin,Xmax,Gmin,Gmax]
fig,ax=plt.subplots(1,4,figsize=(18,4.6))
vmax=max(target.max(),hFL.max(),hDB.max())
for a,h,t in zip(ax[:3],[target,hFL,hDB],
    ["TARGET  flowed density $p(s,T)$","FL-DB  sampled  (L1=%.2f)"%cFL[-1,1],"plain DB  sampled  (L1=%.2f)"%cDB[-1,1]]):
    im=a.imshow(h,origin="lower",aspect="auto",extent=ext,cmap="magma",vmin=0,vmax=vmax)
    a.set_title(t,fontweight="bold",fontsize=10.5);a.set_xlabel("X biomass [g]");a.set_ylabel("G glucose [mmol]")
fig.colorbar(im,ax=ax[2],shrink=0.85,label="prob")
ax[3].plot(cFL[:,0],cFL[:,1],lw=2.2,color="crimson",label="FL-DB (dense credit)")
ax[3].plot(cDB[:,0],cDB[:,1],lw=2.2,color="steelblue",label="plain DB (terminal only)")
ax[3].set_xlabel("training step");ax[3].set_ylabel("L1 distance to flowed dist.")
ax[3].set_title("Credit assignment over a 10-step horizon",fontweight="bold",fontsize=10.5)
ax[3].grid(alpha=.3);ax[3].legend(fontsize=9.5);ax[3].set_ylim(0,2.05)
fig.suptitle("Faithful on-policy FL-GFN over the dcFBA flow  (terminal $G$–$X$ distribution).  "
             "FL-DB matches the flowed density; plain DB cannot assign credit over 10 steps.",
             fontweight="bold",fontsize=12.5)
fig.tight_layout(rect=[0,0,1,0.94]);fig.savefig("flgfn_faithful.png",dpi=140,bbox_inches="tight")
print("wrote flgfn_faithful.png")
