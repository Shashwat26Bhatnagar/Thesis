"""
flgfn_faithful.py -- FAITHFUL on-policy FL-GFN over the dcFBA flow, mirroring
ling-pan/FL-GFN (learn_from_fl).  Fixes the straight-line artifact:
  * ON-POLICY: P_F samples the trajectories it trains on (no flow-matching crutch).
  * STATE-FUNCTION energy E(s)=-log p_flow(s,t) from the flowed density (so
    E(s->s')=E(s')-E(s) is path-independent: Assumption 4.1 holds).
  * Discrete time-layered (G,X) grid DAG (FL-GFN is inherently discrete).
Trains FL-DB and plain DB; compares how fast each matches the flowed distribution.
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

# ---- fast dcFBA field (feeding snapshot) ----
d=np.load("field_tables.npz");Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016;cG=500.0/MW_GLC;F_SNAP=0.05
_vg=lambda S:np.interp(S,Sg,vg_t);_mu=lambda S:np.interp(S,Sg,mu_t)
def field(s):
    G,X,V=s;V=max(V,1e-6);S=max(G,0)/V*MW_GLC
    return np.array([F_SNAP*cG+_vg(S)*X,_mu(S)*X,F_SNAP])
MU0=np.array([40.,3.,0.55]);SIG0=np.diag([10.,1.,0.03])**2;DT=1.0;NSTEP=10

# ---- flow many Gaussian samples -> (G,X) trajectories ----
NSAMP=8000
S0=np.random.multivariate_normal(MU0,SIG0,size=NSAMP)
traj=np.zeros((NSAMP,NSTEP+1,2))
s=S0.copy()
for k in range(NSTEP+1):
    traj[:,k,0]=s[:,0]; traj[:,k,1]=s[:,1]
    for i in range(NSAMP): s[i]=s[i]+field(s[i])*DT
Gmin,Gmax=traj[:,:,0].min(),traj[:,:,0].max()
Xmin,Xmax=traj[:,:,1].min(),traj[:,:,1].max()
NG,NX=14,10
def gbin(g):return np.clip(((g-Gmin)/(Gmax-Gmin)*NG).astype(int),0,NG-1)
def xbin(x):return np.clip(((x-Xmin)/(Xmax-Xmin)*NX).astype(int),0,NX-1)
GI=gbin(traj[:,:,0]);XI=xbin(traj[:,:,1])           # (NSAMP, NSTEP+1)

# ---- flowed density p(gi,xi,t) -> state-function energy E=-log p ----
P=np.zeros((NSTEP+1,NG,NX))
for t in range(NSTEP+1):
    for i in range(NSAMP): P[t,GI[i,t],XI[i,t]]+=1
P/=P.sum(axis=(1,2),keepdims=True)
E=-np.log(P+1e-6)                                    # energy per (t,gi,xi)
print(f"flow binned. G[{Gmin:.0f},{Gmax:.0f}] X[{Xmin:.1f},{Xmax:.1f}]  grid {NG}x{NX}x{NSTEP+1}")

# ---- action menu (offsets in bins), from observed per-step jumps ----
dG=GI[:,1:]-GI[:,:-1]; dX=XI[:,1:]-XI[:,:-1]
print(f"per-step bin jumps: dG [{dG.min()},{dG.max()}]  dX [{dX.min()},{dX.max()}]")
GOFF=sorted(set(range(int(dG.min())-1,int(dG.max())+2))|{0}); XOFF=sorted(set(range(int(dX.min())-1,int(dX.max())+2))|{0})
ACT=[(a,b) for a in GOFF for b in XOFF]; nA=len(ACT)
ACTg=torch.tensor([a for a,_ in ACT]); ACTx=torch.tensor([b for _,b in ACT])
print(f"{nA} actions  GOFF={GOFF} XOFF={XOFF}")

Et=torch.tensor(E,dtype=torch.float32)              # (T+1,NG,NX)
# start distribution (t=0 histogram)
p0=P[0].flatten(); start_cells=np.array([(gi,xi) for gi in range(NG) for xi in range(NX)])

def feat(gi,xi,t):  # normalized state features for the nets
    return torch.stack([gi/NG, xi/NX, torch.full_like(gi,t/NSTEP,dtype=torch.float32)],-1)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        h=128
        s.body=nn.Sequential(nn.Linear(3,h),nn.LeakyReLU(),nn.Linear(h,h),nn.LeakyReLU())
        s.pf=nn.Linear(h,nA); s.pb=nn.Linear(h,nA); s.lf=nn.Linear(h,1)
    def forward(s,gi,xi,t):
        z=s.body(feat(gi,xi,t)); return s.pf(z),s.pb(z),s.lf(z).squeeze(-1)

def child(gi,xi,a):
    gi2=torch.clamp(gi+ACTg[a],0,NG-1); xi2=torch.clamp(xi+ACTx[a],0,NX-1); return gi2,xi2
def valid_fwd(gi,xi):  # mask actions whose child stays in-grid AND is reachable (non-clip)
    g2=gi.unsqueeze(1)+ACTg.unsqueeze(0); x2=xi.unsqueeze(1)+ACTx.unsqueeze(0)
    return ((g2>=0)&(g2<NG)&(x2>=0)&(x2<NX)).float()
def valid_bwd(gi,xi):  # parent = current - offset must be in-grid
    g0=gi.unsqueeze(1)-ACTg.unsqueeze(0); x0=xi.unsqueeze(1)-ACTx.unsqueeze(0)
    return ((g0>=0)&(g0<NG)&(x0>=0)&(x0<NX)).float()

def sample_traj(net,mb,explore=0.3):
    idx=np.random.choice(len(p0),size=mb,p=p0)
    gi=torch.tensor(start_cells[idx,0]); xi=torch.tensor(start_cells[idx,1])
    Gs=[gi];Xs=[xi];As=[]
    for t in range(NSTEP):
        with torch.no_grad():
            pf,_,_=net(gi,xi,t); m=valid_fwd(gi,xi)
            logp=(pf-1e9*(1-m)).log_softmax(1)
            msum=m.sum(1,keepdim=True).clamp(min=1.0)
            pr=(1-explore)*logp.exp()+explore*m/msum
            pr=torch.where(pr.sum(1,keepdim=True)>0,pr,m.clamp(min=1e-6))
            a=torch.multinomial(pr,1).squeeze(1)
        gi,xi=child(gi,xi,a); Gs.append(gi);Xs.append(xi);As.append(a)
    return torch.stack(Gs,1),torch.stack(Xs,1),torch.stack(As,1)   # (mb,T+1),(mb,T)

def loss_fldb(net,G,X,A):
    L=0.
    for t in range(NSTEP):
        gi,xi=G[:,t],X[:,t]; gi2,xi2=G[:,t+1],X[:,t+1]; a=A[:,t]
        pf,_,lf=net(gi,xi,t); _,pb2,lf2=net(gi2,xi2,t+1)
        lpf=(pf-1e9*(1-valid_fwd(gi,xi))).log_softmax(1).gather(1,a[:,None]).squeeze(1)
        lpb=(pb2-1e9*(1-valid_bwd(gi2,xi2))).log_softmax(1).gather(1,a[:,None]).squeeze(1)
        e=Et[t+1,gi2,xi2]-Et[t,gi,xi]                 # E(s')-E(s): state-function diff
        anch=(lf2**2).mean() if t==NSTEP-1 else 0.0   # logF~(terminal)=0 => F(term)=R
        L=L+((lf+lpf-lpb-lf2+e)**2).mean()+0.5*anch
    return L/NSTEP

def loss_db(net,G,X,A):     # plain DB: reward only at terminal
    L=0.
    for t in range(NSTEP):
        gi,xi=G[:,t],X[:,t]; gi2,xi2=G[:,t+1],X[:,t+1]; a=A[:,t]
        pf,_,lf=net(gi,xi,t); _,pb2,lf2=net(gi2,xi2,t+1)
        lpf=(pf-1e9*(1-valid_fwd(gi,xi))).log_softmax(1).gather(1,a[:,None]).squeeze(1)
        lpb=(pb2-1e9*(1-valid_bwd(gi2,xi2))).log_softmax(1).gather(1,a[:,None]).squeeze(1)
        if t<NSTEP-1: L=L+((lf+lpf-lpb-lf2)**2).mean()
        else:         L=L+((lf+lpf-lpb-(-Et[NSTEP,gi2,xi2]))**2).mean()  # logF(term)=logR=-E
    return L/NSTEP

target=P[NSTEP].flatten()
def eval_l1(net,M=3000):
    G,X,_=sample_traj(net,M,explore=0.0)
    h=np.zeros(NG*NX)
    for i in range(M): h[G[i,-1].item()*NX+X[i,-1].item()]+=1
    h/=h.sum(); return np.abs(h-target).sum(), h

def train(lossfn,steps=4000):
    net=Net();opt=torch.optim.Adam(net.parameters(),1e-3);curve=[]
    for it in range(steps):
        G,X,A=sample_traj(net,64)
        loss=lossfn(net,G,X,A)
        opt.zero_grad();loss.backward();opt.step()
        if it%100==0: curve.append((it,eval_l1(net)[0]))
    return net,curve

print("training FL-DB ...");  netFL,cFL=train(loss_fldb)
print("training plain DB ..."); netDB,cDB=train(loss_db)
_,hFL=eval_l1(netFL,5000); _,hDB=eval_l1(netDB,5000)
print(f"final L1 to flowed dist:  FL-DB {cFL[-1][1]:.3f}   DB {cDB[-1][1]:.3f}")
np.savez("flgfn_faithful.npz",target=target,hFL=hFL,hDB=hDB,
         cFL=np.array(cFL),cDB=np.array(cDB),NG=NG,NX=NX,
         Gmin=Gmin,Gmax=Gmax*1.04 if False else Gmax,Xmin=Xmin,Xmax=Xmax,P=P,traj=traj)
torch.save(netFL.state_dict(),"flgfn_faithful_PF.pt")
print("saved flgfn_faithful.npz + flgfn_faithful_PF.pt")
