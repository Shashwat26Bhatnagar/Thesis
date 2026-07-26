"""
flgfn_denoiser.py
=================
Train a Forward-Looking GFlowNet (Pan et al. 2023) over the dcFBA state space
(G,X,V), where the per-transition energy is the LOCAL DIVERGENCE of the flow.

Chain of reasoning (matches the uploaded derivation + FL-GFN paper):
  * A Gaussian p(s,0) over (G,X,V) flows under the dcFBA field f(s) via the
    continuity equation  dp/dt + div(p f) = 0.
  * Along a trajectory,  d eta/dt = -div f   with eta = log p   (CNF log-density).
    => eta is ADDITIVE along the path: eta(s_n)-eta(s_0) = -sum_k div f(s_k) dt.
    This is exactly FL-GFN's additive-energy assumption (Eq.6), with
        E(s) = -eta(s),   R(x)=e^{-E}=p(x),   E(s->s') = +div f(s) dt.
  * Train P_F, P_B, and the forward-looking flow F~ with the FL-DB loss (Eq.11):
        L = ( logF~(s)+logP_F(s'|s) - logF~(s')-logP_B(s|s') + E(s->s') )^2
  * Roll out P_F = the forward "denoiser"; plot trajectories; save the model.

Continuous-state GFlowNet: policies are Gaussian kernels (Lahlou et al. 2023 style).
Field is FAST: mu(S), vglu(S) precomputed 1-D tables (no LP in the loop).
"""
import numpy as np, torch, torch.nn as nn
torch.manual_seed(0); np.random.seed(0)

# ---------------- fast dcFBA field + divergence (raw G,X,V coords) ------------
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016; cG=500.0/MW_GLC
def _mu(S):  return np.interp(S,Sg,mu_t)
def _vg(S):  return np.interp(S,Sg,vg_t)
def _dvg(S): h=1e-3; return (np.interp(S+h,Sg,vg_t)-np.interp(S-h,Sg,vg_t))/(2*h)
F_SNAP,C_SNAP=0.05,0.23                         # autonomous field snapshot (feeding)
def field(s):
    G,X,V=s; V=max(V,1e-6); S=max(G,0)/V*MW_GLC
    return np.array([F_SNAP*cG+_vg(S)*X, _mu(S)*X, F_SNAP])
def divergence(s):
    G,X,V=s; V=max(V,1e-6); S=max(G,0)/V*MW_GLC
    return X*_dvg(S)*(MW_GLC/V) + _mu(S)

# ---------------- generate flow trajectories (the data) -----------------------
SC=np.array([3600.,150.,1.5]); OFF=np.array([0.,0.,0.5])      # normalize: ns=(s-OFF)/SC
def norm(s):  return (s-OFF)/SC
def denorm(n):return n*SC+OFF
MU0=np.array([40.,3.,0.55]); SIG0=np.diag([10.,1.,0.03])**2
DT,NSTEP=0.5,24
def make_traj(m):
    S0=np.random.multivariate_normal(MU0,SIG0,size=m)
    traj=np.zeros((m,NSTEP+1,3)); en=np.zeros((m,NSTEP))
    traj[:,0]=S0
    for k in range(NSTEP):
        for i in range(m):
            s=traj[i,k]; en[i,k]=divergence(s)*DT      # E(s->s') = div*dt
            traj[i,k+1]=s+field(s)*DT
    return traj,en

print("generating flow trajectories (data) ...")
TR,EN=make_traj(256)
print(f"  {TR.shape[0]} trajectories x {NSTEP} steps; "
      f"G {TR[:,:,0].min():.0f}-{TR[:,:,0].max():.0f}, X {TR[:,:,1].min():.1f}-{TR[:,:,1].max():.1f}")

# ---------------- FL-GFN networks --------------------------------------------
def mlp(i,o,h=64):
    return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))
class FLGFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fdrift=mlp(3,3)      # forward drift (denoiser) in normalized space
        self.bdrift=mlp(3,3)      # backward drift
        self.logF  =mlp(3,1)      # log forward-looking flow  logF~(s)
        self.logsig=nn.Parameter(torch.tensor(-2.0))   # shared Gaussian log-std
    def logpF(self,ns,ns2):
        mean=ns+self.fdrift(ns); var=torch.exp(2*self.logsig)
        return (-0.5*((ns2-mean)**2/var).sum(-1)-1.5*(self.logsig*2+np.log(2*np.pi)))
    def logpB(self,ns,ns2):
        mean=ns2+self.bdrift(ns2); var=torch.exp(2*self.logsig)
        return (-0.5*((ns-mean)**2/var).sum(-1)-1.5*(self.logsig*2+np.log(2*np.pi)))
net=FLGFN(); opt=torch.optim.Adam(net.parameters(),lr=3e-3)

NS=torch.tensor(norm(TR),dtype=torch.float32)      # (m,NSTEP+1,3)
E =torch.tensor(EN,dtype=torch.float32)            # (m,NSTEP)
m=NS.shape[0]
print("training FL-DB objective ...")
for it in range(4000):
    bi=torch.randint(0,m,(128,)); ki=torch.randint(0,NSTEP,(128,))
    s=NS[bi,ki]; s2=NS[bi,ki+1]; e=E[bi,ki]
    lF=net.logF(s).squeeze(-1); lF2=net.logF(s2).squeeze(-1)
    res=lF+net.logpF(s,s2)-lF2-net.logpB(s,s2)+e
    fldb=(res**2).mean()
    # flow-matching anchor: forward/backward drifts must reproduce the flow step
    fm = ((net.fdrift(s)-(s2-s))**2).sum(-1).mean() + ((net.bdrift(s2)-(s-s2))**2).sum(-1).mean()
    # terminal anchor: logF~(terminal)=0  (F=R=e^-E => F~=1)
    term=NS[bi,NSTEP]; tan=(net.logF(term).squeeze(-1)**2).mean()
    loss=fldb + 1.0*fm + 0.1*tan
    opt.zero_grad(); loss.backward(); opt.step()
    if it%800==0: print(f"  it {it:4d}  loss {loss.item():.4f}  fldb {fldb.item():.4f}  fm {fm.item():.4f}")
print(f"  final loss {loss.item():.4f}")

# ---------------- roll out the FORWARD DENOISER -------------------------------
print("rolling out forward denoiser ...")
net.eval()
with torch.no_grad():
    s0=np.random.multivariate_normal(MU0,SIG0,size=64)
    ns=torch.tensor(norm(s0),dtype=torch.float32)
    roll=[ns.numpy()]
    for k in range(NSTEP):
        ns=ns+net.fdrift(ns)                 # deterministic mean drift = denoiser
        roll.append(ns.numpy())
roll=np.array([denorm(r) for r in roll])     # (NSTEP+1, 64, 3)
np.savez("denoiser_rollout.npz",roll=roll,data_traj=TR)
torch.save({"state_dict":net.state_dict(),"MU0":MU0,"SIG0":SIG0,"SC":SC,"OFF":OFF,
            "DT":DT,"NSTEP":NSTEP,"F_SNAP":F_SNAP,"C_SNAP":C_SNAP},
           "flgfn_denoiser.pt")
print("saved flgfn_denoiser.pt + denoiser_rollout.npz")
print(f"  denoiser endpoints: G {roll[-1,:,0].mean():.0f}+-{roll[-1,:,0].std():.0f}, "
      f"X {roll[-1,:,1].mean():.1f}+-{roll[-1,:,1].std():.1f}, V {roll[-1,:,2].mean():.2f}")
print(f"  flow     endpoints: G {TR[:,-1,0].mean():.0f}+-{TR[:,-1,0].std():.0f}, "
      f"X {TR[:,-1,1].mean():.1f}+-{TR[:,-1,1].std():.1f}, V {TR[:,-1,2].mean():.2f}")
