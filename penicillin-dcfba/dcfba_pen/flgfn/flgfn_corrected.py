"""
flgfn_corrected.py -- FL-GFN diffusion sampler on the BIOLOGICALLY-CORRECTED field.
State = (S, X) in concentration units [g/L, g].  FED-BATCH field with DEATH,
covering the FULL 150 h fermentation.  Feed is a STATE-FEEDBACK law F(S,X)
(not a time schedule), so the field stays AUTONOMOUS and needs no time input:
    dS/dt = (F(S,X)*cG + vglu(S)*X)*MW/V   (feed in, glucose drawn down)
    dX/dt = (mu(S) - KD)*X                 (grow, with death/autolysis)
Per-transition energy E(s->s') = divergence(s)*dt  (FL-GFN additive energy, Eq.6).
Trains FL-DB loss (Eq.11); rolls out the forward denoiser. Saves rollout in g/L.
"""
import numpy as np, torch, torch.nn as nn
from torch.distributions import MultivariateNormal   # full 3x3 covariance density
torch.manual_seed(0); np.random.seed(0)
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC=0.18016; KD=0.02; VMAX=2.0                # V is now a STATE; VMAX = vessel limit [L]
cG=500.0/MW_GLC                                  # feed glucose conc [mmol/L]
FMAX,KP=0.05,0.6                                 # max feed [L/h], controller gain [1/h]
S_HI,S_LO,X_SW,W=10.0,0.15,25.0,8.0              # growth->production setpoint switch on X
_mu =lambda S:np.interp(S,Sg,mu_t)
_vg =lambda S:np.interp(S,Sg,vg_t)
_dvg=lambda S:(np.interp(S+1e-3,Sg,vg_t)-np.interp(S-1e-3,Sg,vg_t))/2e-3

def _sset(X):   # setpoint: high S while biomass small (growth), low S later (production)
    return S_LO+(S_HI-S_LO)/(1.0+np.exp((X-X_SW)/W))
def _feed(S,X,V):  # consumption-matching P controller, gated off when vessel is full
    raw=(-_vg(S)*X+KP*(_sset(X)-S)*V/MW_GLC)/cG
    return float(np.clip(raw,0.0,FMAX)/(1.0+np.exp((V-VMAX)/0.02)))
def field(s):
    S,X,V=s; S=max(S,0.0); V=max(V,1e-3)
    F=_feed(S,X,V)
    return np.array([(F*cG+_vg(S)*X)*MW_GLC/V - S*F/V,   # dS  (incl. dilution)
                     (_mu(S)-KD)*X,                       # dX
                     F])                                  # dV
def divergence(s,h=1e-4):                        # numerical: feed law breaks the analytic form
    out=0.0
    for i in range(3):
        sp=list(s); sm=list(s); sp[i]+=h; sm[i]-=h
        out+=(field(sp)[i]-field(sm)[i])/(2*h)
    return float(out)

# ---- forward flow of a Gaussian = the data (batch: grow then decline) ----
MU0=np.array([15.,0.5,0.5]); SIG0=np.diag([2.,0.15,0.03])**2  # inoculum (S0,X0,V0)
DT,NSTEP=1.0,150                                      # 150 x 1.0 h = FULL 150 h fermentation
PNOISE=np.array([0.12,0.06,0.004])                    # process noise /sqrt(h): makes Sigma real
def make_traj(m):
    S0=np.random.multivariate_normal(MU0,SIG0,size=m)
    traj=np.zeros((m,NSTEP+1,3)); en=np.zeros((m,NSTEP)); traj[:,0]=S0
    for k in range(NSTEP):
        for i in range(m):
            s=traj[i,k]; en[i,k]=divergence(s)*DT
            nxt=s+field(s)*DT+PNOISE*np.sqrt(DT)*np.random.randn(3)
            traj[i,k+1]=np.clip(nxt,[0.0,1e-6,1e-3],[np.inf,np.inf,VMAX])
    return traj,en
print("flowing Gaussian under corrected field (data) ...")
TR,EN=make_traj(256)
print(f"  S {TR[:,:,0].min():.1f}-{TR[:,:,0].max():.1f} g/L | X {TR[:,:,1].min():.1f}-{TR[:,:,1].max():.1f} g"
      f" | V {TR[:,:,2].min():.2f}-{TR[:,:,2].max():.2f} L")

# ---- FL-GFN (3D, FULL covariance) ----
SC=np.array([20.,80.,2.5]); OFF=np.array([0.,0.,0.]); norm=lambda s:(s-OFF)/SC
D=3; NTRI=D*(D+1)//2                             # 3 diag + 3 off-diag = 6 Cholesky entries
_DI=torch.arange(D); _TI=torch.tril_indices(D,D,offset=-1)
def _tril(raw):
    """raw (...,6) -> lower-triangular L (...,3,3) with positive diagonal; Sigma = L L^T."""
    L=torch.zeros(*raw.shape[:-1],D,D,dtype=raw.dtype,device=raw.device)
    L[...,_DI,_DI]=nn.functional.softplus(raw[...,:D]-3.0)+1e-3   # -3 bias: start tight
    L[...,_TI[0],_TI[1]]=raw[...,D:]                              # free off-diagonals
    return L
def mlp(i,o,h=64): return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))
class FLGFN(nn.Module):
    def __init__(s):
        super().__init__(); s.fdrift=mlp(D,D); s.bdrift=mlp(D,D); s.logF=mlp(D,1)
        s.fcov=mlp(D,NTRI); s.bcov=mlp(D,NTRI)   # state-dependent FULL covariance factors
    def distF(s,a):  return MultivariateNormal(a+s.fdrift(a),scale_tril=_tril(s.fcov(a)))
    def distB(s,b):  return MultivariateNormal(b+s.bdrift(b),scale_tril=_tril(s.bcov(b)))
    def logpF(s,a,b): return s.distF(a).log_prob(b)
    def logpB(s,a,b): return s.distB(b).log_prob(a)
net=FLGFN(); opt=torch.optim.Adam(net.parameters(),lr=3e-3)
NS=torch.tensor(norm(TR),dtype=torch.float32); E=torch.tensor(EN,dtype=torch.float32); m=NS.shape[0]
print("training FL-DB ...")
for it in range(9000):
    bi=torch.randint(0,m,(256,)); ki=torch.randint(0,NSTEP,(256,))
    s=NS[bi,ki]; s2=NS[bi,ki+1]; e=E[bi,ki]
    res=net.logF(s).squeeze(-1)+net.logpF(s,s2)-net.logF(s2).squeeze(-1)-net.logpB(s,s2)+e
    fm=((net.fdrift(s)-(s2-s))**2).sum(-1).mean()+((net.bdrift(s2)-(s-s2))**2).sum(-1).mean()
    tan=(net.logF(NS[bi,NSTEP]).squeeze(-1)**2).mean()
    loss=(res**2).mean()+300*fm+tan
    opt.zero_grad(); loss.backward(); opt.step()
    if it%3000==0: print(f"  it {it}: FL-DB={(res**2).mean().item():.4f} fm={fm.item():.3e}")

# ---- roll out forward denoiser from fresh Gaussian ----
print("rolling out denoiser ...")
net.eval()
with torch.no_grad():
    s0=np.random.multivariate_normal(MU0,SIG0,size=64)
    ns=torch.tensor(norm(s0),dtype=torch.float32); roll=[ns.numpy()]
    for k in range(NSTEP):
        ns=ns+net.fdrift(ns)            # deterministic rollout (drift matches flow step)
        roll.append(ns.numpy())
roll=np.array(roll)*SC+OFF                     # back to (S,X,V) physical units
roll[:,:,0]=np.clip(roll[:,:,0],0,None); roll[:,:,1]=np.clip(roll[:,:,1],1e-6,None)
roll[:,:,2]=np.clip(roll[:,:,2],1e-3,VMAX)
np.savez("flgfn_corrected.npz",roll=roll,TR=TR,DT=DT,NSTEP=NSTEP,KD=KD,SC=SC,OFF=OFF)
torch.save(net.state_dict(),"flgfn_corrected.pt")   # persist weights for pf_query.PFQuery
print(f"  rollout: S {roll[:,:,0].min():.1f}-{roll[:,:,0].max():.1f} | X {roll[:,:,1].min():.1f}-{roll[:,:,1].max():.1f}"
      f" | V {roll[:,:,2].min():.2f}-{roll[:,:,2].max():.2f}")
with torch.no_grad():                          # inspect the learned FULL covariance mid-run
    a=torch.tensor(norm(TR[:,75].mean(0))[None,:],dtype=torch.float32)
    L=_tril(net.fcov(a))[0]; Sig=(L@L.T).numpy()
    Sig=Sig*np.outer(SC,SC)                    # normalised -> physical units
    sd=np.sqrt(np.diag(Sig)); corr=Sig/np.outer(sd,sd)
    np.set_printoptions(precision=4,suppress=True)
    print("\nlearned Sigma at t=75 h (physical units, 3x3):\n",Sig)
    print("correlation matrix (off-diagonals are NON-ZERO -> full covariance):\n",corr)
gt=TR.mean(0); lr_=roll.mean(1)                     # learned vs true cloud means over 150 h
print(f"{'t[h]':>6}{'S learn':>10}{'S true':>9}{'X learn':>10}{'X true':>9}{'V learn':>10}{'V true':>9}")
for k in range(0,NSTEP+1,25): print(f"{k*DT:6.0f}{lr_[k,0]:10.3f}{gt[k,0]:9.3f}{lr_[k,1]:10.2f}{gt[k,1]:9.2f}{lr_[k,2]:10.3f}{gt[k,2]:9.3f}")
print(f"mean rel err over {NSTEP*DT:.0f} h: S={np.abs(lr_[:,0]-gt[:,0]).mean()/gt[:,0].mean():.2%} "
      f"X={np.abs(lr_[:,1]-gt[:,1]).mean()/gt[:,1].mean():.2%} "
      f"V={np.abs(lr_[:,2]-gt[:,2]).mean()/gt[:,2].mean():.2%}")
print("saved flgfn_corrected.npz / flgfn_corrected.pt")
