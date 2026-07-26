"""
print_forward_trajectory.py
Roll out the trained forward DENOISER (FL-GFN P_F) through the (G,X,V) state
space and PRINT the forward trajectory step by step.  Needs flgfn_denoiser.pt.

    python3 print_forward_trajectory.py            # 1 detailed trajectory
    python3 print_forward_trajectory.py 5          # 5 sample trajectories
"""
import sys, numpy as np, torch, torch.nn as nn

NTRAJ = int(sys.argv[1]) if len(sys.argv) > 1 else 1
MW_GLC = 0.18016

# --- rebuild the network skeleton so the saved weights load ---
def mlp(i,o,h=64):
    return nn.Sequential(nn.Linear(i,h),nn.SiLU(),nn.Linear(h,h),nn.SiLU(),nn.Linear(h,o))
class FLGFN(nn.Module):
    def __init__(s):
        super().__init__()
        s.fdrift=mlp(3,3); s.bdrift=mlp(3,3); s.logF=mlp(3,1)
        s.logsig=nn.Parameter(torch.tensor(-2.0))
ckpt=torch.load("flgfn_denoiser.pt",map_location="cpu",weights_only=False)
net=FLGFN(); net.load_state_dict(ckpt["state_dict"]); net.eval()
MU0,SIG0=ckpt["MU0"],ckpt["SIG0"]; SC,OFF=ckpt["SC"],ckpt["OFF"]
NSTEP,DT=ckpt["NSTEP"],ckpt["DT"]
norm=lambda s:(s-OFF)/SC; denorm=lambda n:n*SC+OFF

def rollout(s0):
    ns=torch.tensor(norm(s0),dtype=torch.float32)[None]
    out=[denorm(ns[0].numpy())]
    with torch.no_grad():
        for _ in range(NSTEP):
            ns=ns+net.fdrift(ns); out.append(denorm(ns[0].numpy()))
    return np.array(out)                      # (NSTEP+1, 3)

np.random.seed(0)
print(f"# forward denoiser trajectory over (G,X,V)   field snapshot F={ckpt['F_SNAP']}, c={ckpt['C_SNAP']}")
print(f"# {NSTEP} steps, dt={DT} h\n")
for j in range(NTRAJ):
    s0=np.random.multivariate_normal(MU0,SIG0)
    tr=rollout(s0)
    if NTRAJ>1: print(f"=== trajectory {j+1} ===")
    print(f"{'step':>4} {'t[h]':>6} {'G[mmol]':>10} {'X[g]':>9} {'V[L]':>7} {'S[g/L]':>8}")
    for k,(G,X,V) in enumerate(tr):
        S=max(G,0)/max(V,1e-9)*MW_GLC
        print(f"{k:>4} {k*DT:>6.1f} {G:>10.2f} {X:>9.3f} {V:>7.3f} {S:>8.2f}")
    print()
