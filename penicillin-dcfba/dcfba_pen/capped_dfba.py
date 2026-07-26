"""capped_dfba.py -- biologically-corrected dFBA:
  * substrate cap (S<=S_CAP)              -> no osmotic-lethal glucose
  * biomass death/autolysis (kd)          -> decline phase
  * glucose-LIMITED uptake                -> no negative S
  * penicillin via Bajpai-Reuss qp(S)     -> production peaks at LOW S (Haldane),
    suppressed at high S (catabolite repression); plus product hydrolysis kh
  FBA solved per step (cached mu(S),vglu(S) tables from core iAL1006).
  Optimises 10-segment feed to maximise titre; saves realistic trajectory."""
import numpy as np
d=np.load("field_tables.npz"); Sg,mu_t,vg_t=d["Sgrid"],d["mu_tab"],d["vg_tab"]
MW_GLC,MW_PENG=0.18016,0.33439; cG=500.0/MW_GLC
mu=lambda S:np.interp(S,Sg,mu_t); vglu=lambda S:np.interp(S,Sg,vg_t)
# Bajpai-Reuss penicillin kinetics (Haldane): peaks at low S, inhibited at high S
MU_P,K_P,K_I=0.045,2.0e-4,0.1
qp=lambda S: MU_P*S/(K_P + S*(1.0+S/K_I))          # g pen / gDW / h
KH=0.01                                            # product hydrolysis [1/h]

S_CAP=35.0; KD=0.02; T=150.0; nFE=10; V0=0.5; X0=0.5
G0=15.0/MW_GLC*V0

def simulate(feed,dt=0.2):
    n=int(T/dt); ts=np.linspace(0,T,n+1)
    G=np.empty(n+1);X=np.empty(n+1);P=np.empty(n+1);V=np.empty(n+1)
    G[0],X[0],P[0],V[0]=G0,X0,0.0,V0
    for k in range(n):
        tt=ts[k]; Vk=max(V[k],1e-6); Xk=max(X[k],1e-12); S=max(G[k],0)/Vk*MW_GLC
        F=feed[min(int(tt/T*nFE),nFE-1)] if V[k]<2.0 else 0.0
        if S>=S_CAP: F=min(F,max(-vglu(S)*Xk/cG,0.0))
        qdem=-vglu(S)*Xk; avail=G[k]/dt+F*cG; qact=min(qdem,max(avail,0.0))
        frac=qact/qdem if qdem>1e-12 else 0.0
        g=mu(S)*frac
        G[k+1]=max(G[k]+(F*cG-qact)*dt,0.0)
        X[k+1]=X[k]+(g-KD)*Xk*dt
        P[k+1]=max(P[k]+(qp(S)*Xk - KH*P[k])*dt,0.0)        # Haldane production - hydrolysis
        V[k+1]=min(V[k]+F*dt,2.0)
    S=G/np.maximum(V,1e-6)*MW_GLC
    return ts,G,X,P,V,S

def score(feed):
    feed=np.clip(feed,0,0.05); ts,G,X,P,V,S=simulate(feed)
    return P[-1]/V[-1] - 1e3*max(S.max()-S_CAP-1,0)**2
print("optimising capped feed (Haldane penicillin) ...")
feed=np.full(nFE,0.015); base=score(feed)
for it in range(60):
    imp=False
    for i in range(nFE):
        for delta in (0.005,-0.005,0.001,-0.001):
            cand=feed.copy(); cand[i]=np.clip(cand[i]+delta,0,0.05)
            v=score(cand)
            if v>base: feed,base=cand,v; imp=True
    if not imp: break
ts,G,X,P,V,S=simulate(feed)
print(f"  feed [L/h]: {np.round(feed,4)}")
print(f"  X_final={X[-1]:.1f} g | P_final={P[-1]:.1f} g | V={V[-1]:.2f} L")
print(f"  biomass conc={X[-1]/V[-1]:.1f} g/L | TITRE={P[-1]/V[-1]:.1f} g/L  (Batch29 ~36)")
print(f"  S: max={S.max():.1f} (cap {S_CAP}) final={S[-1]:.2f} g/L")
print(f"  X peaks {X.max():.1f} g at t={ts[X.argmax()]:.0f}h -> {X[-1]:.1f} g (autolysis)")
print(f"  P peaks {P.max():.1f} g at t={ts[P.argmax()]:.0f}h")
np.savez("capped_traj.npz",t=ts,G=G,X=X,P=P,V=V,S=S,feed=feed,KD=KD,S_CAP=S_CAP)
print("saved capped_traj.npz")
