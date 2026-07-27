import glob, os
import numpy as np

PENSIMENV = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensimenv"
PENSIM    = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim"
ACT = ["Discharge","Sugar","Soilbean","Aeration","BackPres","WaterInj"]
OBS = ["pH","Temp","Fa","Fb","Fc","Fh","Wt","DO2"]
SETPOINT = np.array([200.059,77.410,25.803,63.991,0.939,154.953])
PHASES = [(0,35,"lag/growth"),(35,51,"transition"),(51,1e9,"production")]
GROUPS = [("random (recipe)", PENSIMENV+"/random_batch_*.csv"),
          ("gpei (BayesOpt)", PENSIMENV+"/gpei_batch_*.csv"),
          ("CDIL unbounded",  PENSIM+"/cdil_batch_*.csv"),
          ("CDIL +/-10%",     PENSIM+"/cdil10pct_batch_*.csv")]
MIN_ROWS = 100

def load(p):
    out=[]
    for f in sorted(glob.glob(p)):
        d=np.atleast_2d(np.genfromtxt(f,delimiter=",",skip_header=1))
        if d.shape[0]>=MIN_ROWS: out.append((os.path.basename(f),d))
    return out

data = {n: load(p) for n,p in GROUPS}

print("="*78); print("TOTAL YIELD PER BATCH"); print("="*78)
print(f"{'dataset':20s}{'n':>4}{'mean':>11}{'std':>10}{'min':>11}{'max':>11}")
print("-"*78)
stats={}
for n,r in data.items():
    if not r: print(f"{n:20s}   -- none --"); continue
    y=np.array([d[:,-1].sum() for _,d in r]); stats[n]=y
    print(f"{n:20s}{len(y):>4}{y.mean():>11.1f}{y.std():>10.1f}{y.min():>11.1f}{y.max():>11.1f}")
if "gpei (BayesOpt)" in stats:
    ref=stats["gpei (BayesOpt)"].mean()
    print(f"\nrelative to gpei = {ref:.1f}:")
    for n,y in stats.items(): print(f"  {n:20s} {100*y.mean()/ref:7.1f}%")

print("\n"+"="*78); print("MEAN ACTION vs SETPOINT"); print("="*78)
print(f"{'dataset':20s}"+"".join(f"{a:>10s}" for a in ACT))
print(f"{'SETPOINT':20s}"+"".join(f"{v:>10.1f}" for v in SETPOINT)); print("-"*78)
for n,r in data.items():
    if not r: continue
    m=np.mean([d[:,1:7].mean(0) for _,d in r],axis=0)
    print(f"{n:20s}"+"".join(f"{v:>10.2f}" for v in m))
    print(f"{'  ratio':20s}"+"".join(f"{v:>10.2f}" for v in m/SETPOINT))

print("\n"+"="*78); print("PER-PHASE ACTION AND YIELD"); print("="*78)
for n,r in data.items():
    if not r: continue
    print(f"\n{n}")
    print(f"  {'phase':13s}"+"".join(f"{a:>10s}" for a in ACT)+f"{'yield':>12s}")
    for lo,hi,lbl in PHASES:
        A,Y=[],[]
        for _,d in r:
            m=(d[:,0]>=lo)&(d[:,0]<hi)
            if m.any(): A.append(d[m,1:7].mean(0)); Y.append(d[m,-1].sum())
        if A:
            a=np.mean(A,axis=0)
            print(f"  {lbl:13s}"+"".join(f"{v:>10.2f}" for v in a)+f"{np.mean(Y):>12.1f}")

print("\n"+"="*78); print("STATE COVERAGE (std)"); print("="*78)
print(f"{'dataset':20s}"+"".join(f"{o:>10s}" for o in OBS)); print("-"*78)
for n,r in data.items():
    if not r: continue
    a=np.vstack([d[:,7:15] for _,d in r])
    print(f"{n:20s}"+"".join(f"{v:>10.3g}" for v in a.std(0)))
