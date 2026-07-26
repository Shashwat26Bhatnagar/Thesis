"""
compute_ct.py -- compute the time-varying ratio c(t) = P(t)/X(t) for the dcFBA
from the IndPenSim CSV.  PURE STANDARD LIBRARY: no numpy, no pandas -> immune to
the NumPy 1.x/2.x binary-incompatibility errors.

Uses real OFFLINE biomass + penicillin (~26 samples/batch), picks the OPTIMAL
batch (highest penicillin), computes c=P/X, prints the `const C_PX = [...]` array.

USAGE:
    python compute_ct.py                                   # default CSV, best batch
    python compute_ct.py 100_Batches_IndPenSim_V3.csv      # explicit path
    python compute_ct.py 100_Batches_IndPenSim_V3.csv 29   # force a batch number
"""
import sys, csv

CSV_PATH    = sys.argv[1] if len(sys.argv) > 1 else "100_Batches_IndPenSim_V3.csv"
force_batch = int(sys.argv[2]) if len(sys.argv) > 2 else None
N_FE        = 10           # must match nFE in the Julia
TIME_MODE   = "fraction"   # "fraction" (batch progress 0..1) or "absolute" (real time)

with open(CSV_PATH, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)
    rows = list(reader)

def col_idx(*needles):
    for j, name in enumerate(header):
        low = name.lower()
        if all(n.lower() in low for n in needles): return j
    raise KeyError(f"no column matching {needles}; check the CSV header")
jT    = col_idx("time")
jPON  = col_idx("penicillin concentration", "p:g/l")
jXOFF = col_idx("x_offline")
jPOFF = col_idx("p_offline")
print(f"columns -> time[{jT}], P_online[{jPON}], X_offline[{jXOFF}], P_offline[{jPOFF}]")

def fnum(s):
    s = s.strip()
    if s == "" or s.lower() in ("nan", "na"): return None
    try: return float(s)
    except ValueError: return None

T    = [fnum(r[jT])    for r in rows]
PON  = [fnum(r[jPON])  for r in rows]
XOFF = [fnum(r[jXOFF]) for r in rows]
POFF = [fnum(r[jPOFF]) for r in rows]

# batch boundaries: each batch starts where Time == 0.2
starts = [i for i, v in enumerate(T) if v is not None and abs(v - 0.2) < 1e-9]
ends   = [s - 1 for s in starts[1:]] + [len(T) - 1]
print(f"found {len(starts)} batches")

def batch_maxP(i):
    vals = [PON[k] for k in range(starts[i], ends[i]+1) if PON[k] is not None]
    return max(vals) if vals else float("-inf")

bi = (force_batch - 1) if force_batch is not None else max(range(len(starts)), key=batch_maxP)
s, e = starts[bi], ends[bi]
dur = max(T[k] for k in range(s, e+1) if T[k] is not None)
print(f"using batch #{bi+1}  (max online P = {batch_maxP(bi):.2f} g/L, duration {dur:.0f} h)")

# real offline samples in this batch (drop rows missing X or P)
pts = []
for k in range(s, e+1):
    if XOFF[k] is not None and POFF[k] is not None and T[k] is not None:
        pts.append((T[k], XOFF[k], POFF[k]))
pts.sort()
t = [p[0] for p in pts]
c = [p[2] / p[1] if p[1] > 0 else 0.0 for p in pts]   # c = P/X
print(f"{len(pts)} offline samples;  final measured c = P/X = {c[-1]:.3f} g pen/g biomass")

def lin_interp(xq, xs, ys):              # manual linear interpolation (clamped)
    if xq <= xs[0]:  return ys[0]
    if xq >= xs[-1]: return ys[-1]
    for k in range(1, len(xs)):
        if xq <= xs[k]:
            f = (xq - xs[k-1]) / (xs[k] - xs[k-1])
            return ys[k-1] + f * (ys[k] - ys[k-1])
    return ys[-1]

t0, t1 = t[0], t[-1]
if TIME_MODE == "fraction":
    xax  = [(ti - t0) / (t1 - t0) for ti in t]
    mids = [(i - 0.5) / N_FE for i in range(1, N_FE + 1)]
else:
    xax  = t
    h = (t1 - t0) / N_FE
    mids = [t0 + h * (i - 0.5) for i in range(1, N_FE + 1)]
c_fe = [lin_interp(mm, xax, c) for mm in mids]

print("\nc(t) = P/X at finite-element midpoints:")
for mm, cm in zip(mids, c_fe):
    lab = f"frac={mm:.2f}" if TIME_MODE == "fraction" else f"t={mm:6.1f}h"
    print(f"   FE {lab}   c={cm:.4f}")
print("\n--- paste into penicillin_dcfba.jl AND verify_ial1006.py ---")
print("const C_PX = [" + ", ".join(f"{v:.4f}" for v in c_fe) + "]")
