"""
print_denoiser_traj.py -- print the forward denoiser diffusion trajectory
on the (G, X, V) state space.  Loads the saved rollout and prints each step.
"""
import numpy as np

d = np.load("denoiser_rollout.npz")
roll = d["roll"]          # (nstep+1, nparticles, 3)
nstep, npart, _ = roll.shape[0]-1, roll.shape[1], roll.shape[2]

print("=" * 70)
print("FORWARD DENOISER DIFFUSION  -- state-space trajectory")
print(f"  {npart} particles  x  {nstep} steps")
print(f"  states:  G = total glucose [mmol]  |  X = biomass [g]  |  V = volume [L]")
print("=" * 70)

# print one representative particle in full, then per-step cloud statistics
REP = 0    # index of the representative particle to trace

print(f"\n--- Representative particle #{REP} ---")
print(f"{'step':>4}  {'G [mmol]':>10}  {'X [g]':>8}  {'V [L]':>7}")
print("-" * 35)
for k in range(nstep + 1):
    G, X, V = roll[k, REP]
    print(f"{k:>4}  {G:>10.1f}  {X:>8.3f}  {V:>7.4f}")

print(f"\n--- All {npart} particles: cloud statistics per step ---")
print(f"{'step':>4}  {'G mean':>9}  {'G std':>7}  {'X mean':>8}  {'X std':>6}  {'V mean':>7}")
print("-" * 50)
for k in range(nstep + 1):
    G = roll[k, :, 0]; X = roll[k, :, 1]; V = roll[k, :, 2]
    print(f"{k:>4}  {G.mean():>9.1f}  {G.std():>7.1f}  {X.mean():>8.3f}  {X.std():>6.3f}  {V.mean():>7.4f}")

print("\n--- Initial vs final cloud summary ---")
for label, k in [("initial (t=0)", 0), ("final   (t=T)", nstep)]:
    G = roll[k, :, 0]; X = roll[k, :, 1]; V = roll[k, :, 2]
    print(f"  {label}:  G={G.mean():.1f}+-{G.std():.1f}  X={X.mean():.3f}+-{X.std():.3f}  V={V.mean():.4f}+-{V.std():.4f}")
