#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
policy_learning/explore_with_policy.py

Run the CDIL-trained policy on the REAL PenSim environment and save each episode as
a time-series CSV in the exact format PeniControlData reads, so the data can be fed
straight back into world-model training.

    python policy_learning/explore_with_policy.py -n 1 -max_steps 20     # smoke test
    python policy_learning/explore_with_policy.py -n 10 -p_dropout 0.25
    python policy_learning/explore_with_policy.py -n 10 -clip10 -tag cdil10pct

NO WORLD MODEL IS NEEDED HERE. The phase models were only used to train the policy;
exploration runs the policy against the real simulator.

=== UNIT CHAIN (the part that silently breaks things) ===
The policy lives in z-scored space, the env in physical units, separated by TWO
transforms:

    physical  --PeniControlData min-max-->  [-1,1]  --Standardizer-->  z-scored
              <--------------------------          <----------------

CRITICAL: the Standardizer was fitted on data normalised by PeniControlData's OWN
bounds, which are NOT PenSimEnvGym's defaults -- they come out exactly half (time
276.0 vs 552.0). Using the env's bounds here would put every channel on the wrong
scale. The bounds are therefore read off a PeniControlData instance at runtime.

=== NUMPY ABI NOTE ===
This environment has two numpy module objects loaded, so torch's .numpy() returns an
ndarray of a FOREIGN type: multiplying it by a normal array raises a ufunc error, and
numpy then crashes while formatting that error ("TypeError" from dtype_is_implied).
Every torch->numpy hop therefore goes through .tolist().

=== ACTION CLIPPING: STATIC vs TIME-VARYING (-clip10 vs -cliprecipe) ===
-clip10 clips to +/-10% of the DATASET-MEAN action, one fixed band for all 230 h.
That band is wrong, and measurably so. DISCHARGE_DEFAULT_PROFILE is a STEP function:
0 until t=100 h, then pulses 0 <-> 4000 every 20 h. Its dataset mean is ~200 -- a value
the recipe never actually holds. So the static band [180, 220] FORCES ~180 L/h of
discharge during the first 100 h, when the correct action is 0.

Measured in bnd_rbf_iter0_batch_2.csv: discharge sits at 180.05 (exactly 0.9*mean) for
the whole second half of the batch, i.e. the policy wants to go LOWER and the clip
stops it. Continuously draining ~180 L/h removes product that should be accumulating,
which is consistent with the yield gap (gpei 3.30/step vs CDIL ~3.00).

-cliprecipe instead clips to +/-10% of the recipe profile AT THE CURRENT TIME, read
from RecipeCombo.get_values_dict_at(t). Two adjustments are needed because the profile
is a step function:
    floor    a zero setpoint gives a zero-width band, which is unsatisfiable
    smooth   the profile jumps 0->4000 within 2 h; an unsmoothed edge makes the bound
             discontinuous between consecutive 12-minute steps

=== ACTION MAGNITUDE WARNING ===
The policy was trained with a flat u_max = 3.0 in z-units. The PenSim docs restrict
the search space to +/-10% of the setpoint recipe, which in z-units is
    [0.023, 0.323, 0.554, 0.628, 0.702, 0.103]
i.e. the policy may emit actions 4x-128x wider than the process permits. Actions are
always clipped to the env's own physical bounds; pass -clip10 to additionally clip to
+/-10% of the recipe setpoint, which is what a real reactor would accept.

=== TERMINAL error_reward (fixed) ===
At episode termination PenSim's done_calculator returns error_reward = -100 INSTEAD
of a yield. Writing that row put a spurious -100 in every CSV -- always the last row,
regardless of what the policy did (the actions on that step are unremarkable and well
inside every bound). It depressed every batch total by exactly 100 and, since the
shipped gpei/random baselines contain no such row, biased every comparison against
our runs: one file measured 3712.3 with it and 3812.3 without, i.e. 3.3179 mean/step
against the paper's 3.3071 baseline -- above it rather than 13% below.

The terminal row is therefore DROPPED: it carries no yield information, and the state
it records is the post-termination state.

CSV FORMAT (matches random_batch_*.csv exactly, 16 columns):
    Time Step, <6 actions>, <8 observations>, Yield Per Step
"""
import argparse
import os
import sys
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import model_learning.pensim_dataset as pdata
import policy_learning.Policy as Policy
from policy_learning.policy_variants import rebuild_policy
from policy_learning.chance_constraints import RecipeBounds

from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA
from pensimpy.data.constants import (FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE,
                                     FG_DEFAULT_PROFILE, PRESS_DEFAULT_PROFILE,
                                     DISCHARGE_DEFAULT_PROFILE, WATER_DEFAULT_PROFILE,
                                     PAA_DEFAULT_PROFILE)
from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData

dtype, device = torch.float64, torch.device("cpu")

# ----------------------------------------------------------------- config ----
OUT_DIR = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensim"
POLICY_PATH_DEFAULT = os.path.join(_REPO, "results_pensim", "cdil_policy_phasemodels.pt")

CSV_COLUMNS = [
    "Time Step",
    "Discharge rate", "Sugar feed rate", "Soil bean feed rate",
    "Aeration rate", "Back pressure", "Water injection/dilution",
    "pH", "Temperature", "Acid flow rate", "Base flow rate",
    "Cooling water", "Heating water", "Vessel Weight",
    "Dissolved oxygen concentration",
    "Yield Per Step",
]

U_MAX_FLAT = 3.0                 # must match training (ENFORCE_ACTION_LIMITS was False)
ERROR_REWARD = -100.0            # PenSimEnvGym error_reward, returned at termination
ACTION_LIMIT_FRAC = 0.10

_p = argparse.ArgumentParser("explore PenSim with the CDIL-trained policy")
_p.add_argument("-n", type=int, default=5, help="episodes to collect")
_p.add_argument("-policy", type=str, default=POLICY_PATH_DEFAULT)
_p.add_argument("-out", type=str, default=OUT_DIR)
_p.add_argument("-tag", type=str, default="cdil", help="prefix -> <tag>_batch_<i>.csv")
_p.add_argument("-seed", type=int, default=0)
_p.add_argument("-p_dropout", type=float, default=0.0,
                help="policy dropout during rollout; >0 gives more diverse exploration")
_p.add_argument("-clip10", action="store_true",
                help="clip to +/-10%% of the DATASET-MEAN action (static band; note the "
                     "recipe is a step function, so this forces discharge during the "
                     "first 100 h when the correct action is 0)")
_p.add_argument("-cliprecipe", action="store_true",
                help="clip to +/-10%% of the RECIPE PROFILE at the current time "
                     "(time-varying; the physically meaningful constraint)")
_p.add_argument("-recipe_frac", type=float, default=0.10,
                help="half-width of the recipe band, as a fraction of the setpoint")
_p.add_argument("-recipe_floor", type=float, default=0.05,
                help="minimum half-width as a fraction of the channel span, so a ZERO "
                     "setpoint does not give a zero-width band")
_p.add_argument("-recipe_smooth", type=float, default=2.0,
                help="average the profile over +/-this many hours (it steps 0->4000 "
                     "within 2 h)")
_p.add_argument("-max_steps", type=int, default=None, help="cap steps (debug)")
args = _p.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)


def _to_np(t):
    """torch tensor -> numpy, via .tolist() to dodge the duplicate-numpy ABI clash."""
    return np.array(t.detach().reshape(-1).tolist(), dtype=np.float64)


# ============================================================== unit chain ====
# PeniControlData's bounds are what the Standardizer was fitted through -- read them
# from an instance rather than assuming PenSimEnvGym's defaults (they differ by 2x).
_pcd = PeniControlData(dataset_folder=pdata.default_dataset_folder(), normalize=True)
PCD_MAX_OBS = np.array(np.asarray(_pcd.max_observations).tolist(), dtype=np.float64)
PCD_MIN_OBS = np.array(np.asarray(_pcd.min_observations).tolist(), dtype=np.float64)
PCD_MAX_ACT = np.array(np.asarray(_pcd.max_actions).tolist(), dtype=np.float64)
PCD_MIN_ACT = np.array(np.asarray(_pcd.min_actions).tolist(), dtype=np.float64)
print(f"[units] PeniControlData obs bounds: time [{PCD_MIN_OBS[0]:.2f}, {PCD_MAX_OBS[0]:.2f}] h")
print(f"[units] PeniControlData act bounds: {np.round(PCD_MIN_ACT,2)} .. {np.round(PCD_MAX_ACT,2)}")

ck = torch.load(args.policy, map_location=device, weights_only=False)
STD_OBS_MU = np.array(np.asarray(ck["std_obs_mu"]).tolist(), dtype=np.float64)   # (8,)
STD_OBS_SD = np.array(np.asarray(ck["std_obs_sd"]).tolist(), dtype=np.float64)
STD_ACT_MU = np.array(np.asarray(ck["std_act_mu"]).tolist(), dtype=np.float64)   # (6,)
STD_ACT_SD = np.array(np.asarray(ck["std_act_sd"]).tolist(), dtype=np.float64)


def obs_phys_to_z(o_phys):
    """(9,) physical observation -> (8,) z-scored policy input (time dropped)."""
    o_n = 2.0 * (o_phys - PCD_MIN_OBS) / (PCD_MAX_OBS - PCD_MIN_OBS) - 1.0
    o_n8 = np.delete(o_n, pdata.TIME_INDEX)
    return (o_n8 - STD_OBS_MU) / STD_OBS_SD


def act_z_to_phys(a_z):
    """(6,) z-scored policy output -> (6,) physical action."""
    a_n = a_z * STD_ACT_SD + STD_ACT_MU
    return (a_n + 1.0) / 2.0 * (PCD_MAX_ACT - PCD_MIN_ACT) + PCD_MIN_ACT


# ================================================================== policy ====
# Rebuild whatever architecture the checkpoint records. Older checkpoints predate
# policy_meta and are always Sum_of_gaussians, so fall back to that.
_meta = ck.get("policy_meta")
if _meta is None:
    centers_init = np.array(np.asarray(ck["centers_init"]).tolist(), dtype=np.float64)
    n_basis, state_dim = centers_init.shape
    _ls = ck.get("lengthscales_init")
    _meta = {"kind": "rbf", "state_dim": state_dim, "input_dim": pdata.ACT_DIM,
             "u_max": U_MAX_FLAT, "num_basis": n_basis,
             "centers_init": centers_init.tolist(),
             "lengthscales_init": (np.asarray(_ls).tolist() if _ls is not None
                                   else np.ones(state_dim).tolist())}
policy = rebuild_policy(_meta, dtype=dtype, device=device)
policy.load_state_dict(ck["policy_state_dict"])
policy.eval()
state_dim = _meta["state_dim"]
_hist = ck.get("hist", [])
print(f"[policy] {args.policy}")
print(f"[policy] kind={_meta['kind']} state_dim={state_dim} u_max={U_MAX_FLAT}"
      + (f" | final training W2 = {_hist[-1]:.4f}" if _hist else ""))

# +/-10% band around the recipe setpoint, in PHYSICAL units
SETPOINT_PHYS = (STD_ACT_MU + 1.0) / 2.0 * (PCD_MAX_ACT - PCD_MIN_ACT) + PCD_MIN_ACT
LIM_LO = SETPOINT_PHYS * (1.0 - ACTION_LIMIT_FRAC)
LIM_HI = SETPOINT_PHYS * (1.0 + ACTION_LIMIT_FRAC)
print(f"[action] recipe setpoint (physical): {np.round(SETPOINT_PHYS, 3)}")
_modes = []
if args.clip10: _modes.append("static +/-10% of dataset mean")
if args.cliprecipe: _modes.append("time-varying +/-10% of recipe profile")
print(f"[action] clipping: {' + '.join(_modes) if _modes else 'OFF (env bounds only)'}")


# ===================================================================== env ====
recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
               FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
               FG: Recipe(FG_DEFAULT_PROFILE, FG),
               PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
               DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
               WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
               PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}
# the seed goes in the constructor; some smpl releases also expose .seed(), this one
# may not -- hence the hasattr guard
def make_env(seed):
    """A FRESH env per episode.

    Reusing a single instance leaks state across episodes: every SECOND rollout
    aborted at step 1 with reward = error_reward (-100), in a perfectly alternating
    pattern, even though the reset observations were all in range (pH 6.44, T 297.7,
    Wt 62k -- indistinguishable from the successful episodes). A completed episode
    evidently leaves internal state that reset() does not clear.
    """
    e = PenSimEnvGym(recipe_combo=RecipeCombo(recipe_dict=recipe_dict),
                     normalize=False, random_seed=seed)
    if hasattr(e, "seed"):
        e.seed(seed)
    return e


env = make_env(args.seed)          # module-level instance: only used for action bounds
ENV_MIN_ACT_ = np.array(np.asarray(env.min_actions).tolist(), dtype=np.float64)
ENV_MAX_ACT_ = np.array(np.asarray(env.max_actions).tolist(), dtype=np.float64)

# time-varying recipe band, in PHYSICAL units (the env is run with normalize=False).
# The action vector order is [discharge, sugar, soilbean, aeration, backpressure,
# waterinj]; PAA is in recipe_dict for the simulator but is not a policy action.
RECIPE_BOUNDS = None
if args.cliprecipe:
    _keys = [DISCHARGE, FS, FOIL, FG, PRES, WATER]
    _rc = RecipeCombo(recipe_dict={k: recipe_dict[k] for k in _keys})
    # identity standardizer: we want the band in PHYSICAL units here, not z-scored
    # RecipeBounds maps physical -> smpl min-max -> z. Passing mu=0, sd=1 leaves the
    # result in smpl-normalised units, so it is converted back below.
    _ident_mu = np.zeros(pdata.ACT_DIM)
    _ident_sd = np.ones(pdata.ACT_DIM)
    RECIPE_BOUNDS = RecipeBounds(_rc, _keys, ENV_MIN_ACT_, ENV_MAX_ACT_,
                                 _ident_mu, _ident_sd,
                                 frac=args.recipe_frac,
                                 floor_frac=args.recipe_floor,
                                 smooth_h=args.recipe_smooth)
    print(f"[action] recipe band: +/-{100*args.recipe_frac:.0f}% of profile, "
          f"floor {100*args.recipe_floor:.0f}% of span, smoothed +/-{args.recipe_smooth} h")
    for _t in (10.0, 50.0, 110.0, 200.0):
        _lo, _hi = RECIPE_BOUNDS.at(_t)
        print(f"    t={_t:6.1f} h  discharge [{_lo[0]:9.2f}, {_hi[0]:9.2f}]   "
              f"sugar [{_lo[1]:7.2f}, {_hi[1]:7.2f}]")
ENV_MIN_ACT = np.array(np.asarray(env.min_actions).tolist(), dtype=np.float64)
ENV_MAX_ACT = np.array(np.asarray(env.max_actions).tolist(), dtype=np.float64)


# ================================================================ rollout ====
def run_episode(ep, seed):
    env = make_env(seed)                       # fresh env -- no cross-episode leakage
    o = np.array(np.asarray(env.reset()).reshape(-1).tolist(), dtype=np.float64)
    rows, total_yield, t = [], 0.0, 0
    while True:
        z = obs_phys_to_z(o)
        with torch.no_grad():
            _a = policy(states=torch.tensor(z[None, :], dtype=dtype, device=device),
                        t=t, p_dropout=args.p_dropout)
        a_z = _to_np(_a)
        a_phys = act_z_to_phys(a_z)
        if args.clip10:                        # static band (dataset mean)
            a_phys = np.clip(a_phys, LIM_LO, LIM_HI)
        if RECIPE_BOUNDS is not None:          # time-varying band (recipe profile)
            t_h = float(o[pdata.TIME_INDEX])   # current time, physical hours
            _lo_n, _hi_n = RECIPE_BOUNDS.at(t_h)
            # RecipeBounds returns smpl-normalised units; convert back to physical
            _span = ENV_MAX_ACT_ - ENV_MIN_ACT_
            lo_p = (np.asarray(_lo_n) + 1.0) / 2.0 * _span + ENV_MIN_ACT_
            hi_p = (np.asarray(_hi_n) + 1.0) / 2.0 * _span + ENV_MIN_ACT_
            a_phys = np.clip(a_phys, lo_p, hi_p)
        a_phys = np.clip(a_phys, ENV_MIN_ACT, ENV_MAX_ACT)        # env bounds always

        step = env.step(a_phys)
        o_next = np.array(np.asarray(step[0]).reshape(-1).tolist(), dtype=np.float64)
        reward, done = float(step[1]), bool(step[2])

        # PenSim returns error_reward (-100) INSTEAD of a yield at termination. That
        # row carries no yield information, so it is neither summed nor written --
        # otherwise every episode total is 100 low and the shipped baselines (which
        # have no such row) look artificially better.
        is_error_row = reward <= ERROR_REWARD + 1e-9
        if not is_error_row:
            total_yield += reward
            # CSV row: time, 6 actions, 8 observations (time dropped), yield
            rows.append([o_next[pdata.TIME_INDEX]] + list(a_phys) +
                        list(np.delete(o_next, pdata.TIME_INDEX)) + [reward])
            n_err = 0
        else:
            n_err = 1

        o, t = o_next, t + 1
        if done or (args.max_steps and t >= args.max_steps):
            if is_error_row:
                print(f"    dropped terminal error_reward row at t={o_next[pdata.TIME_INDEX]:.1f} h",
                      flush=True)
            break

    rows = np.array(rows, dtype=np.float64)
    # a diverged simulator writes non-finite observations, which then poison any model
    # trained on the file -- discard the whole episode instead
    if rows.size and not np.isfinite(rows).all():
        print(f"    non-finite values ({int((~np.isfinite(rows)).sum())}) -- "
              f"simulator diverged; episode discarded", flush=True)
        return rows[:0], 0.0, 0
    return rows, total_yield, len(rows)


print(f"\ncollecting {args.n} episodes -> {args.out}", flush=True)
summary = []
MIN_STEPS = 100        # anything shorter is an aborted reset, not a real rollout
for ep in range(args.n):
    for attempt in range(4):
        rows, y, n_steps = run_episode(ep, args.seed + 1000 * attempt + ep)
        if n_steps >= MIN_STEPS or args.max_steps:
            break
        print(f"  episode {ep}: aborted after {n_steps} step(s) (yield={y:.1f}) "
              f"-- retrying with a new seed", flush=True)
    if n_steps < MIN_STEPS and not args.max_steps:
        print(f"  episode {ep}: FAILED after 4 attempts, not written", flush=True)
        continue
    path = os.path.join(args.out, f"{args.tag}_batch_{ep}.csv")
    np.savetxt(path, rows, delimiter=",", header=",".join(CSV_COLUMNS),
               comments="", fmt="%.10g")
    summary.append((ep, n_steps, y, rows[-1, 0]))
    print(f"  episode {ep}: {n_steps} steps, final t={rows[-1,0]:.1f} h, "
          f"total yield={y:.4f}  ->  {os.path.basename(path)}", flush=True)

ys = np.array([s[2] for s in summary])
ns = np.array([s[1] for s in summary], dtype=np.float64)
print("\nsummary:")
if len(ys):
    print(f"  episodes written: {len(ys)}/{args.n}")
    print(f"  total yield  mean={ys.mean():.4f}  min={ys.min():.4f}  max={ys.max():.4f}")
    per_step = ys / np.maximum(ns, 1)
    print(f"  yield/step   mean={per_step.mean():.4f}  min={per_step.min():.4f}  "
          f"max={per_step.max():.4f}   (SMPL paper baseline: 3.3071)")
else:
    print("  no episodes completed")
print(f"  files written to {args.out}")
print("\nNOTE: this folder is NOT the one the trainer reads "
      f"({pdata.default_dataset_folder()}).\n"
      "      Copy the CSVs there (or repoint default_dataset_folder) to retrain on them.")
