#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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

U_MAX_FLAT = 3.0
ERROR_REWARD = -100.0
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
_p.add_argument("-discharge_csv", type=str, default=None,
                help="replay the discharge column of this CSV, looked up by time. "
                     "Prefer this to -recipe_discharge: the raw profile is a "
                     "zero-order hold that discharges for 28 h straight and empties "
                     "the vessel, whereas the recorded gpei actions are 2-hour pulses.")
_p.add_argument("-recipe_discharge", action="store_true",
                help="ABLATION: take discharge from the recipe profile at the current "
                     "time instead of from the policy. Isolates how much of the yield "
                     "gap is the valve channel.")
_p.add_argument("-fix_discharge", type=float, default=None,
                help="hold discharge at this PHYSICAL value below -fix_until hours")
_p.add_argument("-fix_until", type=float, default=1e9,
                help="apply -fix_discharge only below this batch time")
_p.add_argument("-phase_policies", type=str, default=None,
                help="PREFIX for three per-phase policies <prefix>_ph{0,1,2}.pt, "
                     "selected by batch time. Overrides -policy.")
_p.add_argument("-valve_policy", type=str, default=None,
                help="P_C checkpoint: a separate discharge policy. -policy then "
                     "supplies channels 1..5 only.")
_p.add_argument("-max_steps", type=int, default=None, help="cap steps (debug)")
args = _p.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)


def _to_np(t):
    """torch tensor -> numpy, via .tolist() to dodge the duplicate-numpy ABI clash."""
    return np.array(t.detach().reshape(-1).tolist(), dtype=np.float64)


_pcd = PeniControlData(dataset_folder=pdata.default_dataset_folder(), normalize=True)
PCD_MAX_OBS = np.array(np.asarray(_pcd.max_observations).tolist(), dtype=np.float64)
PCD_MIN_OBS = np.array(np.asarray(_pcd.min_observations).tolist(), dtype=np.float64)
PCD_MAX_ACT = np.array(np.asarray(_pcd.max_actions).tolist(), dtype=np.float64)
PCD_MIN_ACT = np.array(np.asarray(_pcd.min_actions).tolist(), dtype=np.float64)
print(f"[units] PeniControlData obs bounds: time [{PCD_MIN_OBS[0]:.2f}, {PCD_MAX_OBS[0]:.2f}] h")
print(f"[units] PeniControlData act bounds: {np.round(PCD_MIN_ACT,2)} .. {np.round(PCD_MAX_ACT,2)}")

if args.phase_policies and not os.path.exists(args.policy):
    args.policy = f"{args.phase_policies}_ph0.pt"
ck = torch.load(args.policy, map_location=device, weights_only=False)
STD_OBS_MU = np.array(np.asarray(ck["std_obs_mu"]).tolist(), dtype=np.float64)
STD_OBS_SD = np.array(np.asarray(ck["std_obs_sd"]).tolist(), dtype=np.float64)
STD_ACT_MU = np.array(np.asarray(ck["std_act_mu"]).tolist(), dtype=np.float64)
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
_sd = ck["policy_state_dict"]
if any(k.startswith("base.") for k in _sd):
    _sd = {k[5:]: v for k, v in _sd.items() if k.startswith("base.")}
    print("[policy] stripped 'base.' prefix from a wrapped checkpoint")
policy.load_state_dict(_sd)
policy.eval()
state_dim = _meta["state_dim"]
PHASE_POLS = None
if args.phase_policies:
    PHASE_POLS = {}
    for _p in (0, 1, 2):
        _pp = f"{args.phase_policies}_ph{_p}.pt"
        _pc = torch.load(_pp, map_location=device, weights_only=False)
        _pol = rebuild_policy(_pc["policy_meta"], dtype=dtype, device=device)
        _psd = _pc["policy_state_dict"]
        if any(k.startswith("base.") for k in _psd):
            _psd = {k[5:]: v for k, v in _psd.items() if k.startswith("base.")}
        _pol.load_state_dict(_psd)
        _pol.eval()
        _lo, _hi = pdata.PHASES[_p]
        PHASE_POLS[_p] = _pol
        print(f"[phase] {_p}: {os.path.basename(_pp)}  t in [{_lo:g}, "
              f"{'inf' if _hi > 1e8 else f'{_hi:g}'}) h  "
              f"trained on {_pc.get('n_windows', '?')} windows")
    print(f"[phase] boundaries from PENSIM_PHASE_BOUNDS="
          f"{os.environ.get('PENSIM_PHASE_BOUNDS', '35.0,51.0')} -- these MUST match "
          f"the ones the policies were trained with")


def _policy_at(t_h):
    if PHASE_POLS is None:
        return policy
    for _p in (0, 1, 2):
        _lo, _hi = pdata.PHASES[_p]
        if _lo <= t_h < _hi:
            return PHASE_POLS[_p]
    return PHASE_POLS[2]


DISCH_TRACE = None
if args.discharge_csv:
    import csv as _csv
    _h = [c.strip() for c in next(_csv.reader(open(args.discharge_csv)))]
    _d = np.genfromtxt(args.discharge_csv, delimiter=",", skip_header=1)
    _ti, _di = _h.index("Time Step"), _h.index("Discharge rate")
    DISCH_TRACE = (np.asarray(_d[:, _ti], dtype=np.float64),
                   np.asarray(_d[:, _di], dtype=np.float64))
    _hi = DISCH_TRACE[1] > 0.5 * DISCH_TRACE[1].max()
    print(f"[ablation] discharge replayed from {os.path.basename(args.discharge_csv)}: "
          f"{len(_d)} steps, duty={100*_hi.mean():.1f}%, peak={DISCH_TRACE[1].max():.0f}, "
          f"volume={(DISCH_TRACE[1]*0.2).sum():.0f}")

DISCH_RECIPE = None
if args.recipe_discharge:
    DISCH_RECIPE = Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE)
    print("[ablation] discharge taken from DISCHARGE_DEFAULT_PROFILE, not the policy")
    for _t in (10, 50, 90, 102, 150, 200):
        print(f"    t={_t:4d} h -> {DISCH_RECIPE.get_value_at(_t):8.1f} L/h")
    if args.valve_policy:
        print("[ablation] WARNING: -recipe_discharge overrides -valve_policy")

VALVE = None
if args.valve_policy:
    _ckv = torch.load(args.valve_policy, map_location=device, weights_only=False)
    _pc = rebuild_policy(_ckv["policy_meta"], dtype=dtype, device=device)
    _pc.load_state_dict(_ckv["policy_state_dict"])
    _pc.eval()
    VALVE = {"pc": _pc,
             "z_off": float(_ckv["z_off"]), "z_max": float(_ckv["z_max"]),
             "gain": float(_ckv.get("gate_gain", 2.0)),
             "thresh": float(_ckv.get("open_thresh_z",
                                      _ckv["z_off"] + 0.5 * (_ckv["z_max"]
                                                             - _ckv["z_off"]))),
             "hours": None}
    print(f"[valve] P_C: {os.path.basename(args.valve_policy)}  "
          f"off={VALVE['z_off']:.3f} z  max={VALVE['z_max']:.3f} z  "
          f"open_thresh={VALVE['thresh']:.3f} z")
    print(f"[valve] trained against P_B={os.path.basename(str(_ckv.get('policy_actions')))}"
          f"  target duty={_ckv.get('target_duty')}")
    if args.clip10:
        print("[valve] WARNING: -clip10 with -valve_policy crushes the gate's "
              "{0, ~4000} into the static band and undoes the mechanism")

_hist = ck.get("hist", [])
print(f"[policy] {args.policy}")
print(f"[policy] kind={_meta['kind']} state_dim={state_dim} u_max={U_MAX_FLAT}"
      + (f" | final training W2 = {_hist[-1]:.4f}" if _hist else ""))

SETPOINT_PHYS = (STD_ACT_MU + 1.0) / 2.0 * (PCD_MAX_ACT - PCD_MIN_ACT) + PCD_MIN_ACT
LIM_LO = SETPOINT_PHYS * (1.0 - ACTION_LIMIT_FRAC)
LIM_HI = SETPOINT_PHYS * (1.0 + ACTION_LIMIT_FRAC)
print(f"[action] recipe setpoint (physical): {np.round(SETPOINT_PHYS, 3)}")
_modes = []
if args.clip10: _modes.append("static +/-10% of dataset mean")
if args.cliprecipe: _modes.append("time-varying +/-10% of recipe profile")
print(f"[action] clipping: {' + '.join(_modes) if _modes else 'OFF (env bounds only)'}")


recipe_dict = {FS: Recipe(FS_DEFAULT_PROFILE, FS),
               FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
               FG: Recipe(FG_DEFAULT_PROFILE, FG),
               PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
               DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
               WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
               PAA: Recipe(PAA_DEFAULT_PROFILE, PAA)}
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


env = make_env(args.seed)
ENV_MIN_ACT_ = np.array(np.asarray(env.min_actions).tolist(), dtype=np.float64)
ENV_MAX_ACT_ = np.array(np.asarray(env.max_actions).tolist(), dtype=np.float64)

RECIPE_BOUNDS = None
if args.cliprecipe:
    _keys = [DISCHARGE, FS, FOIL, FG, PRES, WATER]
    _rc = RecipeCombo(recipe_dict={k: recipe_dict[k] for k in _keys})
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


def run_episode(ep, seed):
    env = make_env(seed)
    if VALVE is not None:
        VALVE["hours"] = 0.0
    o = np.array(np.asarray(env.reset()).reshape(-1).tolist(), dtype=np.float64)
    rows, total_yield, t = [], 0.0, 0
    while True:
        z = obs_phys_to_z(o)
        with torch.no_grad():
            _s = torch.tensor(z[None, :], dtype=dtype, device=device)
            _a = _policy_at(float(o[pdata.TIME_INDEX]))(
                states=_s, t=t, p_dropout=args.p_dropout)
            if VALVE is not None:
                _sa = torch.cat([_s, torch.tensor([[VALVE["hours"]]], dtype=dtype,
                                                  device=device)], dim=1)
                _u = VALVE["pc"](states=_sa, t=t, p_dropout=args.p_dropout)
                _pg = torch.sigmoid(VALVE["gain"] * _u[:, 0])
                _gate = torch.bernoulli(_pg)
                _lvl = VALVE["z_off"] + torch.sigmoid(_u[:, 1]) * (VALVE["z_max"]
                                                                  - VALVE["z_off"])
                _a = _a.clone()
                _a[:, 0] = VALVE["z_off"] + _gate * (_lvl - VALVE["z_off"])
                VALVE["hours"] = ((VALVE["hours"] + 0.2)
                                  if float(_a[0, 0]) > VALVE["thresh"] else 0.0)
        a_z = _to_np(_a)
        a_phys = act_z_to_phys(a_z)
        if args.fix_discharge is not None \
                and float(o[pdata.TIME_INDEX]) < args.fix_until:
            a_phys[0] = float(args.fix_discharge)
        if DISCH_TRACE is not None:
            _tt, _dd = DISCH_TRACE
            _k = int(np.searchsorted(_tt, float(o[pdata.TIME_INDEX]), side="right") - 1)
            a_phys[0] = float(_dd[min(max(_k, 0), len(_dd) - 1)])
        if DISCH_RECIPE is not None:
            a_phys[0] = float(DISCH_RECIPE.get_value_at(float(o[pdata.TIME_INDEX])))
        if args.clip10:
            a_phys = np.clip(a_phys, LIM_LO, LIM_HI)
        if RECIPE_BOUNDS is not None:
            t_h = float(o[pdata.TIME_INDEX])
            _lo_n, _hi_n = RECIPE_BOUNDS.at(t_h)
            _span = ENV_MAX_ACT_ - ENV_MIN_ACT_
            lo_p = (np.asarray(_lo_n) + 1.0) / 2.0 * _span + ENV_MIN_ACT_
            hi_p = (np.asarray(_hi_n) + 1.0) / 2.0 * _span + ENV_MIN_ACT_
            a_phys = np.clip(a_phys, lo_p, hi_p)
        a_phys = np.clip(a_phys, ENV_MIN_ACT, ENV_MAX_ACT)

        step = env.step(a_phys)
        o_next = np.array(np.asarray(step[0]).reshape(-1).tolist(), dtype=np.float64)
        reward, done = float(step[1]), bool(step[2])

        is_error_row = reward <= ERROR_REWARD + 1e-9
        if not is_error_row:
            total_yield += reward
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
    if VALVE is not None and len(rows):
        _d = rows[:, 1]
        _hi = _d > 0.5 * max(_d.max(), 1e-9)
        _tr = np.diff(_hi.astype(int))
        print(f"    valve: duty={100*_hi.mean():5.1f}%  opens={int((_tr==1).sum())}  "
              f"peak={_d.max():7.1f}  total volume={(_d*0.2).sum():9.0f}"
              f"   (gpei: duty 5.2%, 6 opens, peak ~4000, volume ~46000)", flush=True)
    if rows.size and not np.isfinite(rows).all():
        print(f"    non-finite values ({int((~np.isfinite(rows)).sum())}) -- "
              f"simulator diverged; episode discarded", flush=True)
        return rows[:0], 0.0, 0
    return rows, total_yield, len(rows)


print(f"\ncollecting {args.n} episodes -> {args.out}", flush=True)
summary = []
MIN_STEPS = 100
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
