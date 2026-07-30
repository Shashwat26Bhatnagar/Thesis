#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_learning/pensim_dataset.py

Adapter between the SMPL PenSim environment and the MC-PILCO RBF world model.

PenSimEnvGym is a Gym env (reset/step), not an ODE f_sim, so MC_PILCO.reinforce()
is bypassed: the world model is fitted here (train_rbf_pensim.py) and then used as a
frozen one-step predictor for policy optimization (cdil_policy_optimization.py).

Raw observation (9) from get_observation_data_reformed:
    [time, pH, T, Fa, Fb, Fc, Fh, Wt, DO2]
Action (6): discharge, sugar feed, soil bean feed, aeration, back pressure, water inj.
NOTE: yield/reward is a SEPARATE return value from env.step(), NOT an observation
channel. Rewards are ignored here (state model only).

TWO PREPROCESSING DECISIONS:

1. DROP THE TIME CHANNEL (obs index 0) from the MODEL INPUT. PenSimEnvGym's own
   docstring says "Time is not in our observation_space. We make the env time unaware
   and MDP", yet get_observation_data_reformed returns t as element 0. A time-indexed
   input breaks the Markov assumption the dynamics model relies on: the GP would learn
   "state at t" rather than "response to (state, action)".
   The time VALUES are still returned separately via load_offline(return_time=True),
   so callers can segment the dataset by fermentation phase without putting t back
   into the GP input.

2. STANDARDIZE (z-score) per variable on the DATASET's own mean/std, on top of
   smpl's min-max scaling. smpl scales each channel against its full PHYSICAL
   ENVELOPE, not against the data: Temperature is controlled inside a ~2 K band
   within a ~125 K envelope, so it occupies ~1.6% of its axis while time occupies
   ~83% -- a ~50x disparity. RBF uses a single squared distance across all input
   dimensions, so without z-scoring the ARD lengthscales must absorb that disparity
   and the near-constant channels are effectively invisible to the kernel.

=== UNITS WARNING ===
load_offline applies TWO transforms in sequence: smpl min-max, then z-score. The
Standardizer statistics are therefore fitted in SMPL-NORMALIZED space, not physical
space. std_obs.inverse_transform() returns you to smpl-normalized units, NOT to
pH / Kelvin / kg. Use to_physical() / from_physical() below for the full round trip.
The time array from return_time=True IS converted back to physical HOURS.
"""
import os
import json
import numpy as np

# ----------------------------------------------------------------- constants ---
OBS_DIM_RAW = 9          # what the env / CSVs return
TIME_INDEX = 0           # position of the time channel in the raw observation
OBS_DIM = 8              # after dropping time
ACT_DIM = 6

OBS_NAMES = ["pH", "T", "Fa", "Fb", "Fc", "Fh", "Wt", "DO2"]
ACT_NAMES = ["discharge", "sugar", "soilbean", "aeration", "backpressure", "waterinj"]

# PenSimEnvGym defaults -- the physical envelope smpl min-max scales against.
_MAX_OBS_RAW = [552.0, 16.10523, 725.6828, 13.717274, 540.0, 3600.0002, 1892.07874,
                253840.11, 47.898834]
_MIN_OBS_RAW = [0.0, 0.0, 118.98977, 0.0, 0.0, 0.0, 0.0, 25003.258, 0.0]
# time bounds, kept separately so the time column can be un-normalised to hours
MAX_TIME = float(_MAX_OBS_RAW[TIME_INDEX])       # 552.0 h
MIN_TIME = float(_MIN_OBS_RAW[TIME_INDEX])       # 0.0 h
MAX_OBS = np.array(_MAX_OBS_RAW[1:], dtype=np.float64)      # (8,) time dropped
MIN_OBS = np.array(_MIN_OBS_RAW[1:], dtype=np.float64)
MAX_ACT = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0], dtype=np.float64)
MIN_ACT = np.array([0.0, 7.0, 21.0, 29.0, 0.5, 0.0], dtype=np.float64)

# fermentation phases for piecewise world models (hours). -1 = use all data.
PHASES = {
    0: (0.0, 35.0),        # lag / early growth
    1: (35.0, 51.0),       # transition
    2: (51.0, 1e9),        # production
    -1: (0.0, 1e9),        # everything
}


def phase_tag(phase):
    """Filename tag for a phase id."""
    return "all" if phase == -1 else f"phase{phase}"

def default_dataset_folder():
    """CSV batches. PENSIM_DATA_DIR overrides everything -- the Dyna loop uses it to
    give each variant (bounded / unbounded) its own accumulating dataset, so the two
    chains never train on each other's trajectories."""
    env_dir = os.environ.get("PENSIM_DATA_DIR")
    if env_dir:
        n = len([f for f in os.listdir(env_dir) if f.endswith(".csv")]) \
            if os.path.isdir(env_dir) else 0
        print(f"[pensim] dataset folder (PENSIM_DATA_DIR): {env_dir}  ({n} CSVs)")
        return env_dir
    folder = "/home/s2892016/Thesis/deps/smpl/smpl/configdata/pensimenv"
    n = len([f for f in os.listdir(folder) if f.endswith(".csv")]) \
        if os.path.isdir(folder) else 0
    print(f"[pensim] dataset folder: {folder}  ({n} CSVs)")
    return folder

def drop_time(obs):
    """Remove the time channel from a (N, 9) observation array -> (N, 8).
    Idempotent: arrays already at OBS_DIM are returned unchanged."""
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] == OBS_DIM_RAW:
        return np.delete(obs, TIME_INDEX, axis=1)
    return obs


def extract_time_hours(obs_raw, smpl_normalized=True):
    """Pull column 0 out of a RAW (N, 9) observation array as PHYSICAL HOURS.

    smpl's normalize maps physical -> [-1, 1], so it is inverted here. Returns None
    if the array has already had the time channel dropped."""
    obs_raw = np.asarray(obs_raw, dtype=np.float64)
    if obs_raw.ndim == 1:
        obs_raw = obs_raw.reshape(1, -1)
    if obs_raw.shape[1] != OBS_DIM_RAW:
        return None
    t = obs_raw[:, TIME_INDEX].copy()
    if smpl_normalized:
        t = (t + 1.0) / 2.0 * (MAX_TIME - MIN_TIME) + MIN_TIME
    return t


def phase_mask(t_hours, phase):
    """Boolean mask selecting the transitions belonging to `phase`."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {sorted(PHASES)}")
    lo, hi = PHASES[phase]
    t = np.asarray(t_hours, dtype=np.float64)
    return (t >= lo) & (t < hi)


def filter_by_phase(obs, act, nobs, t_hours, phase, verbose=True):
    """Restrict a loaded dataset to one fermentation phase.

    Apply this AFTER load_offline so the Standardizer is fitted on the FULL dataset:
    all phase models then share one z-space and their outputs stay comparable.
    """
    m = phase_mask(t_hours, phase)
    lo, hi = PHASES[phase]
    if verbose:
        hi_s = "inf" if hi > 1e8 else f"{hi:g}"
        print(f"[pensim] phase {phase}: t in [{lo:g}, {hi_s}) h  ->  "
              f"{int(m.sum())} of {len(m)} transitions "
              f"({100.0*m.mean():.1f}%)")
        if m.sum() == 0:
            print("[pensim] WARNING: phase is EMPTY")
    return obs[m], act[m], nobs[m], t_hours[m]

# =====================================================================================
# APPEND THIS FUNCTION TO THE END OF  model_learning/pensim_dataset.py
# (numpy is already imported there as np)
# =====================================================================================


def select_pivoted_cholesky(X, m, lengthscales=None, tol=1e-10, verbose=True):
    """Greedy pivoted-Cholesky subset selection.

    Picks the m points that contribute most to the RANK of the RBF kernel matrix,
    skipping near-duplicates.

    WHY: uniform 'stride' subsampling keeps redundant rows. In a narrow time window
    (phase 1 spans 16 h of a 230 h batch) the process barely moves, so hundreds of
    rows are near-identical in the 14-D input space. Then K(x_i,x_j) ~ K(x_i,x_i),
    the kernel matrix is numerically rank-deficient, and torch.cholesky fails with
        "the leading minor of order 229 is not positive-definite".
    Pivoted Cholesky is the standard remedy: it selects a well-conditioned subset.

    The residual trace printed at the end is diagnostic in its own right -- if it
    drops below tol after k << m points, the data genuinely contains only ~k points'
    worth of independent information, and no kernel choice or jitter changes that.

    Parameters
    ----------
    X : (N, d) array -- the GP inputs, e.g. np.hstack([obs, act])
    m : int         -- maximum number of points to select
    lengthscales : (d,) or None -- per-dimension scaling before distances (default 1)
    tol : float     -- stop when the largest residual variance falls below this

    Returns
    -------
    idx : (k,) int array of selected row indices, k <= m
    """
    X = np.asarray(X, dtype=np.float64)
    N, d = X.shape
    ls = np.ones(d) if lengthscales is None else np.asarray(lengthscales, dtype=np.float64)
    Xs = X / ls

    m = int(min(m, N))
    diag = np.ones(N)                      # RBF kernel diagonal is 1 everywhere
    idx = []
    L = np.zeros((m, N))

    for k in range(m):
        j = int(np.argmax(diag))
        if diag[j] < tol:
            if verbose:
                print(f"[pivchol] residual below tol at k={k}: the data has only "
                      f"~{k} independent directions")
            break
        idx.append(j)
        d2 = ((Xs - Xs[j]) ** 2).sum(axis=1)
        row = np.exp(-0.5 * d2)                       # k(x_j, .)
        if k:
            row = row - L[:k, :].T @ L[:k, j]
        row = row / np.sqrt(max(diag[j], 1e-300))
        L[k] = row
        diag = np.maximum(diag - row ** 2, 0.0)

    idx = np.array(sorted(idx), dtype=int)
    if verbose:
        print(f"[pivchol] selected {len(idx)}/{N} points  "
              f"(residual trace {diag.sum():.3e})")
    return idx


# ------------------------------------------------------------- standardizer ---
class Standardizer:
    """Per-variable z-scoring fitted on the dataset.

    Statistics are kept in `self.stats`, keyed by variable name:
        {"pH": {"index":0, "mean":..., "std":..., "min":..., "max":..., "live":True}, ...}

    Channels with std below `eps` are constant across the dataset; they are flagged
    live=False and left unscaled so the transform stays finite.

    NOTE: these statistics are in SMPL-NORMALIZED space (see the module docstring).
    """

    def __init__(self, X, names=None, name="", eps=1e-10):
        X = np.asarray(X, dtype=np.float64)
        self.name = name
        self.eps = eps
        self.mu = X.mean(0)
        self.sd = X.std(0)
        self.min = X.min(0)
        self.max = X.max(0)
        self.live = self.sd > eps
        self.sd_safe = np.where(self.live, self.sd, 1.0)
        self.names = list(names) if names is not None else [f"{name}{i}" for i in range(X.shape[1])]

        self.stats = {
            nm: {"index": int(i), "mean": float(self.mu[i]), "std": float(self.sd[i]),
                 "min": float(self.min[i]), "max": float(self.max[i]),
                 "live": bool(self.live[i])}
            for i, nm in enumerate(self.names)
        }

    # -- states / actions: mean shift + scale --
    def transform(self, X):
        return (np.asarray(X, dtype=np.float64) - self.mu) / self.sd_safe

    def inverse_transform(self, Xs):
        """-> SMPL-NORMALIZED units (not physical). See to_physical()."""
        return np.asarray(Xs, dtype=np.float64) * self.sd_safe + self.mu

    # -- deltas: differences carry no mean offset, so scale only --
    def transform_delta(self, dX):
        return np.asarray(dX, dtype=np.float64) / self.sd_safe

    def inverse_transform_delta(self, dXs):
        return np.asarray(dXs, dtype=np.float64) * self.sd_safe

    # -- predictive variance scales with std^2 --
    def inverse_transform_var(self, Vs):
        return np.asarray(Vs, dtype=np.float64) * (self.sd_safe ** 2)

    def dead_dims(self):
        return np.where(~self.live)[0]

    def report(self):
        print(f"[std:{self.name}]")
        for nm, s in self.stats.items():
            flag = "" if s["live"] else "   <-- ZERO VARIANCE"
            print(f"    {s['index']:2d} {nm:14s} mean={s['mean']:12.5f}  "
                  f"std={s['std']:11.6f}{flag}")

    def to_dict(self):
        return {"name": self.name, "names": self.names,
                "mu": self.mu.tolist(), "sd": self.sd.tolist(), "stats": self.stats}


# --------------------------------------------- full physical <-> model units ---
# smpl's normalize maps physical -> [-1, 1] via  2*(x - min)/(max - min) - 1.
def _smpl_to_physical(x_norm, lo, hi):
    return (np.asarray(x_norm, dtype=np.float64) + 1.0) / 2.0 * (hi - lo) + lo


def _physical_to_smpl(x_phys, lo, hi):
    return 2.0 * (np.asarray(x_phys, dtype=np.float64) - lo) / (hi - lo) - 1.0


def to_physical(z, std, kind="obs"):
    """Model units (z-scored on smpl-normalized) -> PHYSICAL units."""
    lo, hi = (MIN_OBS, MAX_OBS) if kind == "obs" else (MIN_ACT, MAX_ACT)
    return _smpl_to_physical(std.inverse_transform(z), lo, hi)


def from_physical(x_phys, std, kind="obs"):
    """PHYSICAL units -> model units (z-scored on smpl-normalized)."""
    lo, hi = (MIN_OBS, MAX_OBS) if kind == "obs" else (MIN_ACT, MAX_ACT)
    return std.transform(_physical_to_smpl(x_phys, lo, hi))


def save_stats(path, std_obs, std_act, extra=None):
    """Persist fitted statistics as JSON (plain lists -- no numpy pickling)."""
    blob = {"obs": std_obs.to_dict() if std_obs else None,
            "act": std_act.to_dict() if std_act else None,
            "extra": extra or {}}
    with open(path + ".json", "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[pensim] statistics saved -> {path}.json")


def load_stats(path):
    with open(path + ".json") as f:
        return json.load(f)


# ------------------------------------------------------------------- loading ---
def load_offline(dataset_folder=None, smpl_normalize=True, max_transitions=None,
                 drop_time_channel=True, standardize=True, verbose=True,
                 return_time=False):
    """Load PeniControlData (reads every CSV batch in the folder).

    Parameters
    ----------
    return_time : bool
        If True, additionally return the per-transition time in PHYSICAL HOURS,
        taken from raw observation column 0 BEFORE the time channel is dropped.
        Use it with filter_by_phase() to train phase-specific world models.

    Returns
    -------
    obs, act, nobs : (N, 8), (N, 6), (N, 8) float64 -- model units if standardize=True
    std_obs, std_act : Standardizer or None
    t_hours : (N,) float64 -- only when return_time=True
    """
    from smpl.envs.pensimenv import PeniControlData

    folder = dataset_folder or default_dataset_folder()
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"dataset folder not found: {folder}")
    d = PeniControlData(dataset_folder=folder, normalize=smpl_normalize).get_dataset()
    if d is None:
        raise RuntimeError("get_dataset() returned None (no CSVs parsed?)")

    obs = np.asarray(d["observations"], dtype=np.float64)
    act = np.asarray(d["actions"], dtype=np.float64)
    nobs = np.asarray(d["next_observations"], dtype=np.float64)
    if max_transitions is not None:
        obs = obs[:max_transitions]; act = act[:max_transitions]; nobs = nobs[:max_transitions]

    # capture time (hours) BEFORE the channel is dropped
    t_hours = extract_time_hours(obs, smpl_normalized=smpl_normalize)

    if drop_time_channel:
        obs, nobs = drop_time(obs), drop_time(nobs)

    std_obs = std_act = None
    if standardize:
        std_obs = Standardizer(obs, names=OBS_NAMES, name="obs")
        std_act = Standardizer(act, names=ACT_NAMES, name="act")
        if verbose:
            std_obs.report(); std_act.report()
        obs = std_obs.transform(obs)
        nobs = std_obs.transform(nobs)
        act = std_act.transform(act)

    if verbose:
        print(f"[pensim] {obs.shape[0]} transitions | obs {obs.shape[1]}d act {act.shape[1]}d")
        print(f"[pensim] obs per-dim std: {np.round(obs.std(0), 4)}")
        print(f"[pensim] act per-dim std: {np.round(act.std(0), 4)}")
        if t_hours is not None:
            print(f"[pensim] time range: {t_hours.min():.2f} .. {t_hours.max():.2f} h")
            for ph in (0, 1, 2):
                lo, hi = PHASES[ph]
                n = int(phase_mask(t_hours, ph).sum())
                hi_s = "inf" if hi > 1e8 else f"{hi:g}"
                print(f"[pensim]   phase {ph} [{lo:g}, {hi_s}) h: {n} transitions")

    if return_time:
        return obs, act, nobs, std_obs, std_act, t_hours
    return obs, act, nobs, std_obs, std_act


def collect_online(env, num_episodes=1, max_steps=None, policy=None, seed=0,
                   drop_time_channel=True, std_obs=None, std_act=None):
    """Roll the env; returns (obs, act, next_obs). Slow: 1150 ODE solves/episode.

    Pass the EXISTING std_obs/std_act to standardize with the statistics the model was
    trained on -- refitting on a new rollout would silently change the input space.
    """
    rng = np.random.default_rng(seed)
    O, A, N = [], [], []
    for ep in range(num_episodes):
        o = np.asarray(env.reset(), dtype=np.float64).reshape(-1)
        t = 0
        while True:
            a = rng.uniform(0, 1, ACT_DIM) if policy is None else np.asarray(policy(o), float)
            out = env.step(a)
            n, done = np.asarray(out[0], dtype=np.float64).reshape(-1), bool(out[2])
            O.append(o); A.append(a); N.append(n)
            o, t = n, t + 1
            if done or (max_steps and t >= max_steps):
                break
        print(f"[pensim] episode {ep}: {t} transitions")

    O, A, N = np.asarray(O), np.asarray(A), np.asarray(N)
    if drop_time_channel:
        O, N = drop_time(O), drop_time(N)
    if std_obs is not None:
        O, N = std_obs.transform(O), std_obs.transform(N)
    if std_act is not None:
        A = std_act.transform(A)
    return O, A, N


# -------------------------------------------------------------- subsampling ---
def subsample(obs, act, nobs, n_keep=300, mode="stride", seed=0, t_hours=None):
    """Exact GP inference costs O(N^3) per GP (8 GPs here), so N must stay small.
    'stride' preserves temporal coverage of the batch; 'random' samples uniformly.

    If t_hours is given it is subsampled with the same indices and returned as a
    fourth element, so time labels stay aligned with the kept rows.
    """
    N = obs.shape[0]
    if n_keep is None or n_keep >= N:
        return (obs, act, nobs) if t_hours is None else (obs, act, nobs, t_hours)
    if mode == "stride":
        idx = np.linspace(0, N - 1, n_keep).astype(int)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(N, n_keep, replace=False))
    if t_hours is None:
        return obs[idx], act[idx], nobs[idx]
    return obs[idx], act[idx], nobs[idx], t_hours[idx]
