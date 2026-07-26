#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""model_learning/pensim_dataset.py

Adapter between the SMPL PenSim environment and the MC-PILCO RBF world model.

PenSimEnvGym is a Gym env (reset/step), not an ODE f_sim, so MC_PILCO.reinforce()
is bypassed: the world model is fitted here (train_rbf_pensim.py) and then used as a
frozen one-step predictor for policy optimization (optimize_policy_pensim.py).

Raw observation (9) from get_observation_data_reformed:
    [time, pH, T, Fa, Fb, Fc, Fh, Wt, DO2]
Action (6): discharge, sugar feed, soil bean feed, aeration, back pressure, water inj.
NOTE: yield/reward is a SEPARATE return value from env.step(), NOT an observation
channel. Rewards are ignored here (state model only).

TWO PREPROCESSING DECISIONS:

1. DROP THE TIME CHANNEL (obs index 0). PenSimEnvGym's own docstring says "Time is
   not in our observation_space. We make the env time unaware and MDP", yet
   get_observation_data_reformed returns t as element 0. A time-indexed input breaks
   the Markov assumption the dynamics model relies on: the GP would learn "state at
   t" rather than "response to (state, action)".

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
# Index 0 (time) is dropped from the observation lists to match OBS_NAMES.
_MAX_OBS_RAW = [552.0, 16.10523, 725.6828, 13.717274, 540.0, 3600.0002, 1892.07874,
                253840.11, 47.898834]
_MIN_OBS_RAW = [0.0, 0.0, 118.98977, 0.0, 0.0, 0.0, 0.0, 25003.258, 0.0]
MAX_OBS = np.array(_MAX_OBS_RAW[1:], dtype=np.float64)      # (8,) time dropped
MIN_OBS = np.array(_MIN_OBS_RAW[1:], dtype=np.float64)
MAX_ACT = np.array([4100.0, 151.0, 36.0, 76.0, 1.2, 510.0], dtype=np.float64)
MIN_ACT = np.array([0.0, 7.0, 21.0, 29.0, 0.5, 0.0], dtype=np.float64)


def default_dataset_folder():
    """CSV batches shipped inside the installed smpl package."""
    try:
        import smpl
        return os.path.join(os.path.dirname(smpl.__file__), "configdata", "pensimenv")
    except Exception:
        return "smpl/configdata/pensimenv"


def drop_time(obs):
    """Remove the time channel from a (N, 9) observation array -> (N, 8).
    Idempotent: arrays already at OBS_DIM are returned unchanged."""
    obs = np.asarray(obs, dtype=np.float64)
    if obs.ndim == 1:
        obs = obs.reshape(1, -1)
    if obs.shape[1] == OBS_DIM_RAW:
        return np.delete(obs, TIME_INDEX, axis=1)
    return obs


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
                 drop_time_channel=True, standardize=True, verbose=True):
    """Load PeniControlData (reads every CSV batch in the folder).

    Returns
    -------
    obs, act, nobs : (N, 8), (N, 6), (N, 8) float64 -- model units if standardize=True
    std_obs, std_act : Standardizer or None
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
def subsample(obs, act, nobs, n_keep=300, mode="stride", seed=0):
    """Exact GP inference costs O(N^3) per GP (8 GPs here), so N must stay small.
    'stride' preserves temporal coverage of the batch; 'random' samples uniformly."""
    N = obs.shape[0]
    if n_keep >= N:
        return obs, act, nobs
    if mode == "stride":
        idx = np.linspace(0, N - 1, n_keep).astype(int)
    else:
        idx = np.sort(np.random.default_rng(seed).choice(N, n_keep, replace=False))
    return obs[idx], act[idx], nobs[idx]
