"""
Dataset for the latent-representation experiments.

Loads one or more PenSim trajectory CSVs and exposes (state, reward) pairs.

A note on the reward target
---------------------------
The CSVs carry `Yield Per Step` -- a per-step increment, summing to roughly
3729 over a 1150-step batch. Three supervision targets are supported:

  'step'        the per-step yield (default). Dense, one distinct target per
                row, but small in magnitude (~0.05) and noisy.
  'cumulative'  yield accumulated so far within the batch. Monotone in time,
                so a latent that merely encodes elapsed time will predict it
                well -- interpret a low loss here with care.
  'final'       the batch total, broadcast to every row. This matches the
                phrase "the final penicillin yield", but note that with 10
                CSVs there are only 10 distinct target values, so every row
                of a batch carries an identical label and the effective
                sample size for the regression is the number of FILES, not
                the number of rows.

Whichever is chosen, the target is standardised for training and the scaler
is kept so predictions can be reported in physical units.

A note on state units
---------------------
States go through the SAME two-stage chain the rest of the pipeline uses:

    physical --PeniControlData min-max--> [-1, 1] --z-score--> model units

The min-max stage uses PeniControlData's own bounds, which are the bounds the
SINDy off-policy buffer and PenSimGymEnv both normalise against. Training the
projection on physical units instead would produce a W that cannot be applied
to anything the pipeline actually emits.
"""

from __future__ import annotations

import glob
import os
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from latent_repr.latent_utils import (
    ACT_COLS, OBS_COLS, REW_COL, TIME_COL, OBS_NAMES, ACT_NAMES, Standardizer,
)


def load_minmax_bounds(data_dir: str, include_time: bool):
    """
    PeniControlData's observation bounds -- the SAME ones the SINDy buffer
    and PenSimGymEnv normalise against. Read off an instance rather than
    hardcoded: they are NOT PenSimEnvGym's defaults (time 276.0 vs 552.0).
    """
    try:
        from smpl.envs.pensimenv import PeniControlData
    except ImportError as exc:
        raise ImportError(
            'smpl is required to read the PeniControlData min-max bounds, '
            'which are the same bounds the SINDy buffer and PenSimGymEnv '
            'use. Run inside the sindyrl conda env, or pass explicit '
            'minmax_bounds=(lo, hi).') from exc
    pcd = PeniControlData(dataset_folder=os.path.expanduser(data_dir),
                          normalize=True)
    lo = np.asarray(pcd.min_observations, dtype=np.float64)
    hi = np.asarray(pcd.max_observations, dtype=np.float64)
    if not include_time and lo.shape[0] == 9:
        lo, hi = lo[1:], hi[1:]      # drop the time channel
    return lo, hi


def minmax_normalize(x, lo, hi):
    """physical -> [-1, 1]"""
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return 2.0 * (x - lo) / span - 1.0

REWARD_MODES = ('step', 'cumulative', 'final')


class PenSimLatentDataset(Dataset):
    """
    (state, reward) pairs from PenSim trajectory CSVs.

    Args:
        csv_paths:      explicit list of files, or use `from_dir`.
        reward_mode:    one of REWARD_MODES.
        include_time:   prepend the time channel to the state.
        include_action: append the 6 action channels to the state. Off by
                        default -- with actions included the network can
                        predict yield from the control input directly, which
                        is a different (easier) problem than learning a
                        reward-organised STATE representation.
        minmax_bounds:  (lo, hi) from PeniControlData, applied to the
                        observation channels BEFORE the z-score. Pass None
                        only if the states are already normalised.
        state_scaler:   fitted Standardizer, or None to fit on this data.
                        Always pass the TRAINING scaler when building the
                        validation set.
        reward_scaler:  as above, for the target.
        drop_nonfinite: discard rows containing NaN/inf. A diverged simulator
                        writes non-finite values that otherwise poison the fit.
    """

    def __init__(self,
                 csv_paths: Sequence[str],
                 reward_mode: str = 'step',
                 include_time: bool = False,
                 include_action: bool = False,
                 minmax_bounds: tuple | None = None,
                 state_scaler: Standardizer | None = None,
                 reward_scaler: Standardizer | None = None,
                 drop_nonfinite: bool = True,
                 dtype: torch.dtype = torch.float32):

        if reward_mode not in REWARD_MODES:
            raise ValueError(f'reward_mode must be one of {REWARD_MODES}, '
                             f'got {reward_mode!r}')
        if not csv_paths:
            raise ValueError('no CSV files given')

        self.csv_paths = list(csv_paths)
        self.reward_mode = reward_mode
        self.include_time = include_time
        self.include_action = include_action
        self.minmax_bounds = minmax_bounds
        self.dtype = dtype

        states, rewards, traj_ids, times = [], [], [], []

        for tid, path in enumerate(self.csv_paths):
            raw = np.genfromtxt(path, delimiter=',', skip_header=1)
            if raw.ndim != 2 or raw.shape[0] < 2:
                print(f'  skipping {os.path.basename(path)}: too few rows')
                continue

            obs = np.asarray(raw[:, OBS_COLS], dtype=np.float64)
            act = np.asarray(raw[:, ACT_COLS], dtype=np.float64)
            rew = np.asarray(raw[:, REW_COL], dtype=np.float64)
            tim = np.asarray(raw[:, TIME_COL], dtype=np.float64)

            if drop_nonfinite:
                ok = (np.all(np.isfinite(obs), axis=1)
                      & np.all(np.isfinite(act), axis=1)
                      & np.isfinite(rew) & np.isfinite(tim))
                if not ok.all():
                    print(f'  {os.path.basename(path)}: dropping '
                          f'{int((~ok).sum())} non-finite rows')
                obs, act, rew, tim = obs[ok], act[ok], rew[ok], tim[ok]
                if len(obs) < 2:
                    continue

            # Stage 1 of the unit chain: min-max to [-1, 1] using the same
            # PeniControlData bounds the rest of the pipeline uses.
            parts = []
            if include_time:
                parts.append(tim.reshape(-1, 1))
            parts.append(obs)
            s_phys = np.hstack(parts)

            if minmax_bounds is not None:
                lo, hi = minmax_bounds
                if lo.shape[0] != s_phys.shape[1]:
                    raise ValueError(
                        f'min-max bounds have {lo.shape[0]} channels but the '
                        f'state has {s_phys.shape[1]}; check include_time')
                s = minmax_normalize(s_phys, lo, hi)
            else:
                s = s_phys

            # Actions are appended AFTER the min-max stage: they are already
            # in the policy's fractional-deviation space and must not be
            # rescaled by observation bounds.
            if include_action:
                s = np.hstack([s, act])

            # Build the target.
            if reward_mode == 'step':
                y = rew
            elif reward_mode == 'cumulative':
                y = np.cumsum(rew)
            else:  # 'final'
                y = np.full(len(rew), float(rew.sum()))

            states.append(s)
            rewards.append(y)
            traj_ids.append(np.full(len(s), tid, dtype=np.int64))
            times.append(tim)

        if not states:
            raise RuntimeError('every CSV was skipped -- check the input files')

        # X_raw is post-min-max, pre-z-score -- i.e. exactly what the env
        # and the SINDy buffer emit.
        self.X_raw = np.concatenate(states, axis=0)
        self.y_raw = np.concatenate(rewards, axis=0).reshape(-1, 1)
        self.traj_id = np.concatenate(traj_ids, axis=0)
        self.time = np.concatenate(times, axis=0)

        # Fit scalers on this split if none supplied.
        self.state_scaler = state_scaler or Standardizer.fit(self.X_raw)
        self.reward_scaler = reward_scaler or Standardizer.fit(self.y_raw)

        self.X = self.state_scaler.transform(self.X_raw).astype(np.float32)
        self.y = self.reward_scaler.transform(self.y_raw).astype(np.float32)

        self._X_t = torch.from_numpy(self.X).to(dtype)
        self._y_t = torch.from_numpy(self.y).to(dtype)

    # ------------------------------------------------------------------
    @classmethod
    def from_dir(cls, data_dir: str, pattern: str = '*.csv', **kwargs):
        paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir),
                                              pattern)))
        if not paths:
            raise FileNotFoundError(
                f'no files matching {pattern!r} in {data_dir}')
        return cls(paths, **kwargs)

    # ------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self.X.shape[1]

    @property
    def feature_names(self) -> list[str]:
        names = []
        if self.include_time:
            names.append('time')
        names += list(OBS_NAMES)
        if self.include_action:
            names += list(ACT_NAMES)
        return names

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self._X_t[idx], self._y_t[idx]

    def summary(self) -> str:
        return (f'{len(self.csv_paths)} files, {len(self)} rows, '
                f'state_dim={self.state_dim}, mode={self.reward_mode!r}, '
                f'target raw range '
                f'[{self.y_raw.min():.4f}, {self.y_raw.max():.4f}]')


def split_by_trajectory(csv_paths: Sequence[str],
                        val_fraction: float = 0.2,
                        seed: int = 0) -> tuple[list[str], list[str]]:
    """
    Split at the FILE level, not the row level.

    Rows within one batch are highly autocorrelated -- consecutive 12-minute
    steps barely differ -- so a random row split puts near-duplicates in both
    train and validation and the validation loss becomes meaningless.
    Holding out whole trajectories is the honest version.
    """
    paths = list(csv_paths)
    if len(paths) < 2:
        raise ValueError('need at least 2 files to hold one out')
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(paths))
    n_val = max(1, int(round(val_fraction * len(paths))))
    val_idx = set(order[:n_val].tolist())
    train = [p for i, p in enumerate(paths) if i not in val_idx]
    val = [p for i, p in enumerate(paths) if i in val_idx]
    return train, val


def build_datasets(data_dir: str,
                   pattern: str = '*.csv',
                   val_fraction: float = 0.2,
                   seed: int = 0,
                   **ds_kwargs) -> tuple[PenSimLatentDataset,
                                         PenSimLatentDataset]:
    """Train/validation datasets sharing the TRAINING scalers."""
    paths = sorted(glob.glob(os.path.join(os.path.expanduser(data_dir),
                                          pattern)))
    if not paths:
        raise FileNotFoundError(f'no files matching {pattern!r} in {data_dir}')

    include_time = ds_kwargs.get('include_time', False)
    if 'minmax_bounds' not in ds_kwargs:
        ds_kwargs['minmax_bounds'] = load_minmax_bounds(data_dir, include_time)
        lo, hi = ds_kwargs['minmax_bounds']
        print(f'min-max bounds from PeniControlData: {lo.shape[0]} channels')

    train_paths, val_paths = split_by_trajectory(paths, val_fraction, seed)
    print(f'train files ({len(train_paths)}): '
          f'{[os.path.basename(p) for p in train_paths]}')
    print(f'val   files ({len(val_paths)}): '
          f'{[os.path.basename(p) for p in val_paths]}')

    train_ds = PenSimLatentDataset(train_paths, **ds_kwargs)
    val_ds = PenSimLatentDataset(val_paths,
                                 state_scaler=train_ds.state_scaler,
                                 reward_scaler=train_ds.reward_scaler,
                                 **ds_kwargs)
    return train_ds, val_ds
