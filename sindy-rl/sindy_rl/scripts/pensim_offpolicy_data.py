"""
Build the SINDy-RL off-policy trajectory buffer from the shipped PenSim CSVs.

    python scripts/pensim_offpolicy_data.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensimenv \
        --out ~/Thesis/sindyRL/data/pensim_offpi.pkl

Actions are stored in the same fractional-deviation space the env exposes
(a = (u_physical/setpoint - 1)/band), so the off-policy data and the
on-policy rollouts live in one coordinate system. Without this the SINDy
dynamics model would be fit on physical actions and then queried with
normalised ones.

Observations are min-max normalised with PeniControlData's OWN bounds.
"""

import argparse
import glob
import os
import pickle
import numpy as np

from smpl.envs.pensimenv import PeniControlData

from sindy_rl.pensim_env import (
    NUM_STEPS, ACT_COLS, OBS_COLS, REW_COL, resolve_setpoints,
)
from sindy_rl.traj_buffer import BaseTrajectoryBuffer


def normalize_obs(x, lo, hi):
    """physical -> [-1, 1] using PeniControlData bounds."""
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return 2.0 * (x - lo) / span - 1.0


def to_fractional_action(u_phys, setpoints, band):
    """
    Physical action -> the policy's fractional-deviation space, matching
    PenSimGymEnv._to_env_action in reverse. NOTE this is the POLICY space
    ([-1,1] = +/-band around the recipe), not the env's internal [-1,1]
    (which spans min_actions..max_actions). The two are different.
    """
    sp = np.where(setpoints == 0, 1.0, setpoints)
    frac = (u_phys / sp - 1.0) / band
    # Setpoint-zero channels (e.g. discharge early in the batch) carry no
    # meaningful relative deviation; pin them to 0 rather than dividing.
    frac = np.where(setpoints == 0, 0.0, frac)
    return frac


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--setpoint-csv', default=None,
                   help='optional; default is to read setpoints from the '
                        'pensimpy Recipe objects')
    p.add_argument('--pattern', default='gpei_batch_*.csv',
                   help='which CSVs to ingest (default: gpei only)')
    p.add_argument('--band', type=float, default=0.1)
    p.add_argument('--include-time', action='store_true')
    p.add_argument('--clip', type=float, default=2.0)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    setpoints = resolve_setpoints(args.setpoint_csv, n_steps=NUM_STEPS)

    pcd = PeniControlData(dataset_folder=data_dir, normalize=True)
    obs_min = np.asarray(pcd.min_observations, dtype=np.float64)
    obs_max = np.asarray(pcd.max_observations, dtype=np.float64)
    # Drop the time channel from the bounds unless we are keeping it.
    if not args.include_time and obs_min.shape[0] == 9:
        obs_min, obs_max = obs_min[1:], obs_max[1:]

    files = sorted(glob.glob(os.path.join(data_dir, args.pattern)))
    if not files:
        raise SystemExit(f'no CSVs matched in {data_dir}')

    buffer = BaseTrajectoryBuffer()
    kept, dropped = [], []

    for path in files:
        raw = np.genfromtxt(path, delimiter=',', skip_header=1)
        if raw.ndim != 2 or raw.shape[0] < 2:
            dropped.append((os.path.basename(path), 'too short'))
            continue

        n = min(raw.shape[0], NUM_STEPS)
        raw = raw[:n]

        x_phys = np.asarray(raw[:, OBS_COLS], dtype=np.float64)
        u_phys = np.asarray(raw[:, ACT_COLS], dtype=np.float64)
        r = np.asarray(raw[:, REW_COL], dtype=np.float64).reshape(-1)

        if args.include_time:
            t = np.asarray(raw[:, 0], dtype=np.float64).reshape(-1, 1)
            x_phys = np.hstack([t, x_phys])

        # Drop non-finite rows at load; a single NaN row poisons the whole fit.
        finite = (np.all(np.isfinite(x_phys), axis=1)
                  & np.all(np.isfinite(u_phys), axis=1)
                  & np.isfinite(r).reshape(-1))
        if finite.sum() < 0.5 * n:
            dropped.append((os.path.basename(path),
                            f'{n - int(finite.sum())} non-finite rows'))
            continue
        if not finite.all():
            print(f'  {os.path.basename(path)}: dropping '
                  f'{n - int(finite.sum())} non-finite rows')

        x_phys, u_phys, r = x_phys[finite], u_phys[finite], r[finite]
        sp = setpoints[:len(u_phys)]

        x = normalize_obs(x_phys, obs_min, obs_max)
        if args.clip:
            x = np.clip(x, -args.clip, args.clip)
        u = to_fractional_action(u_phys, sp, args.band)

        buffer.append(x, u, r)
        kept.append((os.path.basename(path), len(x), float(r.sum())))

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    buffer.save_data(out)

    print(f'\n{len(kept)} trajectories, {buffer.total_samples()} samples '
          f'-> {out}')
    for name, n, tot in kept:
        print(f'  {name:28s} n={n:5d}  total_yield={tot:9.1f}')
    for name, why in dropped:
        print(f'  SKIPPED {name}: {why}')

    if kept:
        yields = np.array([k[2] for k in kept])
        print(f'\ntotal yield: mean={yields.mean():.1f} '
              f'std={yields.std():.1f} min={yields.min():.1f} '
              f'max={yields.max():.1f}')
        # Sanity check against the known benchmark: gpei ~3729, random ~3628.
        u_frac = np.concatenate(buffer.u_traj_buffer)
        print(f'action deviation range: [{u_frac.min():.2f}, '
              f'{u_frac.max():.2f}]')
        print('  (gpei batches are optimised, not recipe-following, so these '
              'may exceed +/-1; if so the band or the encoding needs a '
              'second look before training)')


if __name__ == '__main__':
    main()
