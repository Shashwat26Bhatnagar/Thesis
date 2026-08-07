"""
Verify that the off-policy buffer and the live env produce the SAME
normalised values for the SAME physical state.

If they disagree, the world model is fit in one coordinate system and
queried in another, which degrades the fit without breaking anything
visibly -- exactly the failure mode that shows up as a one-step MSE
worse than persistence.

    cd ~/Thesis/sindy-rl
    python -u sindy_rl/scripts/check_normalization.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy \
        --buffer ~/Thesis/sindy-rl/data/pensim_offpi.pkl

Checks, in order:
  1. Which bounds each side uses, printed side by side.
  2. Round-trip: a known physical vector normalised by each path.
  3. Distribution overlap: buffer states vs env states, per channel.
  4. Does the CSV's own first row match the env's reset state?
"""

import argparse
import os
import numpy as np

from smpl.envs.pensimenv import PeniControlData
from sindy_rl.pensim_env import PenSimGymEnv, OBS_COLS, N_OBS
from sindy_rl.traj_buffer import BaseTrajectoryBuffer

NAMES_9 = ['time', 'pH', 'Temp', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
NAMES_8 = NAMES_9[1:]


def norm(x, lo, hi):
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return 2.0 * (x - lo) / span - 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--buffer', required=True)
    args = p.parse_args()
    data_dir = os.path.expanduser(args.data_dir)

    # ---------- 1. bounds used by each side ----------
    pcd = PeniControlData(dataset_folder=data_dir, normalize=True)
    pcd_lo = np.asarray(pcd.min_observations, dtype=np.float64)
    pcd_hi = np.asarray(pcd.max_observations, dtype=np.float64)

    env = PenSimGymEnv({'dataset_folder': data_dir, 'max_episode_steps': 5})
    env_lo, env_hi = env.obs_min, env.obs_max

    print('=' * 74)
    print('1. BOUNDS')
    print('=' * 74)
    print(f'{"channel":8s} {"pcd_lo":>12s} {"pcd_hi":>12s} '
          f'{"env_lo":>12s} {"env_hi":>12s}  match')
    same_bounds = True
    for i, nm in enumerate(NAMES_9[:len(pcd_lo)]):
        m = (np.isclose(pcd_lo[i], env_lo[i]) and
             np.isclose(pcd_hi[i], env_hi[i]))
        same_bounds &= m
        print(f'{nm:8s} {pcd_lo[i]:12.4f} {pcd_hi[i]:12.4f} '
              f'{env_lo[i]:12.4f} {env_hi[i]:12.4f}  {"yes" if m else "NO"}')
    print(f'\n  bounds identical: {same_bounds}')
    if not same_bounds:
        print('  !! The buffer and the env normalise against DIFFERENT bounds.')

    # ---------- 2. round-trip on a known vector ----------
    print('\n' + '=' * 74)
    print('2. ROUND TRIP -- same physical state through both paths')
    print('=' * 74)
    # The env's own documented reset state, in physical units.
    phys9 = np.array([0.2, 6.5, 300.0, 0.0, 10.828493, 0.0001,
                      150.0, 62500.0, 14.75])

    via_buffer = norm(phys9, pcd_lo, pcd_hi)[1:]      # drop time, as buffer does
    via_env = env._process_obs(phys9)                 # env's own path

    print(f'{"channel":8s} {"physical":>14s} {"via buffer":>12s} '
          f'{"via env":>12s} {"diff":>12s}')
    max_diff = 0.0
    for i, nm in enumerate(NAMES_8):
        d = abs(via_buffer[i] - via_env[i])
        max_diff = max(max_diff, d)
        flag = '  <== DIFFERS' if d > 1e-9 else ''
        print(f'{nm:8s} {phys9[i+1]:14.4f} {via_buffer[i]:12.6f} '
              f'{via_env[i]:12.6f} {d:12.2e}{flag}')
    print(f'\n  max difference: {max_diff:.3e}')
    print('  => paths agree' if max_diff < 1e-9
          else '  => !! PATHS DISAGREE -- this degrades the world model fit')

    # ---------- 3. distribution overlap ----------
    print('\n' + '=' * 74)
    print('3. DISTRIBUTIONS -- buffer states vs live env states')
    print('=' * 74)
    buf = BaseTrajectoryBuffer()
    buf.load_data(os.path.expanduser(args.buffer))
    X, U, R = buf.to_list()
    Xb = np.concatenate(X, axis=0)

    # Roll the env forward under the recipe (zero action) to get live states.
    live = []
    env2 = PenSimGymEnv({'dataset_folder': data_dir,
                         'max_episode_steps': 60})
    o, _ = env2.reset()
    live.append(o)
    for _ in range(59):
        o, r, term, trunc, info = env2.step(np.zeros(6))
        if info.get('env_error'):
            print('  (env error during rollout; stopping early)')
            break
        live.append(o)
        if term or trunc:
            break
    Xe = np.asarray(live)

    print(f'buffer: {Xb.shape[0]} states   live: {Xe.shape[0]} states\n')
    print(f'{"channel":8s} {"buf_min":>9s} {"buf_max":>9s} {"buf_mean":>9s} '
          f'{"env_min":>9s} {"env_max":>9s} {"env_mean":>9s}')
    for i, nm in enumerate(NAMES_8[:Xb.shape[1]]):
        print(f'{nm:8s} {Xb[:, i].min():9.3f} {Xb[:, i].max():9.3f} '
              f'{Xb[:, i].mean():9.3f} {Xe[:, i].min():9.3f} '
              f'{Xe[:, i].max():9.3f} {Xe[:, i].mean():9.3f}')

    # Live states are only the first 60 steps, so compare against the
    # matching slice of a buffer trajectory rather than the whole buffer.
    print('\n  (live states cover only the first 60 steps of a batch;')
    print('   compare against the buffer\'s own first 60 rows below)')
    X0 = X[0][:len(Xe)]
    print(f'\n{"channel":8s} {"buf[0][:n] mean":>16s} {"live mean":>12s} '
          f'{"abs diff":>12s}')
    for i, nm in enumerate(NAMES_8[:X0.shape[1]]):
        d = abs(X0[:, i].mean() - Xe[:, i].mean())
        flag = '  <== LARGE' if d > 0.1 else ''
        print(f'{nm:8s} {X0[:, i].mean():16.4f} {Xe[:, i].mean():12.4f} '
              f'{d:12.4f}{flag}')

    # ---------- 4. CSV row 0 vs env reset ----------
    print('\n' + '=' * 74)
    print('4. CSV FIRST ROW vs ENV RESET (physical units)')
    print('=' * 74)
    csv = os.path.join(data_dir, 'random_batch_0.csv')
    if os.path.exists(csv):
        raw = np.genfromtxt(csv, delimiter=',', skip_header=1)
        csv_obs = np.asarray(raw[0, OBS_COLS], dtype=np.float64)
        print(f'{"channel":8s} {"csv row 0":>14s} {"env reset":>14s} '
              f'{"diff":>12s}')
        for i, nm in enumerate(NAMES_8[:len(csv_obs)]):
            d = abs(csv_obs[i] - phys9[i + 1])
            flag = '  <== DIFFERS' if d > 1e-3 else ''
            print(f'{nm:8s} {csv_obs[i]:14.4f} {phys9[i+1]:14.4f} '
                  f'{d:12.4f}{flag}')
        print('\n  A mismatch here means the CSVs and the live env start from')
        print('  different states, so off-policy and on-policy data are not')
        print('  drawn from the same distribution.')
    else:
        print(f'  {csv} not found; skipping')


if __name__ == '__main__':
    main()
