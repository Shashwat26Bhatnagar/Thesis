"""
Find out what units PenSimEnvGym actually expects for actions.

    cd ~/Thesis/sindy-rl
    python -u sindy_rl/scripts/diagnose_actions.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy

Prints the env's declared spaces, the CSV action magnitudes, and then tries
one step under each of three hypotheses to see which avoids the -100.
"""

import argparse
import os
import numpy as np

from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData
from sindy_rl.pensim_env import default_recipe_combo, ACT_COLS, OBS_COLS


def show_spaces(normalize):
    env = PenSimEnvGym(recipe_combo=default_recipe_combo(),
                       normalize=normalize, random_seed=0)
    print(f'\n--- PenSimEnvGym(normalize={normalize}) ---')
    print('action_space     :', env.action_space)
    print('  low            :', np.asarray(env.action_space.low))
    print('  high           :', np.asarray(env.action_space.high))
    print('observation_space:', env.observation_space)
    for attr in ('min_actions', 'max_actions',
                 'min_observations', 'max_observations'):
        if hasattr(env, attr):
            print(f'  {attr:18s}:', np.asarray(getattr(env, attr)))
    return env


def try_step(label, normalize, action_fn):
    """Build a fresh env, take one step, report the reward."""
    env = PenSimEnvGym(recipe_combo=default_recipe_combo(),
                       normalize=normalize, random_seed=0)
    res = env.reset()
    obs = res[0] if isinstance(res, tuple) else res
    a = action_fn(env)
    try:
        out = env.step(a)
    except Exception as exc:
        print(f'  {label:38s} EXCEPTION {type(exc).__name__}: {exc}')
        return
    rew = out[1]
    flag = '  <-- no penalty' if rew > -99 else ''
    print(f'  {label:38s} reward={rew:10.4f}{flag}')
    print(f'  {"":38s} action={np.round(np.asarray(a, float), 3)}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    args = p.parse_args()
    data_dir = os.path.expanduser(args.data_dir)

    for norm in (True, False):
        try:
            show_spaces(norm)
        except Exception as exc:
            print(f'normalize={norm} failed to construct: {exc}')

    # What magnitudes do the shipped CSVs actually contain?
    csv = os.path.join(data_dir, 'random_batch_0.csv')
    raw = np.genfromtxt(csv, delimiter=',', skip_header=1)
    acts = np.asarray(raw[:, ACT_COLS], dtype=np.float64)
    print(f'\n--- {os.path.basename(csv)} action columns ---')
    print('first row :', np.round(acts[0], 3))
    print('col min   :', np.round(acts.min(axis=0), 3))
    print('col max   :', np.round(acts.max(axis=0), 3))
    print('col mean  :', np.round(acts.mean(axis=0), 3))

    first_phys = acts[0].copy()

    print('\n--- one step under each hypothesis ---')

    print(' normalize=True:')
    try_step('zeros (mid-range if normalised)', True,
             lambda e: np.zeros(6))
    try_step('CSV row 0 as-is (physical)', True,
             lambda e: first_phys)
    try_step('sampled from action_space', True,
             lambda e: e.action_space.sample())

    print(' normalize=False:')
    try_step('CSV row 0 as-is (physical)', False,
             lambda e: first_phys)
    try_step('zeros', False,
             lambda e: np.zeros(6))
    try_step('sampled from action_space', False,
             lambda e: e.action_space.sample())

    print('\nWhichever line shows no penalty tells us the expected units.')


if __name__ == '__main__':
    main()
