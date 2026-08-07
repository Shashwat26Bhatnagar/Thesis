"""
Print the REAL observation that enters observation_done_and_reward_calculator,
and run each of that function's error checks by hand to see which one fires.

    cd ~/Thesis/sindy-rl
    python -u sindy_rl/scripts/diagnose_box2.py
"""

import math
import numpy as np
import smpl.envs.pensimenv as m
from sindy_rl.pensim_env import default_recipe_combo

NAMES = ['time', 'pH', 'Temp', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']


def main():
    env = m.PenSimEnvGym(recipe_combo=default_recipe_combo(),
                         normalize=False, random_seed=0)
    env.reset()

    a = np.array([2050.0, 79.0, 28.5, 52.5, 0.85, 255.0], dtype=env.np_dtype)

    # Replicate the try block exactly.
    env.step_count += 1
    vd = env.recipe_combo.get_values_dict_at(env.step_count * m.STEP_IN_MINUTES)
    po, x, ypr, done = super(m.PenSimEnvGym, env).step(
        env.step_count, env.x, a[1], a[2], a[3], a[4], a[0], a[5], vd['Fpaa'])
    env.x = x
    obs = m.get_observation_data_reformed(x, env.step_count - 1)

    obs_arr = np.array(obs, dtype=env.np_dtype)
    lo = np.asarray(env.min_observations, dtype=np.float64)
    hi = np.asarray(env.max_observations, dtype=np.float64)

    print('REAL observation entering the calculator:')
    for i, name in enumerate(NAMES[:len(obs_arr)]):
        v = float(obs_arr[i])
        out = v < lo[i] or v > hi[i]
        nan = math.isnan(v)
        mark = ''
        if nan:
            mark = '   <== NaN'
        elif out:
            mark = '   <== OUTSIDE'
        print(f'  [{i}] {name:5s} {v:16.6f}   box [{lo[i]:.4f}, {hi[i]:.4f}]{mark}')

    print('\nDtype of obs array   :', obs_arr.dtype)
    print('Any NaN              :', bool(np.any(np.isnan(obs_arr))))
    print('Any inf              :', bool(np.any(np.isinf(obs_arr))))
    print('len(obs)             :', len(obs_arr), ' expected', env.observation_dim)

    print('\n--- the calculator\'s own checks ---')
    print('observation_beyond_box(obs) :', env.observation_beyond_box(obs_arr))
    print('action_beyond_box(action)   :', env.action_beyond_box(a))
    print('step_count >= max_steps     :',
          env.step_count >= env.max_steps,
          f'({env.step_count} >= {env.max_steps})')
    print('step_reward == error_reward :', ypr == env.error_reward,
          f'(reward={ypr})')
    print('isnan(step_reward)          :', math.isnan(ypr))

    # If observation_beyond_box is True, find out how it decides.
    if env.observation_beyond_box(obs_arr):
        print('\nobservation_beyond_box fired. Per-element comparison as the')
        print('function would do it (note dtype casting):')
        lo32 = np.array(env.min_observations, dtype=env.np_dtype)
        hi32 = np.array(env.max_observations, dtype=env.np_dtype)
        below = obs_arr < lo32
        above = obs_arr > hi32
        for i, name in enumerate(NAMES[:len(obs_arr)]):
            if below[i] or above[i]:
                print(f'  [{i}] {name}: value={obs_arr[i]!r} '
                      f'lo={lo32[i]!r} hi={hi32[i]!r} '
                      f'below={below[i]} above={above[i]}')

    import inspect
    print('\n--- source of observation_beyond_box ---')
    print(inspect.getsource(type(env).observation_beyond_box))


if __name__ == '__main__':
    main()
