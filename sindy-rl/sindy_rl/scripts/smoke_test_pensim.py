"""
10-second smoke test. Run this before any Slurm submission.

    python -u scripts/smoke_test_pensim.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy

Checks, in order:
  1. env constructs, resets, steps, and returns finite 8-D observations
  2. consecutive episodes do NOT alternate into the -100 boundary penalty
     (i.e. the fresh-env-per-reset fix is working)
  3. a zero action reproduces the default recipe yield (~3628 over a full
     batch) -- this is the real check that the setpoint mapping is right
  4. the SINDy dynamics model fits on the off-policy buffer and beats
     persistence on one-step MSE
"""

import argparse
import os
import numpy as np

from sindy_rl.pensim_env import PenSimGymEnv


def check_env(data_dir, n_ep=4, n_steps=5):
    cfg = {
        'dataset_folder': data_dir,
        'max_episode_steps': n_steps,
    }
    env = PenSimGymEnv(cfg)
    print(f'setpoints {env.setpoints.shape} from recipe objects')
    print(f'obs_space {env.observation_space.shape} '
          f'act_space {env.action_space.shape}')

    first_rews = []
    for ep in range(n_ep):
        obs, reset_info = env.reset()
        assert obs.shape == (env.obs_dim,), obs.shape
        assert np.all(np.isfinite(obs)), 'non-finite reset obs'
        rews = []
        for _ in range(n_steps):
            obs, rew, term, trunc, info = env.step(np.zeros(env.act_dim))
            rews.append(rew)
            if info.get('env_error'):
                print(f'    env_error at step {len(rews)}: {info}')
            if term or trunc:
                break
        first_rews.append(rews[0])
        print(f'  ep {ep} seed={reset_info["seed"]:>10} '
              f'r[0]={rews[0]:8.3f} sum={np.sum(rews):8.3f}')

    # The -100 penalty appearing on alternate episodes is the signature of a
    # leaked step counter -> band computed against a stale setpoint.
    penalised = [r <= -99 for r in first_rews]
    if any(penalised):
        print(f'  !! boundary penalty on episodes {np.where(penalised)[0]} '
              f'-- setpoint mapping or env reuse is wrong')
        return False
    print('  no boundary penalties across consecutive episodes')
    return True


def check_recipe_yield(data_dir, n_steps=1150):
    """Zero action == exactly the recipe. Total yield should land near the
    random_batch benchmark (~3628). If it does not, the setpoint trajectory
    is misaligned and every downstream number is meaningless."""
    cfg = {
        'dataset_folder': data_dir,
        'max_episode_steps': n_steps,
    }
    env = PenSimGymEnv(cfg)
    env.reset()
    total = 0.0
    for _ in range(n_steps):
        _, rew, term, trunc, _ = env.step(np.zeros(env.act_dim))
        total += rew
        if term or trunc:
            break
    print(f'zero-action total yield = {total:.1f}  (recipe benchmark ~3628)')
    if not (3000 < total < 4200):
        print('  !! outside the plausible band -- check setpoint alignment')
        return False
    return True


def check_dynamics(buffer_path):
    from sindy_rl.dynamics import EnsembleSINDyDynamicsModel
    from sindy_rl.traj_buffer import BaseTrajectoryBuffer

    buf = BaseTrajectoryBuffer()
    buf.load_data(os.path.expanduser(buffer_path))
    X, U, _ = buf.to_list()
    print(f'buffer: {len(X)} trajectories, {buf.total_samples()} samples')

    model = EnsembleSINDyDynamicsModel({
        'dt': 1, 'discrete': True,
        'optimizer': {
            'base_optimizer': {'name': 'STLSQ',
                               'kwargs': {'alpha': 1e-5, 'threshold': 1e-3}},
            'ensemble': {'bagging': True, 'library_ensemble': True,
                         'n_models': 20},
        },
        'feature_library': {
            'name': 'affine',
            'kwargs': {'poly_deg': 2, 'n_state': X[0].shape[1],
                       'n_control': U[0].shape[1],
                       'poly_int': True, 'tensor': False},
        },
    })

    # hold out the last trajectory
    model.fit(X[:-1], U[:-1])
    model.set_median_coef_()          # required before predict()
    x_te, u_te = X[-1], U[-1]

    pred = np.array([model.predict(x, u) for x, u in zip(x_te[:-1], u_te[:-1])])
    truth = x_te[1:]
    mse = float(np.mean((pred - truth) ** 2))
    persist = float(np.mean((x_te[:-1] - truth) ** 2))
    print(f'one-step MSE {mse:.5f}  persistence {persist:.5f}  '
          f'ratio {persist / max(mse, 1e-12):.2f}x')
    if mse >= persist:
        print('  !! no better than predicting s_{t+1} = s_t')
        return False
    return True


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', required=True)
    p.add_argument('--buffer', default=None,
                   help='off-policy pickle; skips the dynamics check if unset')
    p.add_argument('--full-batch', action='store_true',
                   help='also run the 1150-step recipe yield check (~minutes)')
    args = p.parse_args()

    data_dir = os.path.expanduser(args.data_dir)
    ok = check_env(data_dir)
    if args.full_batch:
        ok &= check_recipe_yield(data_dir)
    if args.buffer:
        ok &= check_dynamics(args.buffer)
    print('\nSMOKE TEST', 'PASSED' if ok else 'FAILED')
    raise SystemExit(0 if ok else 1)
