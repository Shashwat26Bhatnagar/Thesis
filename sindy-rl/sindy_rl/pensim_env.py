# ---------------------------------------------------------------
# PenSim (SMPL / PenSimPy) environment wrapper for SINDy-RL.
#
# CRITICAL: the underlying env is constructed with normalize=False
# and driven with PHYSICAL actions. normalize=True is broken in this
# smpl build:
#
#     def observation_beyond_box(self, observation):
#         return (not self.observation_space.contains(observation)) or ...
#
# With normalize=True, observation_space is Box(-1, 1, (9,)) but
# step() passes the observation in PHYSICAL units (Wt ~ 62500) into
# that check. It is never contained, so error_occurred is set and
# error_reward (-100) is returned on EVERY step regardless of the
# action. This matches the working MCPILCO pipeline, which also uses
# normalize=False and passes physical actions.
#
# Consequently this wrapper does all scaling itself:
#
#   observations: physical --PeniControlData min-max--> [-1, 1]
#     using pcd.min_observations / pcd.max_observations, which are
#     the SAME bounds the off-policy buffer was built with. (The env's
#     own bounds differ -- time 552.0 vs 276.0 -- so mixing them would
#     train the world model in one scale and query it in another.)
#
#   actions: policy [-1,1] = fractional deviation from the recipe
#     setpoint -> physical, clipped to [min_actions, max_actions].
#     A zero action reproduces the default recipe exactly.
#
# Other notes:
#   * A FRESH PenSimEnvGym per reset(); reusing one leaks state and
#     every second rollout aborts at step 1 (observed in MCPILCO).
#   * The -100 error row carries no yield information: it is not
#     summed into the episode return and terminates the episode.
#   * The time channel is dropped from the observation (9-D -> 8-D).
# ---------------------------------------------------------------

import os
import numpy as np
import gymnasium
from gymnasium.spaces import Box

from pensimpy.examples.recipe import Recipe, RecipeCombo
from pensimpy.data.constants import (
    FS, FOIL, FG, PRES, DISCHARGE, WATER, PAA,
    FS_DEFAULT_PROFILE, FOIL_DEFAULT_PROFILE, FG_DEFAULT_PROFILE,
    PRESS_DEFAULT_PROFILE, DISCHARGE_DEFAULT_PROFILE,
    WATER_DEFAULT_PROFILE, PAA_DEFAULT_PROFILE,
)
from smpl.envs.pensimenv import PenSimEnvGym, PeniControlData

NUM_STEPS = 1150          # 1150 * 0.2h = 230h batch
N_OBS = 8                 # 9-D state minus the time channel
N_ACT = 6
ERROR_REWARD = -100.0    # PenSimEnvGym error_reward

# Column layout of the shipped CSVs (16 columns).
# 0: Time Step | 1-6: actions | 7-14: observations | 15: Yield Per Step
ACT_COLS = slice(1, 7)
OBS_COLS = slice(7, 15)
REW_COL = 15


def default_recipe_combo():
    recipe_dict = {
        FS: Recipe(FS_DEFAULT_PROFILE, FS),
        FOIL: Recipe(FOIL_DEFAULT_PROFILE, FOIL),
        FG: Recipe(FG_DEFAULT_PROFILE, FG),
        PRES: Recipe(PRESS_DEFAULT_PROFILE, PRES),
        DISCHARGE: Recipe(DISCHARGE_DEFAULT_PROFILE, DISCHARGE),
        WATER: Recipe(WATER_DEFAULT_PROFILE, WATER),
        PAA: Recipe(PAA_DEFAULT_PROFILE, PAA),
    }
    return RecipeCombo(recipe_dict=recipe_dict)


# ---------------- action unit conversions ----------------

def physical_to_env_action(u_phys, min_actions, max_actions):
    """Physical units -> the [-1, 1] the env's step() expects."""
    lo = np.asarray(min_actions, dtype=np.float64)
    hi = np.asarray(max_actions, dtype=np.float64)
    u = np.clip(np.asarray(u_phys, dtype=np.float64).reshape(-1), lo, hi)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    return 2.0 * (u - lo) / span - 1.0


def env_action_to_physical(a_env, min_actions, max_actions):
    """Inverse of the above; mirrors smpl's denormalize_spaces."""
    lo = np.asarray(min_actions, dtype=np.float64)
    hi = np.asarray(max_actions, dtype=np.float64)
    a = np.asarray(a_env, dtype=np.float64).reshape(-1)
    return lo + (a + 1.0) * 0.5 * (hi - lo)


def load_setpoint_trajectory(csv_path):
    """
    Per-step recipe setpoints (1150 x 6) from a rollout of the DEFAULT
    recipe (random_batch_*.csv). Do not point this at a gpei batch --
    those are optimised trajectories and would anchor the band to the
    wrong values.
    """
    raw = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
    setpoints = np.asarray(raw[:, ACT_COLS], dtype=np.float64)
    if setpoints.shape[0] < NUM_STEPS:
        raise ValueError(
            f'{csv_path} has {setpoints.shape[0]} rows, need >= {NUM_STEPS}'
        )
    return setpoints[:NUM_STEPS]


class PenSimGymEnv(gymnasium.Env):
    """
    gymnasium-compliant PenSim env for sindy-rl.

    env_config:
      dataset_folder:    folder of CSVs (normalisation bounds)
      setpoint_csv:      path to a random_batch_*.csv (default recipe rollout)
      action_band:       0.1 -> policy spans +/-10% of setpoint
      max_episode_steps: default 1150
      normalize:         True -> observations in [-1, 1]
      include_time:      False -> drop the time channel (default)
      obs_clip:          clip normalised obs to +/- this (None to disable)
    """

    def __init__(self, env_config=None):
        config = dict(env_config or {})
        self.config = config

        self.dataset_folder = config['dataset_folder']
        self.setpoint_csv = config.get('setpoint_csv') or os.path.join(self.dataset_folder, 'random_batch_0.csv')
        self.action_band = config.get('action_band', 0.1)
        self.max_episode_steps = int(config.get('max_episode_steps', NUM_STEPS))
        self.normalize = config.get('normalize', True)
        self.include_time = config.get('include_time', False)
        self.obs_clip = config.get('obs_clip', 2.0)

        self.setpoints = load_setpoint_trajectory(self.setpoint_csv)

        # One throwaway instance to read the action/observation bounds off,
        # so the conversions below use exactly what step() will use.
        probe = PenSimEnvGym(recipe_combo=default_recipe_combo(),
                             normalize=False, random_seed=0)
        self.min_actions = np.asarray(probe.min_actions, dtype=np.float64)
        self.max_actions = np.asarray(probe.max_actions, dtype=np.float64)
        del probe

        pcd = PeniControlData(dataset_folder=self.dataset_folder,
                              normalize=True)
        self.obs_min = np.asarray(pcd.min_observations, dtype=np.float64)
        self.obs_max = np.asarray(pcd.max_observations, dtype=np.float64)

        obs_dim = N_OBS + (1 if self.include_time else 0)
        self.obs_dim = obs_dim
        self.act_dim = N_ACT

        # Policy space: fractional deviation from the recipe.
        self.action_space = Box(low=-1.0, high=1.0,
                                shape=(N_ACT,), dtype=np.float64)

        bound = self.obs_clip if self.obs_clip else np.inf
        self.observation_space = Box(low=-bound, high=bound,
                                     shape=(obs_dim,), dtype=np.float64)

        self.env = None
        self._n_steps = 0
        self.last_seed = None
        self.episode_return = 0.0

    # ---------------- internals ----------------

    def _new_seed(self):
        return int(np.random.SeedSequence().generate_state(1)[0] % (2 ** 31 - 1))

    def _build_env(self, seed=None):
        # normalize=False -- see the module docstring. A fresh instance every
        # time: reusing one leaks state across episodes.
        self.last_seed = self._new_seed() if seed is None else int(seed)
        e = PenSimEnvGym(recipe_combo=default_recipe_combo(),
                         normalize=False,
                         random_seed=self.last_seed)
        if hasattr(e, 'seed'):
            e.seed(self.last_seed)
        return e

    def setpoint_at(self, step_idx):
        idx = min(int(step_idx), self.setpoints.shape[0] - 1)
        return self.setpoints[idx]

    def _to_physical_action(self, action):
        """
        Policy [-1,1] (fractional deviation from the recipe setpoint)
        -> PHYSICAL action, clipped to the env's action box.
        The env is normalize=False, so it consumes physical units directly.
        """
        a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        u_phys = self.setpoint_at(self._n_steps) * (1.0 + self.action_band * a)
        return np.clip(u_phys, self.min_actions, self.max_actions)

    def _process_obs(self, obs_phys):
        """
        PHYSICAL 9-D observation -> normalised, time-dropped model units.
        Uses PeniControlData's bounds, the same ones the off-policy buffer
        was built with, so the world model is fit and queried in one scale.
        """
        x = np.asarray(obs_phys, dtype=np.float64).reshape(-1)
        lo, hi = self.obs_min, self.obs_max
        if self.normalize:
            span = np.where((hi - lo) == 0, 1.0, hi - lo)
            x = 2.0 * (x - lo) / span - 1.0
        if x.shape[0] == N_OBS + 1 and not self.include_time:
            x = x[1:]
        if self.obs_clip:
            x = np.clip(x, -self.obs_clip, self.obs_clip)
        return x

    # ---------------- gymnasium API ----------------

    def reset(self, seed=None, options=None):
        self.env = self._build_env(seed=seed)
        self._n_steps = 0
        self.episode_return = 0.0

        res = self.env.reset()
        obs = res[0] if isinstance(res, tuple) else res
        return self._process_obs(obs), {'seed': self.last_seed}

    def step(self, action):
        u_phys = self._to_physical_action(action)
        res = self.env.step(u_phys)

        if len(res) == 5:
            obs, rew, term, trunc, info = res
            done = bool(term or trunc)
        else:
            obs, rew, done, info = res
            done = bool(done)

        info = dict(info or {})
        obs = self._process_obs(obs)
        rew = float(rew)
        self._n_steps += 1

        # PenSim returns error_reward (-100) INSTEAD of a yield at
        # termination. That row carries no yield information, so it is not
        # summed -- otherwise every episode total is 100 low and the shipped
        # baselines (which have no such row) look artificially better.
        if rew <= ERROR_REWARD + 1e-9 or info.get('error_occurred'):
            info['env_error'] = True
            info['episode_return'] = self.episode_return
            return obs, 0.0, True, False, info

        if not (np.all(np.isfinite(obs)) and np.isfinite(rew)):
            info['nonfinite'] = True
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
            return obs, 0.0, True, False, info

        self.episode_return += rew
        truncated = self._n_steps >= self.max_episode_steps
        info['episode_return'] = self.episode_return
        return obs, rew, done, truncated, info


class PenSimGymEnvWithTime(PenSimGymEnv):
    """Keeps the time channel. Fermentation is strongly non-autonomous, so
    if the sparse fit is poor without an explicit phase coordinate, try this
    before enlarging the library."""

    def __init__(self, env_config=None):
        config = dict(env_config or {})
        config['include_time'] = True
        super().__init__(config)


def resolve_setpoints(setpoint_csv=None, n_steps=NUM_STEPS,
                      data_dir=None):
    """
    Setpoint trajectory for the default recipe.

    Reads them from a rollout of the UNMODIFIED default recipe
    (random_batch_0.csv). Never point this at a gpei batch -- those are
    optimised trajectories and would anchor the +/-band to the wrong values.
    """
    if not setpoint_csv:
        base = data_dir or os.path.join(
            os.path.expanduser('~'),
            'Thesis/deps/smpl/smpl/configdata/pensim_sindy')
        setpoint_csv = os.path.join(base, 'random_batch_0.csv')
    return load_setpoint_trajectory(os.path.expanduser(setpoint_csv))
