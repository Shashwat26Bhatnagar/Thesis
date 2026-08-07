"""
Stage 5 -- the latent-observation environment.

The simulator is unchanged and still runs in its original state space. This
wrapper sits between it and everything else, projecting each observation
through the frozen encoder before it is returned:

    env.step(a) -> s_{t+1}  --encoder-->  z_{t+1}   (what the caller sees)

Because the projection happens at the environment boundary, no other part of
the pipeline needs to know about it. The policy's input dimension follows
from `observation_space`, so RLlib builds a network of the right shape
automatically; the SINDy dynamics model sees latent states because that is
what the buffer and the collector give it. Stages 3, 4 and 5 all fall out of
this one wrapper -- which is why no algorithm code is modified.

The policy never receives the original high-dimensional state: it is
projected inside `reset()` and `step()` before either returns.
"""

from __future__ import annotations

import numpy as np
import gymnasium
from gymnasium.spaces import Box

from latent_repr.latent_encoder import FrozenLinearEncoder
from sindy_rl.pensim_env import PenSimGymEnv


class PenSimLatentEnv(gymnasium.Env):
    """
    PenSimGymEnv with observations projected into the learned latent space.

    env_config: everything PenSimGymEnv accepts, plus
        encoder_checkpoint: path to a best.pt from latent_train.py  (required)
        latent_bounds:      [[lo, hi], ...] per coordinate. Take these from
                            the *_stats.json written by
                            latent_transform_buffer.py -- they cannot be
                            guessed, since the latent scale depends on W.
        latent_clip:        clip latents into latent_bounds (default True).
                            Off-manifold states otherwise produce latents far
                            outside anything the world model was fit on.

    The underlying env keeps its own settings; in particular `normalize`
    stays True, because the encoder consumes exactly that [-1, 1]
    representation.
    """

    def __init__(self, env_config=None):
        config = dict(env_config or {})

        ckpt = config.pop('encoder_checkpoint', None)
        if ckpt is None:
            raise ValueError('env_config needs `encoder_checkpoint`')
        bounds = config.pop('latent_bounds', None)
        self.latent_clip = config.pop('latent_clip', True)

        self.encoder = FrozenLinearEncoder(ckpt)
        self.inner = PenSimGymEnv(config)

        inner_dim = self.inner.observation_space.shape[0]
        if inner_dim != self.encoder.state_dim:
            raise ValueError(
                f'the env emits {inner_dim}-D observations but the encoder '
                f'expects {self.encoder.state_dim}-D. The encoder was trained '
                f'on {self.encoder.feature_names}; check include_time.')

        self.latent_dim = self.encoder.latent_dim
        self.action_space = self.inner.action_space

        if bounds is not None:
            b = np.asarray(bounds, dtype=np.float64)
            self._lo, self._hi = b[:, 0], b[:, 1]
        else:
            # No bounds supplied: leave the space unbounded rather than
            # inventing numbers. Rollouts will not be truncated on state
            # bounds, which is a real choice, not a safe default -- prefer
            # passing the values from the buffer stats.
            self._lo = np.full(self.latent_dim, -np.inf)
            self._hi = np.full(self.latent_dim, np.inf)
            self.latent_clip = False

        self.observation_space = Box(low=self._lo.astype(np.float64),
                                     high=self._hi.astype(np.float64),
                                     shape=(self.latent_dim,),
                                     dtype=np.float64)

    # ------------------------------------------------------------------

    def _project(self, obs) -> np.ndarray:
        z = self.encoder.encode(np.asarray(obs, dtype=np.float64).reshape(-1))
        if self.latent_clip:
            z = np.clip(z, self._lo, self._hi)
        return z

    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        obs, info = self.inner.reset(seed=seed, options=options)
        return self._project(obs), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.inner.step(action)
        return self._project(obs), rew, terminated, truncated, info

    # Pass-through so callers that reach for the underlying env still work.
    @property
    def setpoints(self):
        return self.inner.setpoints

    @property
    def episode_return(self):
        return self.inner.episode_return

    def setpoint_at(self, i):
        return self.inner.setpoint_at(i)


class PenSimLatentEnvWithTime(PenSimLatentEnv):
    """Latent env whose underlying observation keeps the time channel. Use
    only with an encoder trained with --include-time, or the dimension check
    in __init__ will reject it."""

    def __init__(self, env_config=None):
        config = dict(env_config or {})
        config['include_time'] = True
        super().__init__(config)
