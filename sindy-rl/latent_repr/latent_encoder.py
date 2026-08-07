"""
Frozen linear encoder: physical state -> latent state.

UNIT CHAIN
----------
The projection was trained on states that had already been min-max
normalised with PeniControlData's bounds -- the same normalisation the SINDy
off-policy buffer and PenSimGymEnv both apply -- and then z-scored:

    physical --min-max--> [-1, 1] --z-score--> model units --W,b--> z

This encoder therefore consumes states in the [-1, 1] min-max representation,
i.e. EXACTLY what PenSimGymEnv.step() returns and exactly what is stored in
the off-policy buffer. No physical-unit conversion is needed anywhere: the
env output can be fed straight in.

The encoder is frozen: no gradients, never updated during RL. It is loaded
once and reused by the dataset transform, the env wrapper, and any
evaluation code, so all three are guaranteed to agree.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from latent_repr.latent_utils import Standardizer, load_checkpoint


class FrozenLinearEncoder:
    """
    z = W @ standardize(s) + b

    Args:
        checkpoint: path to a `best.pt` written by latent_train.py
        device:     'cpu' is the right default -- this runs inside env.step()
                    one row at a time, where GPU transfer costs more than the
                    matmul saves.

    Frozen by construction: this class holds numpy arrays, not nn.Parameters,
    so there is no path by which RL training could update the encoder.
    """

    def __init__(self, checkpoint: str, device: str = 'cpu'):
        self.checkpoint_path = os.path.expanduser(checkpoint)
        ckpt = load_checkpoint(self.checkpoint_path, map_location=device)

        cfg = ckpt['config']
        self.config = cfg
        self.state_dim = int(cfg['state_dim'])
        self.latent_dim = int(cfg['latent_dim'])
        self.feature_names = cfg.get('feature_names', None)
        self.reward_mode = cfg.get('reward_mode', 'step')

        sd = ckpt['projection']
        # nn.Linear stores weight as (out, in); keep that convention.
        self.W = sd['linear.weight'].detach().cpu().numpy().astype(np.float64)
        self.b = (sd['linear.bias'].detach().cpu().numpy().astype(np.float64)
                  if 'linear.bias' in sd else np.zeros(self.latent_dim))

        if 'standardizer' not in ckpt:
            raise KeyError(
                'checkpoint has no standardizer. The projection was trained '
                'on z-scored states, so the scaler is required to reproduce '
                'the same latent coordinates. Retrain with a version of '
                'latent_train.py that saves it.')
        self.standardizer = Standardizer.from_dict(ckpt['standardizer'])

        if self.W.shape != (self.latent_dim, self.state_dim):
            raise ValueError(
                f'W has shape {self.W.shape}, expected '
                f'({self.latent_dim}, {self.state_dim})')

    # ------------------------------------------------------------------

    def encode(self, s_normalized) -> np.ndarray:
        """
        Min-max normalised state(s) -> latent.

        Input is the [-1, 1] representation emitted by PenSimGymEnv and stored
        in the off-policy buffer -- NOT physical units. Accepts a single
        vector (D,) or a batch (N, D); returns (d,) or (N, d).
        """
        s = np.asarray(s_normalized, dtype=np.float64)
        single = (s.ndim == 1)
        if single:
            s = s.reshape(1, -1)
        if s.shape[1] != self.state_dim:
            raise ValueError(
                f'expected state_dim={self.state_dim}, got {s.shape[1]}. '
                f'Encoder was trained on features: {self.feature_names}')

        s_z = self.standardizer.transform(s)
        z = s_z @ self.W.T + self.b
        return z[0] if single else z

    __call__ = encode

    # ------------------------------------------------------------------

    def latent_bounds(self, s_normalized_samples, margin: float = 1.5):
        """
        Empirical per-coordinate latent bounds from a sample of real states
        (min-max normalised, as everywhere else in this class),
        widened by `margin`.

        The surrogate env needs `obs_bounds` in LATENT units, and there is no
        analytic value for them -- the latent scale depends entirely on the
        learned W. Deriving them from data avoids either truncating valid
        rollouts (bounds too tight) or letting a diverging model run away
        (bounds too loose).
        """
        Z = self.encode(s_normalized_samples)
        lo, hi = Z.min(axis=0), Z.max(axis=0)
        centre = 0.5 * (lo + hi)
        half = 0.5 * (hi - lo) * margin
        # A coordinate that never varies would give a zero-width box.
        half = np.where(half < 1e-6, 1.0, half)
        return centre - half, centre + half

    def summary(self) -> str:
        return (f'FrozenLinearEncoder({os.path.basename(self.checkpoint_path)}): '
                f'{self.state_dim} -> {self.latent_dim}, '
                f'features={self.feature_names}')

    def describe(self) -> str:
        """Row norms and singular values -- how many directions W really spans."""
        norms = np.linalg.norm(self.W, axis=1)
        svals = np.linalg.svd(self.W, compute_uv=False)
        rank = int((svals > 1e-3 * svals.max()).sum())
        return (f'  row norms      : {np.round(norms, 4)}\n'
                f'  singular values: {np.round(svals, 4)}\n'
                f'  effective rank : {rank} of {self.latent_dim}')
