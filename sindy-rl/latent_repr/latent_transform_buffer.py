"""
Stage 2 -- project the offline dataset into the latent space.

Reads the SINDy-RL trajectory buffer, replaces every state with its latent
encoding, and writes a new buffer:

    (s_t, a_t, r_t, s_{t+1})  ->  (z_t, a_t, r_t, z_{t+1})

Actions and rewards are untouched. Trajectory boundaries are preserved, so
`s_{t+1}` never crosses from the end of one batch into the start of another.

    cd ~/Thesis/sindy-rl
    python -u latent_repr/latent_transform_buffer.py \
        --buffer data/pensim_offpi.pkl \
        --checkpoint runs/latent_d4/best.pt \
        --out data/pensim_offpi_latent_d4.pkl

Also writes a small JSON of latent statistics -- per-coordinate min/max/mean
and suggested `obs_bounds` for the surrogate env, which cannot be guessed
analytically because the latent scale depends entirely on the learned W.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from latent_repr.latent_encoder import FrozenLinearEncoder
from latent_repr.latent_utils import save_json
from sindy_rl.traj_buffer import BaseTrajectoryBuffer


def main(argv=None):
    p = argparse.ArgumentParser('project the offline buffer into latent space')
    p.add_argument('--buffer', required=True, help='input buffer pickle')
    p.add_argument('--checkpoint', required=True, help='latent best.pt')
    p.add_argument('--out', required=True, help='output buffer pickle')
    p.add_argument('--margin', type=float, default=1.5,
                   help='widen empirical latent bounds by this factor')
    p.add_argument('--stats-out', default=None,
                   help='JSON of latent statistics (default: alongside --out)')
    args = p.parse_args(argv)

    enc = FrozenLinearEncoder(os.path.expanduser(args.checkpoint))
    print(enc.summary())
    print(enc.describe())

    buf = BaseTrajectoryBuffer()
    buf.load_data(os.path.expanduser(args.buffer))
    X, U, R = buf.to_list()
    print(f'\ninput buffer: {len(X)} trajectories, '
          f'{sum(len(x) for x in X)} samples, state_dim={X[0].shape[1]}')

    if X[0].shape[1] != enc.state_dim:
        raise ValueError(
            f'buffer states are {X[0].shape[1]}-D but the encoder expects '
            f'{enc.state_dim}-D. The encoder was trained on features '
            f'{enc.feature_names}; check --include-time / --include-action '
            f'used during latent training.')

    out_buf = BaseTrajectoryBuffer()
    all_Z = []

    for i, (x, u, r) in enumerate(zip(X, U, R)):
        # Encode the whole trajectory at once. Because the encoding is applied
        # per-row and the row order is preserved, z_{t+1} for row t is simply
        # row t+1 of the result -- the transition structure carries over
        # without any explicit pairing, and trajectory boundaries stay intact
        # because each trajectory is encoded separately.
        z = enc.encode(np.asarray(x, dtype=np.float64))
        all_Z.append(z)
        out_buf.append(z, np.asarray(u, dtype=np.float64),
                       np.asarray(r, dtype=np.float64))

    out_path = os.path.expanduser(args.out)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_buf.save_data(out_path)

    Z = np.concatenate(all_Z, axis=0)
    print(f'output buffer: {len(all_Z)} trajectories, {len(Z)} samples, '
          f'latent_dim={Z.shape[1]}')
    print(f'saved -> {out_path}')

    # ---------------- latent statistics ----------------
    lo_emp, hi_emp = Z.min(axis=0), Z.max(axis=0)
    centre = 0.5 * (lo_emp + hi_emp)
    half = 0.5 * (hi_emp - lo_emp) * args.margin
    half = np.where(half < 1e-6, 1.0, half)
    lo_b, hi_b = centre - half, centre + half

    print(f'\n{"coord":>6s} {"min":>10s} {"max":>10s} {"mean":>10s} '
          f'{"std":>10s} {"bound_lo":>10s} {"bound_hi":>10s}')
    for j in range(Z.shape[1]):
        print(f'z{j:<5d} {lo_emp[j]:10.4f} {hi_emp[j]:10.4f} '
              f'{Z[:, j].mean():10.4f} {Z[:, j].std():10.4f} '
              f'{lo_b[j]:10.4f} {hi_b[j]:10.4f}')

    dead = [j for j in range(Z.shape[1]) if Z[:, j].std() < 1e-6]
    if dead:
        print(f'\n  coordinates {dead} are constant across the whole dataset '
              f'-- the projection is not using them, so the effective latent '
              f'dimension is {Z.shape[1] - len(dead)}.')

    stats = {
        'latent_dim': int(Z.shape[1]),
        'n_trajectories': len(all_Z),
        'n_samples': int(len(Z)),
        'min': lo_emp.tolist(),
        'max': hi_emp.tolist(),
        'mean': Z.mean(axis=0).tolist(),
        'std': Z.std(axis=0).tolist(),
        'obs_bounds': [[float(a), float(b)] for a, b in zip(lo_b, hi_b)],
        'margin': args.margin,
        'encoder_checkpoint': enc.checkpoint_path,
        'encoder_features': enc.feature_names,
    }
    stats_path = args.stats_out or os.path.splitext(out_path)[0] + '_stats.json'
    save_json(stats_path, stats)
    print(f'\nlatent statistics -> {stats_path}')
    print('\nPaste these into the config as obs_bounds (and obs_dim '
          f'= {Z.shape[1]}):')
    for a, b in zip(lo_b, hi_b):
        print(f'          - [{a:.4f}, {b:.4f}]')


if __name__ == '__main__':
    main()
