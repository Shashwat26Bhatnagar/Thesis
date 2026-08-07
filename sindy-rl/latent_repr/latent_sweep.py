"""
Sweep latent dimensions and report reward-prediction error for each.

    cd ~/Thesis/sindy-rl
    python -u latent_repr/latent_sweep.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy \
        --dims 2 4 8 16 32 --epochs 100 --out-dir runs/sweep

Runs each dimension over several seeds and reports mean +/- std, because a
single seed on a model this small produces spread that is easy to mistake for
a real difference between dimensions.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from latent_repr.latent_train import parse_args as train_parse_args, run
from latent_repr.latent_utils import save_json


def main(argv=None):
    p = argparse.ArgumentParser('latent dimension sweep')
    p.add_argument('--data-dir', required=True)
    p.add_argument('--pattern', default='*.csv')
    p.add_argument('--dims', type=int, nargs='+', default=[2, 4, 8, 16, 32])
    p.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 64])
    p.add_argument('--reward-mode', default='step',
                   choices=['step', 'cumulative', 'final'])
    p.add_argument('--include-time', action='store_true')
    p.add_argument('--include-action', action='store_true')
    p.add_argument('--early-stop', type=int, default=20)
    p.add_argument('--out-dir', default='runs/sweep')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    all_results = []

    for d in args.dims:
        for seed in args.seeds:
            tag = f'd{d}_s{seed}'
            print('\n' + '=' * 70)
            print(f'latent_dim = {d}   seed = {seed}')
            print('=' * 70)

            argv_run = [
                '--data-dir', args.data_dir,
                '--pattern', args.pattern,
                '--latent-dim', str(d),
                '--epochs', str(args.epochs),
                '--batch-size', str(args.batch_size),
                '--lr', str(args.lr),
                '--seed', str(seed),
                '--reward-mode', args.reward_mode,
                '--early-stop', str(args.early_stop),
                '--out-dir', os.path.join(args.out_dir, tag),
                '--log-every', '25',
                '--hidden-sizes', *[str(h) for h in args.hidden_sizes],
            ]
            if args.include_time:
                argv_run.append('--include-time')
            if args.include_action:
                argv_run.append('--include-action')
            if args.cpu:
                argv_run.append('--cpu')

            res = run(train_parse_args(argv_run))
            res['seed'] = seed
            all_results.append(res)

    # ---------------- aggregate ----------------
    print('\n' + '=' * 78)
    print('SUMMARY -- validation MSE (standardised targets)')
    print('=' * 78)
    header = (f'{"dim":>5s} {"val mean":>11s} {"val std":>10s} '
              f'{"baseline":>10s} {"RMSE phys":>11s} {"R2":>8s} '
              f'{"eff.rank":>9s}')
    print(header)

    summary = []
    for d in args.dims:
        rows = [r for r in all_results if r['latent_dim'] == d]
        if not rows:
            continue
        vals = np.array([r['best_val_loss'] for r in rows])
        rmse = np.array([r['rmse_physical'] for r in rows])
        r2 = np.array([r['r2'] for r in rows])
        rank = np.array([r['effective_rank'] for r in rows])
        base = rows[0]['baseline_val_loss']
        print(f'{d:5d} {vals.mean():11.5f} {vals.std():10.5f} '
              f'{base:10.5f} {rmse.mean():11.5f} {r2.mean():8.4f} '
              f'{rank.mean():9.1f}')
        summary.append({
            'latent_dim': d,
            'val_loss_mean': float(vals.mean()),
            'val_loss_std': float(vals.std()),
            'baseline_val_loss': float(base),
            'rmse_physical_mean': float(rmse.mean()),
            'r2_mean': float(r2.mean()),
            'effective_rank_mean': float(rank.mean()),
            'beats_baseline': bool(vals.mean() < base),
            'n_seeds': len(rows),
        })

    save_json(os.path.join(args.out_dir, 'sweep_summary.json'),
              {'summary': summary, 'runs': all_results})

    # Interpretation notes -- these are the checks worth making before
    # reading a dimension ranking as meaningful.
    print('\nnotes:')
    beat = [s for s in summary if s['beats_baseline']]
    if not beat:
        print('  NO dimension beat the mean-predictor baseline. The latent')
        print('  space is not capturing reward structure; changing the')
        print('  dimension will not help until that is resolved.')
    else:
        best = min(beat, key=lambda s: s['val_loss_mean'])
        print(f'  best: d={best["latent_dim"]} '
              f'(val {best["val_loss_mean"]:.5f} vs baseline '
              f'{best["baseline_val_loss"]:.5f})')
    for s in summary:
        if s['effective_rank_mean'] < s['latent_dim'] - 0.5:
            print(f'  d={s["latent_dim"]}: effective rank only '
                  f'{s["effective_rank_mean"]:.1f} -- some latent directions '
                  f'are unused, so the extra dimensions are not being spent.')
    print(f'\nwrote {args.out_dir}/sweep_summary.json')


if __name__ == '__main__':
    main()
