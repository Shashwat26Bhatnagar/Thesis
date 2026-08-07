"""
Visualise a trained latent space.

    python -u latent_repr/latent_visualize.py \
        --checkpoint runs/latent_d4/best.pt \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy \
        --out-dir runs/latent_d4

Produces:
  latent_scatter.png    latent coords (or PCA of them) coloured by true reward
  latent_by_time.png    the same points coloured by elapsed batch time
  pred_vs_true.png      predicted vs true reward, with the identity line
  projection_heatmap.png  W as a heatmap: which state features each latent uses

The time-coloured plot is the important control. Fermentation state moves
almost monotonically through a batch, so a latent that merely encodes elapsed
time will look beautifully organised by reward -- because cumulative yield is
itself a function of time. If the reward-coloured and time-coloured plots show
the same structure, the representation may have learned the clock rather than
anything about the process.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')          # headless: DICE nodes have no display
import matplotlib.pyplot as plt

from latent_repr.latent_dataset import PenSimLatentDataset
from latent_repr.latent_models import LinearProjection, RewardPredictor
from latent_repr.latent_utils import Standardizer, get_device


def load_trained(checkpoint_path: str, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt['config']
    proj = LinearProjection(cfg['state_dim'], cfg['latent_dim'],
                            bias=cfg.get('bias', True)).to(device)
    proj.load_state_dict(ckpt['projection'])
    net = RewardPredictor(cfg['latent_dim'],
                          tuple(cfg.get('hidden_sizes', (64, 64))),
                          activation=cfg.get('activation', 'relu')).to(device)
    net.load_state_dict(ckpt['reward_net'])
    proj.eval()
    net.eval()
    scaler = (Standardizer.from_dict(ckpt['standardizer'])
              if 'standardizer' in ckpt else None)
    return proj, net, cfg, scaler


def scatter(ax, xy, colour, title, cbar_label, cmap='viridis'):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=colour, s=4, alpha=0.6,
                    cmap=cmap, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel('component 1')
    ax.set_ylabel('component 2')
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)


def main(argv=None):
    p = argparse.ArgumentParser('visualise the latent space')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--pattern', default='*.csv')
    p.add_argument('--out-dir', default=None)
    p.add_argument('--max-points', type=int, default=8000,
                   help='subsample for legibility (0 = all)')
    p.add_argument('--dpi', type=int, default=130)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args(argv)

    device = get_device(prefer_cuda=not args.cpu)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.checkpoint))
    os.makedirs(out_dir, exist_ok=True)

    proj, net, cfg, scaler = load_trained(args.checkpoint, device)
    print(f'loaded {args.checkpoint}: state_dim={cfg["state_dim"]} '
          f'latent_dim={cfg["latent_dim"]} mode={cfg.get("reward_mode")}')

    ds = PenSimLatentDataset.from_dir(
        args.data_dir, pattern=args.pattern,
        reward_mode=cfg.get('reward_mode', 'step'),
        include_time=cfg.get('feature_names', [''])[0] == 'time',
        include_action='discharge' in cfg.get('feature_names', []),
        state_scaler=scaler,
    )
    print(f'dataset: {ds.summary()}')

    with torch.no_grad():
        S = torch.from_numpy(ds.X).float().to(device)
        Z = proj(S).cpu().numpy()
        pred = net(proj(S)).cpu().numpy().reshape(-1)

    reward = ds.y_raw.reshape(-1)
    time_h = ds.time
    pred_phys = pred * ds.reward_scaler.std[0] + ds.reward_scaler.mean[0]

    # Subsample for plotting only.
    n = len(Z)
    if args.max_points and n > args.max_points:
        idx = np.random.default_rng(0).choice(n, args.max_points,
                                              replace=False)
    else:
        idx = np.arange(n)

    # --- project to 2-D for display -------------------------------------
    if Z.shape[1] == 2:
        XY = Z
        method = 'latent coordinates (d=2, no PCA)'
        evr = None
    else:
        Zc = Z - Z.mean(axis=0)
        U, S_, Vt = np.linalg.svd(Zc, full_matrices=False)
        XY = Zc @ Vt[:2].T
        var = S_ ** 2 / max((len(Zc) - 1), 1)
        evr = var[:2] / var.sum()
        method = (f'PCA of {Z.shape[1]}-D latent '
                  f'(PC1 {evr[0]*100:.1f}%, PC2 {evr[1]*100:.1f}%)')
    print(f'display: {method}')

    # --- 1. latent coloured by reward -----------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, XY[idx], reward[idx],
            f'Latent space coloured by reward\n{method}',
            f'reward ({cfg.get("reward_mode", "step")})')
    fig.tight_layout()
    f1 = os.path.join(out_dir, 'latent_scatter.png')
    fig.savefig(f1, dpi=args.dpi)
    plt.close(fig)

    # --- 2. the same points coloured by time (the control) --------------
    fig, ax = plt.subplots(figsize=(7, 6))
    scatter(ax, XY[idx], time_h[idx],
            'Same latent space coloured by batch time\n'
            '(if this matches the reward plot, the latent may encode the clock)',
            'time (h)', cmap='plasma')
    fig.tight_layout()
    f2 = os.path.join(out_dir, 'latent_by_time.png')
    fig.savefig(f2, dpi=args.dpi)
    plt.close(fig)

    # --- 3. predicted vs true -------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(reward[idx], pred_phys[idx], s=4, alpha=0.4, linewidths=0)
    lim = [min(reward.min(), pred_phys.min()),
           max(reward.max(), pred_phys.max())]
    ax.plot(lim, lim, 'k--', lw=1, label='perfect prediction')
    ax.set_xlabel('true reward')
    ax.set_ylabel('predicted reward')
    ax.set_title('Reward prediction on all data')
    ax.legend()
    fig.tight_layout()
    f3 = os.path.join(out_dir, 'pred_vs_true.png')
    fig.savefig(f3, dpi=args.dpi)
    plt.close(fig)

    # --- 4. projection heatmap ------------------------------------------
    W = proj.W.cpu().numpy()
    names = cfg.get('feature_names',
                    [f'x{i}' for i in range(W.shape[1])])
    fig, ax = plt.subplots(figsize=(1.0 * W.shape[1] + 3,
                                    0.45 * W.shape[0] + 2.5))
    vmax = np.abs(W).max()
    im = ax.imshow(W, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='auto')
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_yticks(range(W.shape[0]))
    ax.set_yticklabels([f'z{i}' for i in range(W.shape[0])])
    ax.set_title('Learned projection W\n(which state features each latent reads)')
    plt.colorbar(im, ax=ax, label='weight')
    fig.tight_layout()
    f4 = os.path.join(out_dir, 'projection_heatmap.png')
    fig.savefig(f4, dpi=args.dpi)
    plt.close(fig)

    # --- correlations, as a numeric check on the plots ------------------
    print('\ncorrelation of each latent coordinate with:')
    print(f'{"":6s} {"reward":>10s} {"time":>10s}')
    for i in range(Z.shape[1]):
        cr = np.corrcoef(Z[:, i], reward)[0, 1]
        ct = np.corrcoef(Z[:, i], time_h)[0, 1]
        print(f'z{i:<5d} {cr:10.4f} {ct:10.4f}')
    r_time = abs(np.corrcoef(reward, time_h)[0, 1])
    print(f'\nreward-time correlation itself: {r_time:.4f}')
    if r_time > 0.8:
        print('  reward and time are strongly correlated in this dataset, so')
        print('  a latent aligned with reward may just be tracking elapsed time.')

    print(f'\nwrote:\n  {f1}\n  {f2}\n  {f3}\n  {f4}')


if __name__ == '__main__':
    main()
