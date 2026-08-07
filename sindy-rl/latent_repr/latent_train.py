"""
Training entry point for the latent reward-supervised representation.

    cd ~/Thesis/sindy-rl
    python -u latent_repr/latent_train.py \
        --data-dir ~/Thesis/deps/smpl/smpl/configdata/pensim_sindy \
        --latent-dim 4 --epochs 100 --out-dir runs/latent_d4

Reports a mean-predictor baseline alongside the model loss. Standardised
targets make the mean predictor score ~1.0, so a validation loss near 1.0
means the model has learned nothing useful -- worth checking before reading
anything into the latent plots.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from latent_repr.latent_dataset import build_datasets
from latent_repr.latent_models import build_model
from latent_repr.latent_utils import (
    RunningMeter, count_parameters, format_matrix, get_device,
    save_checkpoint, save_json, set_seed,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser('latent reward representation')

    # data
    p.add_argument('--data-dir', required=True)
    p.add_argument('--pattern', default='*.csv')
    p.add_argument('--reward-mode', default='step',
                   choices=['step', 'cumulative', 'final'])
    p.add_argument('--include-time', action='store_true')
    p.add_argument('--include-action', action='store_true')
    p.add_argument('--val-fraction', type=float, default=0.2)

    # model
    p.add_argument('--latent-dim', type=int, default=4)
    p.add_argument('--hidden-sizes', type=int, nargs='+', default=[64, 64])
    p.add_argument('--activation', default='relu',
                   choices=['relu', 'tanh', 'gelu', 'elu'])
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--no-bias', action='store_true')
    p.add_argument('--init', default='orthogonal',
                   choices=['orthogonal', 'xavier', 'normal'])

    # optimisation
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--proj-lr', type=float, default=None,
                   help='separate LR for the projection (default: same as --lr)')
    p.add_argument('--weight-decay', type=float, default=0.0)
    p.add_argument('--scheduler', default='plateau',
                   choices=['none', 'plateau', 'cosine', 'step'])
    p.add_argument('--patience', type=int, default=10,
                   help='plateau scheduler patience, in epochs')
    p.add_argument('--early-stop', type=int, default=0,
                   help='stop after N epochs without val improvement (0=off)')
    p.add_argument('--grad-clip', type=float, default=0.0)

    # bookkeeping
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default='runs/latent')
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--log-every', type=int, default=1)
    p.add_argument('--quiet', action='store_true')
    return p.parse_args(argv)


def evaluate(model, loader, criterion, device):
    model.eval()
    meter = RunningMeter()
    preds, targets = [], []
    with torch.no_grad():
        for s, y in loader:
            s, y = s.to(device), y.to(device)
            out = model(s)
            meter.update(criterion(out, y).item(), n=s.size(0))
            preds.append(out.cpu().numpy())
            targets.append(y.cpu().numpy())
    return meter.average, np.concatenate(preds), np.concatenate(targets)


def train_one_epoch(model, loader, criterion, optimizer, device, grad_clip):
    model.train()
    meter = RunningMeter()
    for s, y in loader:
        s, y = s.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(s), y)
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        meter.update(loss.item(), n=s.size(0))
    return meter.average


def run(args) -> dict:
    set_seed(args.seed)
    device = get_device(prefer_cuda=not args.cpu)
    os.makedirs(args.out_dir, exist_ok=True)

    if not args.quiet:
        print(f'device: {device}')

    # ---------------- data ----------------
    train_ds, val_ds = build_datasets(
        args.data_dir,
        pattern=args.pattern,
        val_fraction=args.val_fraction,
        seed=args.seed,
        reward_mode=args.reward_mode,
        include_time=args.include_time,
        include_action=args.include_action,
    )
    print(f'train: {train_ds.summary()}')
    print(f'val  : {val_ds.summary()}')
    print(f'features: {train_ds.feature_names}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(args.batch_size, 1024),
                            shuffle=False, num_workers=args.num_workers)

    # Baseline: predict the training mean for every row. Targets are
    # standardised, so this scores ~1.0 on the training split; the validation
    # figure is the number the model actually has to beat.
    y_tr_mean = float(np.mean(train_ds.y))
    base_train = float(np.mean((train_ds.y - y_tr_mean) ** 2))
    base_val = float(np.mean((val_ds.y - y_tr_mean) ** 2))
    print(f'mean-predictor baseline  train {base_train:.5f}  '
          f'val {base_val:.5f}')

    # ---------------- model ----------------
    cfg = {
        'latent_dim': args.latent_dim,
        'hidden_sizes': args.hidden_sizes,
        'activation': args.activation,
        'dropout': args.dropout,
        'bias': not args.no_bias,
        'init': args.init,
    }
    model = build_model(train_ds.state_dim, cfg).to(device)
    print(f'projection {train_ds.state_dim} -> {args.latent_dim}   '
          f'params: proj {count_parameters(model.projection)}, '
          f'reward net {count_parameters(model.reward_net)}')

    # Separate parameter groups so the projection can take a different step
    # size from the MLP if asked.
    proj_lr = args.proj_lr if args.proj_lr is not None else args.lr
    optimizer = torch.optim.Adam([
        {'params': model.projection.parameters(), 'lr': proj_lr},
        {'params': model.reward_net.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)

    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=args.patience)
    elif args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs)
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)
    else:
        scheduler = None

    criterion = nn.MSELoss()

    # ---------------- loop ----------------
    best_val = float('inf')
    best_epoch = -1
    history = []
    ckpt_path = os.path.join(args.out_dir, 'best.pt')
    epochs_since_best = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer,
                             device, args.grad_clip)
        va, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(va) if args.scheduler == 'plateau' else scheduler.step()

        lr_now = optimizer.param_groups[0]['lr']
        history.append({'epoch': epoch, 'train_loss': tr,
                        'val_loss': va, 'lr': lr_now})

        improved = va < best_val - 1e-12
        if improved:
            best_val, best_epoch = va, epoch
            epochs_since_best = 0
            save_checkpoint(ckpt_path, model.projection, model.reward_net,
                            optimizer, epoch, va,
                            {**cfg, 'state_dim': train_ds.state_dim,
                             'reward_mode': args.reward_mode,
                             'feature_names': train_ds.feature_names},
                            standardizer=train_ds.state_scaler)
        else:
            epochs_since_best += 1

        if not args.quiet and (epoch % args.log_every == 0 or improved):
            mark = ' *' if improved else ''
            print(f'epoch {epoch:4d}/{args.epochs}  '
                  f'train {tr:.6f}  val {va:.6f}  lr {lr_now:.2e}{mark}')

        if args.early_stop and epochs_since_best >= args.early_stop:
            print(f'early stop at epoch {epoch} '
                  f'({args.early_stop} epochs without improvement)')
            break

    elapsed = time.time() - t0

    # ---------------- report ----------------
    ckpt = torch.load(ckpt_path, map_location=device)
    model.projection.load_state_dict(ckpt['projection'])
    model.reward_net.load_state_dict(ckpt['reward_net'])
    final_val, preds, targets = evaluate(model, val_loader, criterion, device)

    # Back to physical units so the error is interpretable.
    rs = val_ds.reward_scaler
    preds_phys = preds * rs.std + rs.mean
    targ_phys = targets * rs.std + rs.mean
    mae_phys = float(np.mean(np.abs(preds_phys - targ_phys)))
    rmse_phys = float(np.sqrt(np.mean((preds_phys - targ_phys) ** 2)))
    ss_res = float(np.sum((targets - preds) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    print(f'\nbest epoch {best_epoch}  val {best_val:.6f}  '
          f'({elapsed:.1f}s)')
    print(f'val vs mean-predictor: {best_val:.5f} vs {base_val:.5f}  '
          f'({"BETTER" if best_val < base_val else "NO BETTER"})')
    print(f'physical units: MAE {mae_phys:.5f}  RMSE {rmse_phys:.5f}  '
          f'R2 {r2:.4f}')

    W = model.projection.W.cpu().numpy()
    print(f'\nlearned projection W  ({W.shape[0]} x {W.shape[1]})')
    print(format_matrix(W, col_names=train_ds.feature_names))
    print(f'\nrow norms      : '
          f'{np.round(model.projection.row_norms().cpu().numpy(), 4)}')
    print(f'effective rank : {model.projection.effective_rank()} '
          f'of {args.latent_dim}')

    result = {
        'latent_dim': args.latent_dim,
        'best_epoch': best_epoch,
        'best_val_loss': best_val,
        'final_val_loss': final_val,
        'baseline_val_loss': base_val,
        'baseline_train_loss': base_train,
        'beats_baseline': bool(best_val < base_val),
        'mae_physical': mae_phys,
        'rmse_physical': rmse_phys,
        'r2': r2,
        'effective_rank': model.projection.effective_rank(),
        'row_norms': model.projection.row_norms().cpu().numpy().tolist(),
        'elapsed_sec': elapsed,
        'config': {**cfg, 'reward_mode': args.reward_mode,
                   'lr': args.lr, 'batch_size': args.batch_size,
                   'epochs': args.epochs, 'seed': args.seed},
    }
    save_json(os.path.join(args.out_dir, 'result.json'), result)
    save_json(os.path.join(args.out_dir, 'history.json'), history)
    np.save(os.path.join(args.out_dir, 'projection_W.npy'), W)
    if model.projection.b is not None:
        np.save(os.path.join(args.out_dir, 'projection_b.npy'),
                model.projection.b.cpu().numpy())
    print(f'\nsaved to {args.out_dir}/  '
          f'(best.pt, result.json, history.json, projection_W.npy)')
    return result


if __name__ == '__main__':
    run(parse_args())
