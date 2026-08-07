"""
Shared helpers for the latent-representation experiments.

Kept deliberately free of project-specific imports so this package can be
lifted out and reused; the only PenSim knowledge lives in the CSV column
constants below, which are documented rather than imported.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
import torch


# ---------------------------------------------------------------------------
# PenSim CSV layout (16 columns), documented here so this module has no
# dependency on the rest of the codebase:
#
#   0      Time Step
#   1-6    actions   [discharge, sugar, soilbean, aeration, pressure, water]
#   7-14   observations [pH, Temp, Fa, Fb, Fc, Fh, Wt, DO2]
#   15     Yield Per Step  (the reward)
# ---------------------------------------------------------------------------
TIME_COL = 0
ACT_COLS = slice(1, 7)
OBS_COLS = slice(7, 15)
REW_COL = 15

OBS_NAMES = ['pH', 'Temp', 'Fa', 'Fb', 'Fc', 'Fh', 'Wt', 'DO2']
ACT_NAMES = ['discharge', 'sugar', 'soilbean', 'aeration', 'pressure', 'water']


def set_seed(seed: int) -> None:
    """Seed python, numpy and torch. Called once at the top of training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN costs a little speed but makes runs comparable,
    # which matters more here than throughput on a model this small.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu')


@dataclass
class Standardizer:
    """
    z-score standardizer fit on the TRAINING split only.

    Fitting on the full dataset leaks validation statistics into training and
    makes the validation loss optimistic, so `fit` is only ever called on the
    training rows.
    """
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, X: np.ndarray, eps: float = 1e-8) -> 'Standardizer':
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        # A channel that never varies would divide by zero; leave it alone.
        std = np.where(std < eps, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std

    def inverse(self, Z: np.ndarray) -> np.ndarray:
        return Z * self.std + self.mean

    def to_dict(self) -> dict:
        return {'mean': self.mean.tolist(), 'std': self.std.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> 'Standardizer':
        return cls(mean=np.asarray(d['mean'], dtype=np.float64),
                   std=np.asarray(d['std'], dtype=np.float64))


class RunningMeter:
    """Accumulates a loss over a epoch without holding every batch value."""

    def __init__(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def average(self) -> float:
        return self.total / max(self.count, 1)


def save_checkpoint(path: str,
                    projection: torch.nn.Module,
                    reward_net: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None,
                    epoch: int,
                    val_loss: float,
                    config: dict,
                    standardizer: Standardizer | None = None) -> None:
    """
    torch.save rather than pickle: the payload holds numpy arrays, and pickle
    is fragile across numpy ABI changes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        'projection': projection.state_dict(),
        'reward_net': reward_net.state_dict(),
        'epoch': epoch,
        'val_loss': val_loss,
        'config': config,
    }
    if optimizer is not None:
        payload['optimizer'] = optimizer.state_dict()
    if standardizer is not None:
        payload['standardizer'] = standardizer.to_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, map_location: str | torch.device = 'cpu') -> dict:
    return torch.load(path, map_location=map_location)


def save_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, '__dataclass_fields__'):
        return asdict(o)
    raise TypeError(f'not JSON serialisable: {type(o)}')


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def format_matrix(M: np.ndarray,
                  row_names: Sequence[str] | None = None,
                  col_names: Sequence[str] | None = None,
                  width: int = 9,
                  precision: int = 3) -> str:
    """Pretty-print a small matrix (used for the learned projection)."""
    M = np.asarray(M)
    lines = []
    if col_names is not None:
        lines.append(' ' * 10 + ''.join(f'{c:>{width}s}' for c in col_names))
    for i, row in enumerate(M):
        label = row_names[i] if row_names is not None else f'z{i}'
        lines.append(f'{label:<10s}' +
                     ''.join(f'{v:{width}.{precision}f}' for v in row))
    return '\n'.join(lines)
