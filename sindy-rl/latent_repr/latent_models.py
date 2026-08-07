"""
Model components: a linear projection into a latent space, and an MLP that
predicts scalar reward from that latent state. Trained jointly so gradients
from the reward loss shape the projection.

The two are kept as separate modules rather than one Sequential so the
projection can be extracted, inspected, and reused on its own -- which is the
point of the exercise.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class LinearProjection(nn.Module):
    """
    z = W s + b

    A single linear layer, no activation. `W` has shape (latent_dim,
    state_dim) following torch's convention, so row i of `W` is the direction
    in state space that latent coordinate i measures.
    """

    def __init__(self,
                 state_dim: int,
                 latent_dim: int,
                 bias: bool = True,
                 init: str = 'orthogonal'):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.linear = nn.Linear(state_dim, latent_dim, bias=bias)
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        if init == 'orthogonal':
            # Orthogonal rows start the latent coordinates measuring
            # independent directions, which keeps early training from
            # collapsing several latents onto the same feature.
            nn.init.orthogonal_(self.linear.weight)
        elif init == 'xavier':
            nn.init.xavier_uniform_(self.linear.weight)
        elif init == 'normal':
            nn.init.normal_(self.linear.weight, std=0.1)
        else:
            raise ValueError(f'unknown init {init!r}')
        if self.linear.bias is not None:
            nn.init.zeros_(self.linear.bias)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.linear(s)

    # -- inspection helpers ------------------------------------------------

    @property
    def W(self) -> torch.Tensor:
        return self.linear.weight.detach()

    @property
    def b(self) -> torch.Tensor | None:
        return None if self.linear.bias is None else self.linear.bias.detach()

    def row_norms(self) -> torch.Tensor:
        """Norm of each latent direction. A near-zero row means that latent
        coordinate is unused and the effective dimensionality is lower than
        `latent_dim`."""
        return self.W.norm(dim=1)

    def effective_rank(self, tol: float = 1e-3) -> int:
        """Singular values of W above `tol` -- how many directions the
        projection actually spans, which can be less than latent_dim."""
        s = torch.linalg.svdvals(self.W)
        return int((s > tol * s.max()).sum().item())


class RewardPredictor(nn.Module):
    """
    MLP mapping latent state -> scalar reward.

    Default depth matches the brief: Linear-ReLU-Linear-ReLU-Linear.
    """

    def __init__(self,
                 latent_dim: int,
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = 'relu',
                 dropout: float = 0.0):
        super().__init__()
        act_cls = {'relu': nn.ReLU,
                   'tanh': nn.Tanh,
                   'gelu': nn.GELU,
                   'elu': nn.ELU}[activation]

        layers: list[nn.Module] = []
        prev = latent_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(act_cls())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))   # scalar output, no activation
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class LatentRewardModel(nn.Module):
    """
    Convenience wrapper holding both components.

    Training can use either this or the two modules separately; the separate
    form makes it easy to give the projection its own learning rate, which
    matters because a single linear layer and a 3-layer MLP do not generally
    want the same step size.
    """

    def __init__(self,
                 state_dim: int,
                 latent_dim: int,
                 hidden_sizes: Sequence[int] = (64, 64),
                 activation: str = 'relu',
                 dropout: float = 0.0,
                 bias: bool = True,
                 init: str = 'orthogonal'):
        super().__init__()
        self.projection = LinearProjection(state_dim, latent_dim,
                                           bias=bias, init=init)
        self.reward_net = RewardPredictor(latent_dim, hidden_sizes,
                                          activation=activation,
                                          dropout=dropout)

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.reward_net(self.projection(s))

    def encode(self, s: torch.Tensor) -> torch.Tensor:
        return self.projection(s)


def build_model(state_dim: int, config: dict) -> LatentRewardModel:
    """Construct from a plain dict so sweeps stay config-driven."""
    return LatentRewardModel(
        state_dim=state_dim,
        latent_dim=config.get('latent_dim', 4),
        hidden_sizes=tuple(config.get('hidden_sizes', (64, 64))),
        activation=config.get('activation', 'relu'),
        dropout=config.get('dropout', 0.0),
        bias=config.get('bias', True),
        init=config.get('init', 'orthogonal'),
    )
