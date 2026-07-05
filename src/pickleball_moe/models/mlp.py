from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class DenseMLP(nn.Module):
    """Dense MLP baseline for tabular four-class shot classification."""

    def __init__(
        self,
        d_in: int,
        n_classes: int = 4,
        hidden_dims: Sequence[int] = (128, 64, 32),
        dropout: float | Sequence[float] = (0.2, 0.2, 0.1),
        batch_norm: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(dropout, (float, int)):
            dropout_values = [float(dropout)] * len(hidden_dims)
        else:
            dropout_values = list(dropout)
        if len(dropout_values) != len(hidden_dims):
            raise ValueError("dropout must be a scalar or match hidden_dims length")

        layers: list[nn.Module] = []
        prev = d_in
        for i, hidden in enumerate(hidden_dims):
            layers.append(nn.Linear(prev, hidden))
            if batch_norm and i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(nn.ReLU())
            if dropout_values[i] > 0:
                layers.append(nn.Dropout(dropout_values[i]))
            prev = hidden
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def active_parameters_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
