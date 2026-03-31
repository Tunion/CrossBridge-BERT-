from __future__ import annotations

import torch
from torch import nn


class EnergyHeadMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.10):
        super().__init__()
        hd = int(max(8, hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(int(input_dim), hd),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hd, hd),
            nn.ReLU(),
            nn.Dropout(float(max(0.0, dropout))),
            nn.Linear(hd, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
