# [M2] Nhanh Tabular: MLP encode du lieu lam sang -> feature vector.
from __future__ import annotations

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """
    Input: [B, in_dim] (8 numeric da scale + Sex = 9 feature).
    Output: [B, feat_dim] feature vector cho lop Fusion.

    M2 TODO:
      - Thu nghiem so lop/hidden size, BatchNorm vs LayerNorm.
      - Co the them embedding rieng cho Sex neu can.
    """

    def __init__(self, in_dim: int = 9, hidden=(64, 32), feat_dim: int = 32,
                 dropout: float = 0.3):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], feat_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class TabularClassifier(nn.Module):
    """Baseline tabular-only (plaque) — MLP + 1 logit. Dung lam mốc so sanh."""

    def __init__(self, in_dim: int = 9, hidden=(64, 32), dropout: float = 0.3):
        super().__init__()
        self.encoder = TabularMLP(in_dim, hidden, hidden[-1], dropout)
        self.classifier = nn.Linear(hidden[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(x))  # [B,1] logit
