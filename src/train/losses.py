# [M4] Multi-task loss: plaque (BCE) + echo (CE, ignore am) + risk (SmoothL1).
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTaskLoss(nn.Module):
    """
    Tong hop loss 3 task voi trong so.
    - plaque: BCEWithLogitsLoss(pos_weight) — xu ly lech lop 205/95.
    - echo:   CrossEntropyLoss(ignore_index=-100) — ca am (None) khong tinh.
    - risk:   SmoothL1Loss — hoi quy Baseline_Risk_Score.

    M4 TODO: thu uncertainty weighting (Kendall 2018) thay trong so co dinh.
    """

    def __init__(self, weights: dict, pos_weight: float | None = None):
        super().__init__()
        self.w = weights
        self.pos_weight_val = pos_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs: dict, labels: dict) -> tuple[torch.Tensor, dict]:
        if self.pos_weight_val is not None:
            pw = torch.tensor([self.pos_weight_val], device=outputs["plaque"].device)
            l_plaque = F.binary_cross_entropy_with_logits(
                outputs["plaque"], labels["plaque"], pos_weight=pw)
        else:
            l_plaque = self.bce(outputs["plaque"], labels["plaque"])
        l_echo = self.ce(outputs["echo"], labels["echo"].squeeze(1))
        l_risk = self.smooth_l1(outputs["risk"], labels["risk"])

        # Neu ca batch deu la ca am -> l_echo co the la NaN; thay bang 0.
        if torch.isnan(l_echo):
            l_echo = torch.zeros((), device=l_plaque.device)

        total = (self.w["plaque"] * l_plaque
                 + self.w["echo"] * l_echo
                 + self.w["risk"] * l_risk)
        return total, {
            "plaque": float(l_plaque.detach()),
            "echo": float(l_echo.detach()),
            "risk": float(l_risk.detach()),
            "total": float(total.detach()),
        }
