# [M4] Multimodal Fusion + 3 multi-task heads.
from __future__ import annotations

import torch
import torch.nn as nn

from .tabular import TabularMLP
from .vision import VisionBranch


class MultimodalFusion(nn.Module):
    """
    Fusion: concat(tab_feat, imt_feat) [+ cca_feat tuy chon] -> joint -> 3 head.

    Chong leakage: head PLAQUE chi nhin (tab_feat, imt_feat).
                   cca_feat (tu CCA, chi ca duong co) -> chi vao head ECHO.

    Heads:
      - plaque: 1 logit (BCEWithLogits + pos_weight)
      - echo:   3 logit (CrossEntropy, ignore_index=-100 cho ca am)
      - risk:   1 gia tri (SmoothL1 regression)

    M4 TODO:
      - Thu bilinear fusion thay concat.
      - use_cca_in_fusion=True neu muon CCA gop vao joint chung (can than leakage).
    """

    def __init__(self, cfg: dict, in_tab: int = 9):
        super().__init__()
        tdim = cfg["tabular"]["feat_dim"]
        vdim = cfg["vision"]["feat_dim"]
        self.use_cca_in_fusion = cfg["fusion"].get("use_cca_in_fusion", False)

        self.tabular = TabularMLP(
            in_dim=in_tab, hidden=tuple(cfg["tabular"]["hidden"]),
            feat_dim=tdim, dropout=cfg["train"]["dropout"],
        )
        self.vision = VisionBranch(
            encoder=cfg["vision"]["encoder"], feat_dim=vdim,
            pretrained=cfg["vision"]["pretrained"], dropout=cfg["train"]["dropout"],
        )

        # Joint cho task plaque/risk: tabular + IMT (KHONG co CCA -> chong leakage).
        joint_in = tdim + vdim + (vdim if self.use_cca_in_fusion else 0)
        self.joint = nn.Sequential(
            nn.Linear(joint_in, 64), nn.ReLU(), nn.Dropout(cfg["train"]["dropout"]),
        )
        self.head_plaque = nn.Linear(64, 1)
        self.head_risk = nn.Linear(64, 1)
        # Head echo nhin them cca_feat (dac trung mang xo vua).
        self.head_echo = nn.Linear(64 + vdim, 3)

    def forward(self, tabular, imt_img, cca_imgs, cca_mask):
        tab_feat = self.tabular(tabular)
        imt_feat, cca_feat = self.vision(imt_img, cca_imgs, cca_mask)

        parts = [tab_feat, imt_feat]
        if self.use_cca_in_fusion:
            parts.append(cca_feat)
        joint = self.joint(torch.cat(parts, dim=1))

        return {
            "plaque": self.head_plaque(joint),                                   # [B,1]
            "risk": self.head_risk(joint),                                       # [B,1]
            "echo": self.head_echo(torch.cat([joint, cca_feat], dim=1)),         # [B,3]
        }
