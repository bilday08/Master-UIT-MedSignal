# [M3] Nhanh Vision: CNN encoder cho anh xam 256x256 + Attention pooling cho CCA.
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """
    Encode 1 anh grayscale [B,1,256,256] -> feature [B,feat_dim].
    Mac dinh ResNet-18 sua input 1 kenh; tuy chon Custom CNN nhe.

    M3 TODO:
      - Augmentation manh (flip/rotate/jitter) — dat o transform cua Dataset.
      - Thu MobileNetV3 neu can nhe hon. pretrained=False vi anh gia lap.
    """

    def __init__(self, encoder: str = "resnet18", feat_dim: int = 128,
                 pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        if encoder == "resnet18":
            from torchvision.models import resnet18
            net = resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            net.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            in_feat = net.fc.in_features
            net.fc = nn.Identity()
            self.backbone = net
            self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_feat, feat_dim))
        elif encoder == "custom_cnn":
            self.backbone = nn.Sequential(
                nn.Conv2d(1, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.ReLU(),
                nn.Conv2d(16, 32, 3, 2, 1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 64, 3, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(64, feat_dim))
        else:
            raise ValueError(f"encoder khong ho tro: {encoder}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class AttentionPool(nn.Module):
    """
    Gated attention pooling (Ilse et al. 2018) gop K feature anh -> 1 vector, CO mask.
    Input:  feats [B,K,D], mask [B,K] (True=anh that)
    Output: pooled [B,D]
    Ca Control (K=0 het mask) -> tra ve vector 0 (khong dong gop vao head plaque).

    M3 TODO: thu thay bang mean/max pooling de ablation.
    """

    def __init__(self, dim: int = 128, hidden: int = 64):
        super().__init__()
        self.attn_V = nn.Linear(dim, hidden)
        self.attn_U = nn.Linear(dim, hidden)
        self.attn_w = nn.Linear(hidden, 1)

    def forward(self, feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a = torch.tanh(self.attn_V(feats)) * torch.sigmoid(self.attn_U(feats))  # [B,K,H]
        scores = self.attn_w(a).squeeze(-1)                                     # [B,K]
        scores = scores.masked_fill(~mask, float("-inf"))
        # Neu toan bo bi mask (Control) -> tranh NaN: tra vector 0.
        all_masked = (~mask).all(dim=1, keepdim=True)                           # [B,1]
        weights = torch.softmax(scores, dim=1)
        weights = torch.nan_to_num(weights, nan=0.0)
        pooled = torch.bmm(weights.unsqueeze(1), feats).squeeze(1)              # [B,D]
        pooled = pooled.masked_fill(all_masked, 0.0)
        return pooled


class VisionBranch(nn.Module):
    """
    Bao tron nhanh Vision: IMT encoder (cho plaque) + CCA encoder+pool (cho echo).
    Tra ve (imt_feat [B,D], cca_feat [B,D]).
    """

    def __init__(self, encoder="resnet18", feat_dim=128, pretrained=False, dropout=0.3):
        super().__init__()
        self.imt_encoder = ImageEncoder(encoder, feat_dim, pretrained, dropout)
        self.cca_encoder = ImageEncoder(encoder, feat_dim, pretrained, dropout)
        self.cca_pool = AttentionPool(feat_dim)

    def forward(self, imt_img, cca_imgs, cca_mask):
        imt_feat = self.imt_encoder(imt_img)                 # [B,D]
        B, K = cca_imgs.shape[0], cca_imgs.shape[1]
        flat = cca_imgs.view(B * K, *cca_imgs.shape[2:])      # [B*K,1,H,W]
        cca_feats = self.cca_encoder(flat).view(B, K, -1)     # [B,K,D]
        cca_feat = self.cca_pool(cca_feats, cca_mask)         # [B,D]
        return imt_feat, cca_feat
