"""Cau hinh + tien xu ly dung chung cho freeze_model va inference.

Tach rieng de freeze (train) va inference (serve) dung CHINH XAC cung:
  - cau hinh v3 (override len base config)
  - val-transform anh (KHONG augment)
  - cach dung vector tabular (encode Sex + scale numeric, dung thu tu feature)

Giu m5/serving doc lap voi script m4_fusion/v3_focal_loss/train.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # m5/serving/common.py -> repo root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P  # noqa: E402

ARTIFACT_DIR = PROJECT_ROOT / "m5" / "serving" / "artifacts"
IMAGE_NORM_MEAN = [0.5]
IMAGE_NORM_STD = [0.5]


def load_v3_config() -> dict:
    """Base config + override v3 (giong main() trong m4_fusion/v3_focal_loss/train.py)."""
    cfg = P.load_config(str(PROJECT_ROOT / "configs" / "config.yaml"))
    cfg["train"]["pos_weight_scale"] = 1.2
    cfg["train"]["batch_size"] = 32
    cfg["train"]["patience"] = 7
    cfg["train"]["focal_gamma"] = 2.0
    return cfg


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_transforms(train: bool, image_size: int):
    """Mirror v3 make_transforms. train=False -> val-transform dung cho inference."""
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGE_NORM_MEAN, std=IMAGE_NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGE_NORM_MEAN, std=IMAGE_NORM_STD),
    ])


class FocalMultiTaskLoss(torch.nn.Module):
    """Focal Loss (plaque) + CE ignore_index (echo) + SmoothL1 (risk). Copy tu v3."""

    def __init__(self, weights: dict, pos_weight: float | None = None, gamma: float = 2.0):
        super().__init__()
        self.w = weights
        self.pos_weight_val = pos_weight
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.smooth_l1 = torch.nn.SmoothL1Loss()

    def forward(self, outputs: dict, labels: dict):
        logits = outputs["plaque"]
        targets = labels["plaque"]
        pw = (torch.tensor([self.pos_weight_val], device=logits.device)
              if self.pos_weight_val else None)
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        prob = torch.sigmoid(logits)
        p_t = prob * targets + (1 - prob) * (1 - targets)
        l_plaque = ((1 - p_t) ** self.gamma * bce).mean()

        l_echo = self.ce(outputs["echo"], labels["echo"].squeeze(1))
        l_risk = self.smooth_l1(outputs["risk"], labels["risk"])
        if torch.isnan(l_echo):
            l_echo = torch.zeros((), device=l_plaque.device)

        total = self.w["plaque"] * l_plaque + self.w["echo"] * l_echo + self.w["risk"] * l_risk
        return total, {"plaque": float(l_plaque.detach()), "echo": float(l_echo.detach()),
                       "risk": float(l_risk.detach()), "total": float(total.detach())}


def youden_threshold(y_true, y_prob) -> float:
    """Nguong toi uu Youden (argmax tpr - fpr) tren ROC."""
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = int((tpr - fpr).argmax())
    return float(thresholds[best_idx])
