"""
M4 Fusion v5 — All improvements combined:
  1. custom_cnn encoder (64K vs 22.5M params)
  2. tabular.feat_dim 32→128 (balance modalities)
  3. CrossAttention fusion (tab↔image)
  4. Focal Loss γ=2.0
  5. WeightedRandomSampler
  6. SMOTE on tabular training fold
  7. CosineAnnealingLR scheduler
  8. Gradient clipping max_norm=1.0
  9. Youden threshold per fold
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.eval.metrics import aggregate_folds, classification_metrics, echo_metrics, risk_metrics
from src.models.vision import ImageEncoder, AttentionPool


# ─────────────────────────── Models ────────────────────────────

class TabularMLP(nn.Module):
    """Tab encoder: 9 features → 128d."""
    def __init__(self, in_dim=9, hidden=(64, 128), feat_dim=128, dropout=0.3):
        super().__init__()
        dims = [in_dim, *hidden]
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], feat_dim)

    def forward(self, x):
        return self.head(self.backbone(x))


class CrossAttentionFusion(nn.Module):
    """
    Tab và image attend lẫn nhau, sau đó concat + residual.
    tab_feat [B,D], imt_feat [B,D] → fused [B, 2D]
    """
    def __init__(self, dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.tab_to_img = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.img_to_tab = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_tab = nn.LayerNorm(dim)
        self.norm_img = nn.LayerNorm(dim)

    def forward(self, tab_feat, imt_feat):
        t = tab_feat.unsqueeze(1)   # [B,1,D]
        v = imt_feat.unsqueeze(1)   # [B,1,D]
        t_out, _ = self.tab_to_img(t, v, v)
        v_out, _ = self.img_to_tab(v, t, t)
        tab_out = self.norm_tab(tab_feat + t_out.squeeze(1))   # residual
        img_out = self.norm_img(imt_feat + v_out.squeeze(1))   # residual
        return torch.cat([tab_out, img_out], dim=1)            # [B, 2D]


class MultimodalFusionV5(nn.Module):
    """
    v5 fusion:
      tabular (9→128) + vision custom_cnn (128) → CrossAttention → 256d → joint 128d → 3 heads
      head_echo: joint + cca_feat (chống leakage: plaque/risk không nhìn CCA)
    """
    def __init__(self, cfg: dict, in_tab=9):
        super().__init__()
        D = 128
        dropout = cfg["train"]["dropout"]

        self.tabular = TabularMLP(in_dim=in_tab, hidden=(64, 128), feat_dim=D, dropout=dropout)
        self.imt_encoder = ImageEncoder("custom_cnn", feat_dim=D, pretrained=False, dropout=dropout)
        self.cca_encoder = ImageEncoder("custom_cnn", feat_dim=D, pretrained=False, dropout=dropout)
        self.cca_pool = AttentionPool(D)

        self.cross_attn = CrossAttentionFusion(dim=D, num_heads=4, dropout=0.1)

        self.joint = nn.Sequential(
            nn.Linear(2 * D, D), nn.LayerNorm(D), nn.ReLU(), nn.Dropout(dropout),
        )
        self.head_plaque = nn.Linear(D, 1)
        self.head_risk   = nn.Linear(D, 1)
        self.head_echo   = nn.Linear(D + D, 3)   # joint + cca_feat

    def forward(self, tabular, imt_img, cca_imgs, cca_mask):
        tab_feat = self.tabular(tabular)
        imt_feat = self.imt_encoder(imt_img)

        B, K = cca_imgs.shape[:2]
        flat = cca_imgs.view(B * K, *cca_imgs.shape[2:])
        cca_feat = self.cca_pool(
            self.cca_encoder(flat).view(B, K, -1), cca_mask
        )

        fused = self.cross_attn(tab_feat, imt_feat)   # [B, 256]
        joint = self.joint(fused)                      # [B, 128]

        return {
            "plaque": self.head_plaque(joint),
            "risk":   self.head_risk(joint),
            "echo":   self.head_echo(torch.cat([joint, cca_feat], dim=1)),
        }


# ─────────────────────────── Loss ───────────────────────────────

class FocalMultiTaskLoss(nn.Module):
    def __init__(self, weights, pos_weight=None, gamma=2.0):
        super().__init__()
        self.w = weights
        self.pos_weight_val = pos_weight
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs, labels):
        logits, targets = outputs["plaque"], labels["plaque"]
        pw = torch.tensor([self.pos_weight_val], device=logits.device) if self.pos_weight_val else None
        bce = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw, reduction="none")
        p_t = torch.sigmoid(logits) * targets + (1 - torch.sigmoid(logits)) * (1 - targets)
        l_plaque = ((1 - p_t) ** self.gamma * bce).mean()

        l_echo = self.ce(outputs["echo"], labels["echo"].squeeze(1))
        l_risk  = self.smooth_l1(outputs["risk"], labels["risk"])
        if torch.isnan(l_echo):
            l_echo = torch.zeros((), device=l_plaque.device)

        total = self.w["plaque"] * l_plaque + self.w["echo"] * l_echo + self.w["risk"] * l_risk
        return total, {"plaque": float(l_plaque.detach()), "echo": float(l_echo.detach()),
                       "risk": float(l_risk.detach()), "total": float(total.detach())}


# ─────────────────────────── Data helpers ───────────────────────

def make_transforms(train: bool, image_size: int):
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def apply_smote(df_train: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    SMOTE trên tabular features của training fold.
    Synthetic samples ghép với ảnh của positive cases gần nhất.
    """
    from imblearn.over_sampling import SMOTE

    feat_cols = P.feature_columns(cfg)
    target_col = cfg["columns"]["target_plaque"]
    df_enc = P.encode_categorical(df_train, cfg)

    X = df_enc[feat_cols].values.astype(float)
    y = df_enc[target_col].values.astype(int)

    n_pos = int((y == 1).sum())
    if n_pos < 2:
        return df_train

    k = min(5, n_pos - 1)
    smote = SMOTE(k_neighbors=k, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    n_orig = len(df_train)
    n_synth = len(X_res) - n_orig
    if n_synth == 0:
        return df_train

    # Synthetic rows: copy non-tabular columns từ random positive samples
    pos_df = df_train[df_train[target_col] == 1]
    synth = pos_df.sample(n_synth, replace=True, random_state=42).reset_index(drop=True)

    # Overwrite tabular values bằng SMOTE-generated values
    # Dùng numeric cols (chưa encoded) — lấy giá trị raw và denormalize sau
    # Thực tế: đặt giá trị SMOTE vào numeric cols của df gốc (chưa scale)
    numeric_cols = cfg["columns"]["numeric"]
    sex_col = "Sex"

    # Map encoded features back: feat_cols = numeric + [Sex]
    synth_tab = X_res[n_orig:]
    for j, col in enumerate(feat_cols):
        if col == sex_col:
            synth[col] = np.where(synth_tab[:, j] >= 0.5, "Male", "Female")
        else:
            synth[col] = synth_tab[:, j]

    synth[target_col] = 1  # tất cả synthetic là positive
    df_aug = pd.concat([df_train, synth], ignore_index=True)
    print(f"    SMOTE: {n_orig} → {len(df_aug)} samples (+{n_synth} synthetic positives)")
    return df_aug


def make_weighted_sampler(df_train, cfg):
    labels = df_train[cfg["columns"]["target_plaque"]].values.astype(int)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    weights = np.where(labels == 1, len(labels) / (2 * n_pos), len(labels) / (2 * n_neg))
    return WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights), replacement=True)


def find_best_threshold(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx])


# ─────────────────────────── Training ───────────────────────────

def validate(model, dl_va, criterion, device):
    model.eval()
    all_pp, all_pt, all_ep, all_et, all_rp, all_rt = [], [], [], [], [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in dl_va:
            out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                        batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            loss, _ = criterion(out, labels)
            total_loss += float(loss.detach())
            all_pp.append(torch.sigmoid(out["plaque"]).cpu())
            all_pt.append(labels["plaque"].cpu())
            all_ep.append(out["echo"].argmax(dim=1).cpu())
            all_et.append(labels["echo"].squeeze(1).cpu())
            all_rp.append(out["risk"].cpu())
            all_rt.append(labels["risk"].cpu())

    prob = torch.cat(all_pp).squeeze(1).numpy()
    true = torch.cat(all_pt).squeeze(1).numpy()
    best_t = find_best_threshold(true, prob)
    m = classification_metrics(true, prob, threshold=best_t)
    m["best_threshold"] = round(best_t, 4)
    m.update(echo_metrics(torch.cat(all_et).numpy(), torch.cat(all_ep).numpy()))
    m.update(risk_metrics(torch.cat(all_rt).squeeze(1).numpy(),
                           torch.cat(all_rp).squeeze(1).numpy()))
    m["val_loss"] = round(total_loss / len(dl_va), 4)
    return m


def train_one_fold(df, train_idx, val_idx, cfg, fold_id, device):
    df_tr_orig = df.iloc[train_idx]
    df_va = df.iloc[val_idx]

    # SMOTE augmentation trên training fold
    df_tr = apply_smote(df_tr_orig, cfg)

    scaler_tab = P.fit_scaler(P.encode_categorical(df_tr_orig, cfg), cfg)  # fit trên gốc, không SMOTE
    img_size = cfg["data"]["image_size"]

    ds_tr = CarotidDataset(df_tr, cfg, scaler_tab, str(PROJECT_ROOT),
                           transform=make_transforms(True, img_size),
                           cca_transform=make_transforms(True, img_size))
    ds_va = CarotidDataset(df_va, cfg, scaler_tab, str(PROJECT_ROOT),
                           transform=make_transforms(False, img_size),
                           cca_transform=make_transforms(False, img_size))

    sampler = make_weighted_sampler(df_tr, cfg)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"],
                       sampler=sampler, collate_fn=collate_fn, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"],
                       shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = MultimodalFusionV5(cfg, in_tab=len(P.feature_columns(cfg))).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"    Model params: {total_params:,}")

    scale = cfg["train"].get("pos_weight_scale", 1.2)
    pw = P.compute_pos_weight(df_tr_orig, cfg) * scale if cfg["train"]["pos_weight_auto"] else None
    criterion = FocalMultiTaskLoss(
        weights={"plaque": 1.0, "echo": 0.5, "risk": 0.5},
        pos_weight=pw,
        gamma=cfg["train"].get("focal_gamma", 2.0),
    )
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=cfg["train"]["lr"],
                                   weight_decay=cfg["train"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["train"]["epochs"], eta_min=1e-5)

    use_amp = str(device) == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)
    grad_clip = cfg["train"].get("grad_clip_norm", 1.0)

    patience = cfg["train"].get("patience", 10)
    best_pr_auc, patience_counter, best_metrics = -1.0, 0, {}
    ckpt_dir = PROJECT_ROOT / "m4_fusion/v5_all_improvements/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in dl_tr:
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                            batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                loss, _ = criterion(out, labels)
            grad_scaler.scale(loss).backward()
            # Gradient clipping
            grad_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += float(loss.detach())

        scheduler.step()
        val_m = validate(model, dl_va, criterion, device)
        pr_auc = val_m.get("pr_auc", 0.0)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"  Fold {fold_id} | Ep {epoch+1:>3} | loss {train_loss/len(dl_tr):.4f} "
              f"| PR-AUC {pr_auc:.4f} | AUC {val_m.get('auc_roc',0):.4f} "
              f"| F1 {val_m.get('f1',0):.4f} | Sens {val_m.get('sensitivity',0):.4f} "
              f"| Spec {val_m.get('specificity',0):.4f} | thr {val_m.get('best_threshold',0):.3f}"
              f"| lr {lr_now:.2e}")

        if pr_auc > best_pr_auc:
            best_pr_auc, best_metrics, patience_counter = pr_auc, val_m.copy(), 0
            torch.save({"fold": fold_id, "epoch": epoch,
                        "model_state": model.state_dict(),
                        "metrics": best_metrics},
                       ckpt_dir / f"fold{fold_id}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop ep {epoch+1}")
                break

    print(f"  Fold {fold_id} best PR-AUC: {best_pr_auc:.4f}")
    return best_metrics


# ─────────────────────────── Main ───────────────────────────────

def main():
    cfg = P.load_config(str(PROJECT_ROOT / "configs/config.yaml"))
    # Override với v5 settings
    cfg["train"].update({
        "pos_weight_scale": 1.2,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "focal_gamma": 2.0,
        "grad_clip_norm": 1.0,
    })
    cfg["vision"]["encoder"] = "custom_cnn"
    cfg["tabular"]["feat_dim"] = 128

    df = P.load_dataframe(cfg, str(PROJECT_ROOT))
    folds = stratified_folds(df, cfg)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device}")
    print("v5: custom_cnn + tab_dim=128 + CrossAttn + SMOTE + CosLR + GradClip + FocalLoss")

    fold_metrics = []
    for i, (tr, va) in enumerate(folds):
        print(f"\n=== Fold {i} ===")
        fold_metrics.append(train_one_fold(df, tr, va, cfg, i, device))

    agg = aggregate_folds(fold_metrics)
    print("\n=== v5 Results (mean ± std) ===")
    for k, v in agg.items():
        print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")

    result = {
        "version": "v5_all_improvements",
        "description": "custom_cnn + tab_dim=128 + CrossAttention + SMOTE + CosLR + GradClip + FocalLoss",
        "summary": agg,
        "fold_results": fold_metrics,
    }
    out = PROJECT_ROOT / "m4_fusion/v5_all_improvements/results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
