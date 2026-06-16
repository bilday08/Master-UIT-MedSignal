"""M4 Fusion v4 — v2 + Uncertainty Weighting (Kendall et al. 2018)."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
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
from src.models.fusion import MultimodalFusion


class UncertaintyMultiTaskLoss(nn.Module):
    """
    Kendall et al. 2018: học log_var cho mỗi task.
    L_total = sum_i [ L_i / (2*exp(log_var_i)) + 0.5*log_var_i ]
    - Classification tasks: dùng 2*sigma^2
    - Regression task: dùng sigma^2
    """
    def __init__(self, pos_weight: float | None = None):
        super().__init__()
        self.log_var_plaque = nn.Parameter(torch.zeros(1))
        self.log_var_echo   = nn.Parameter(torch.zeros(1))
        self.log_var_risk   = nn.Parameter(torch.zeros(1))
        self.pos_weight_val = pos_weight
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, outputs: dict, labels: dict):
        pw = torch.tensor([self.pos_weight_val], device=outputs["plaque"].device) \
             if self.pos_weight_val else None
        l_plaque = F.binary_cross_entropy_with_logits(
            outputs["plaque"], labels["plaque"], pos_weight=pw)
        l_echo = self.ce(outputs["echo"], labels["echo"].squeeze(1))
        l_risk = self.smooth_l1(outputs["risk"], labels["risk"])
        if torch.isnan(l_echo):
            l_echo = torch.zeros((), device=l_plaque.device)

        # uncertainty weighting
        prec_p = torch.exp(-self.log_var_plaque)
        prec_e = torch.exp(-self.log_var_echo)
        prec_r = torch.exp(-self.log_var_risk)
        total = (prec_p * l_plaque + 0.5 * self.log_var_plaque +
                 prec_e * l_echo   + 0.5 * self.log_var_echo +
                 prec_r * l_risk   + 0.5 * self.log_var_risk)
        return total, {
            "plaque": float(l_plaque.detach()), "echo": float(l_echo.detach()),
            "risk": float(l_risk.detach()), "total": float(total.detach()),
            "log_var_plaque": float(self.log_var_plaque.detach()),
            "log_var_echo":   float(self.log_var_echo.detach()),
            "log_var_risk":   float(self.log_var_risk.detach()),
        }


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


def find_best_threshold(y_true, y_prob):
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    best_idx = np.argmax(tpr - fpr)
    return float(thresholds[best_idx])


def make_weighted_sampler(df_train, cfg):
    labels = df_train[cfg["columns"]["target_plaque"]].values.astype(int)
    n_pos = labels.sum(); n_neg = len(labels) - n_pos
    weights = np.where(labels == 1, len(labels)/(2*n_pos), len(labels)/(2*n_neg))
    return WeightedRandomSampler(torch.from_numpy(weights).float(), len(weights), replacement=True)


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
    m.update(risk_metrics(torch.cat(all_rt).squeeze(1).numpy(), torch.cat(all_rp).squeeze(1).numpy()))
    m["val_loss"] = round(total_loss / len(dl_va), 4)
    # log learned weights
    lv = [float(criterion.log_var_plaque.detach()),
          float(criterion.log_var_echo.detach()),
          float(criterion.log_var_risk.detach())]
    m["learned_log_var"] = lv
    return m


def train_one_fold(df, train_idx, val_idx, cfg, fold_id, device):
    df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]
    scaler_tab = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
    img_size = cfg["data"]["image_size"]

    ds_tr = CarotidDataset(df_tr, cfg, scaler_tab, str(PROJECT_ROOT),
                           transform=make_transforms(True, img_size),
                           cca_transform=make_transforms(True, img_size))
    ds_va = CarotidDataset(df_va, cfg, scaler_tab, str(PROJECT_ROOT),
                           transform=make_transforms(False, img_size),
                           cca_transform=make_transforms(False, img_size))
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"],
                       sampler=make_weighted_sampler(df_tr, cfg),
                       collate_fn=collate_fn, num_workers=4)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"],
                       shuffle=False, collate_fn=collate_fn, num_workers=4)

    model = MultimodalFusion(cfg, in_tab=len(P.feature_columns(cfg))).to(device)
    scale = cfg["train"].get("pos_weight_scale", 1.2)
    pos_weight = P.compute_pos_weight(df_tr, cfg) * scale if cfg["train"]["pos_weight_auto"] else None
    criterion = UncertaintyMultiTaskLoss(pos_weight).to(device)

    # uncertainty params cùng lr với model
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    use_amp = str(device) == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)

    patience = cfg["train"].get("patience", 7)
    best_pr_auc, patience_counter, best_metrics = -1.0, 0, {}
    ckpt_dir = PROJECT_ROOT / "m4_fusion/v4_uncertainty/checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["train"]["epochs"]):
        model.train(); criterion.train(); train_loss = 0.0
        for batch in dl_tr:
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                            batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                loss, _ = criterion(out, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer); grad_scaler.update()
            train_loss += float(loss.detach())

        val_m = validate(model, dl_va, criterion, device)
        pr_auc = val_m.get("pr_auc", 0.0)
        lv = val_m.get("learned_log_var", [0,0,0])
        print(f"  Fold {fold_id} | Ep {epoch+1:>3} | loss {train_loss/len(dl_tr):.4f} "
              f"| PR-AUC {pr_auc:.4f} | AUC {val_m.get('auc_roc',0):.4f} "
              f"| F1 {val_m.get('f1',0):.4f} | Sens {val_m.get('sensitivity',0):.4f} "
              f"| Spec {val_m.get('specificity',0):.4f} | thr {val_m.get('best_threshold',0):.3f} "
              f"| logvar [{lv[0]:.2f},{lv[1]:.2f},{lv[2]:.2f}]")

        if pr_auc > best_pr_auc:
            best_pr_auc, best_metrics, patience_counter = pr_auc, val_m.copy(), 0
            torch.save({"fold": fold_id, "epoch": epoch, "model_state": model.state_dict(),
                        "criterion_state": criterion.state_dict(),
                        "metrics": best_metrics}, ckpt_dir / f"fold{fold_id}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop ep {epoch+1}"); break

    print(f"  Fold {fold_id} best PR-AUC: {best_pr_auc:.4f}")
    return best_metrics


def main():
    cfg = P.load_config(str(PROJECT_ROOT / "configs/config.yaml"))
    cfg["train"]["pos_weight_scale"] = 1.2
    cfg["train"]["batch_size"] = 32
    cfg["train"]["patience"] = 7

    df = P.load_dataframe(cfg, str(PROJECT_ROOT))
    folds = stratified_folds(df, cfg)
    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"Device: {device} | v4: UncertaintyWeighting + WeightedSampler + OptimalThreshold")

    fold_metrics = []
    for i, (tr, va) in enumerate(folds):
        print(f"\n=== Fold {i} ===")
        fold_metrics.append(train_one_fold(df, tr, va, cfg, i, device))

    agg = aggregate_folds(fold_metrics)
    print("\n=== v4 Results (mean ± std) ===")
    for k, v in agg.items():
        print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")

    result = {"version": "v4_uncertainty",
              "description": "v2 + Uncertainty Weighting Kendall 2018 (learnable log_var per task)",
              "summary": agg, "fold_results": fold_metrics}
    out = PROJECT_ROOT / "m4_fusion/v4_uncertainty/results.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
