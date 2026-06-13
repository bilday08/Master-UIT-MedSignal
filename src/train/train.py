# [M4] Vong train multimodal multi-task (5-fold).
from __future__ import annotations

import argparse
import os

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision import transforms

from ..data import preprocess as P
from ..data.dataset import CarotidDataset, collate_fn
from ..data.splits import stratified_folds
from ..eval.metrics import aggregate_folds, classification_metrics, echo_metrics, risk_metrics
from ..models.fusion import MultimodalFusion
from .losses import MultiTaskLoss


def make_transforms(train: bool, image_size: int):
    """Augmentation pipeline lay tu M3. Train: flip/rotate/jitter. Val: chi resize+normalize."""
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


def validate(model, dl_va, criterion, device, threshold: float = 0.5):
    model.eval()
    all_plaque_prob, all_plaque_true = [], []
    all_echo_pred, all_echo_true = [], []
    all_risk_pred, all_risk_true = [], []
    total_loss = 0.0

    with torch.no_grad():
        for batch in dl_va:
            out = model(
                batch["tabular"].to(device),
                batch["imt_img"].to(device),
                batch["cca_imgs"].to(device),
                batch["cca_mask"].to(device),
            )
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            loss, _ = criterion(out, labels)
            total_loss += float(loss)

            all_plaque_prob.append(torch.sigmoid(out["plaque"]).cpu())
            all_plaque_true.append(labels["plaque"].cpu())
            all_echo_pred.append(out["echo"].argmax(dim=1).cpu())
            all_echo_true.append(labels["echo"].squeeze(1).cpu())
            all_risk_pred.append(out["risk"].cpu())
            all_risk_true.append(labels["risk"].cpu())

    plaque_prob = torch.cat(all_plaque_prob).squeeze(1).numpy()
    plaque_true = torch.cat(all_plaque_true).squeeze(1).numpy()
    echo_pred = torch.cat(all_echo_pred).numpy()
    echo_true = torch.cat(all_echo_true).numpy()
    risk_pred = torch.cat(all_risk_pred).squeeze(1).numpy()
    risk_true = torch.cat(all_risk_true).squeeze(1).numpy()

    m = classification_metrics(plaque_true, plaque_prob, threshold=threshold)
    m.update(echo_metrics(echo_true, echo_pred))
    m.update(risk_metrics(risk_true, risk_pred))
    m["val_loss"] = round(total_loss / len(dl_va), 4)
    return m


def train_one_fold(df, train_idx, val_idx, cfg, fold_id=0, project_root=".", device="cpu"):
    df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]

    scaler_tab = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
    img_size = cfg["data"]["image_size"]
    threshold = cfg["eval"].get("decision_threshold", 0.3)

    ds_tr = CarotidDataset(df_tr, cfg, scaler_tab, project_root,
                           transform=make_transforms(True, img_size),
                           cca_transform=make_transforms(True, img_size))
    ds_va = CarotidDataset(df_va, cfg, scaler_tab, project_root,
                           transform=make_transforms(False, img_size),
                           cca_transform=make_transforms(False, img_size))
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       collate_fn=collate_fn, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       collate_fn=collate_fn, num_workers=0)

    in_tab = len(P.feature_columns(cfg))
    model = MultimodalFusion(cfg, in_tab=in_tab).to(device)

    # pos_weight nhan them scale de force model du doan lop duong tot hon
    pos_weight = None
    if cfg["train"]["pos_weight_auto"]:
        scale = cfg["train"].get("pos_weight_scale", 1.0)
        pos_weight = P.compute_pos_weight(df_tr, cfg) * scale

    criterion = MultiTaskLoss(cfg["train"]["loss_weights"], pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    use_amp = str(device) == "cuda"
    grad_scaler = GradScaler(enabled=use_amp)

    patience = cfg["train"].get("patience", 7)
    best_pr_auc = -1.0
    patience_counter = 0
    best_metrics = {}

    ckpt_dir = os.path.join(project_root, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"fold{fold_id}_best.pt")

    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        train_loss = 0.0
        for batch in dl_tr:
            optimizer.zero_grad()
            with autocast(enabled=use_amp):
                out = model(
                    batch["tabular"].to(device),
                    batch["imt_img"].to(device),
                    batch["cca_imgs"].to(device),
                    batch["cca_mask"].to(device),
                )
                labels = {k: v.to(device) for k, v in batch["labels"].items()}
                loss, _ = criterion(out, labels)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss += float(loss)

        val_m = validate(model, dl_va, criterion, device, threshold=threshold)
        pr_auc = val_m.get("pr_auc", 0.0)

        print(
            f"  Fold {fold_id} | Epoch {epoch+1:>3} "
            f"| train_loss {train_loss/len(dl_tr):.4f} "
            f"| val_loss {val_m['val_loss']:.4f} "
            f"| PR-AUC {pr_auc:.4f} "
            f"| F1 {val_m.get('f1', float('nan')):.4f} "
            f"| Sens {val_m.get('sensitivity', float('nan')):.4f}"
        )

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_metrics = val_m.copy()
            patience_counter = 0
            torch.save({
                "fold": fold_id,
                "epoch": epoch,
                "model_state": model.state_dict(),
                "scaler": scaler_tab,
                "cfg": cfg,
                "metrics": best_metrics,
            }, ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1} (patience={patience})")
                break

    print(f"  Fold {fold_id} best PR-AUC: {best_pr_auc:.4f} -> saved: {ckpt_path}")
    return best_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    cfg = P.load_config(args.config)
    df = P.load_dataframe(cfg, args.project_root)
    folds = stratified_folds(df, cfg)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    fold_metrics = []
    for i, (tr, va) in enumerate(folds):
        print(f"\n=== Fold {i} ===")
        m = train_one_fold(df, tr, va, cfg, fold_id=i,
                           project_root=args.project_root, device=device)
        fold_metrics.append(m)

    print("\n=== 5-Fold Results (mean ± std) ===")
    agg = aggregate_folds(fold_metrics)
    for k, v in agg.items():
        print(f"  {k:20s}: {v['mean']:.4f} ± {v['std']:.4f}")


if __name__ == "__main__":
    main()
