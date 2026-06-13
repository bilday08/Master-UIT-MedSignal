"""
M3 Ablation Study — Attention Pooling vs Mean Pooling cho CCA.

So sánh hai chiến lược pooling trong VisionBranch:
  - AttentionPool (gated attention, Ilse 2018)
  - MaskedMeanPool (mean đơn giản, có mask)

Cả hai đều train VisionBranch end-to-end với:
  - imt_feat -> plaque head (BCE)
  - cca_feat -> echo head (CE, ignore_index=-100)

Usage:
  python3 m3_vision/ablation_pooling.py \\
    --encoder custom_cnn --epochs 20 --folds 5 \\
    --output m3_vision/results/ablation_pooling_metrics.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.eval.metrics import classification_metrics
from src.models.vision import VisionBranch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_transforms(train: bool, image_size: int):
    from torchvision import transforms

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


class AblationModel(nn.Module):
    """VisionBranch + plaque head + echo head để ablation pooling."""

    def __init__(self, vision: VisionBranch, feat_dim: int = 128,
                 n_echo_classes: int = 3):
        super().__init__()
        self.vision = vision
        self.plaque_head = nn.Linear(feat_dim, 1)
        self.echo_head = nn.Linear(feat_dim, n_echo_classes)

    def forward(self, imt_img, cca_imgs, cca_mask):
        imt_feat, cca_feat = self.vision(imt_img, cca_imgs, cca_mask)
        plaque_logit = self.plaque_head(imt_feat)       # [B,1]
        echo_logit = self.echo_head(cca_feat)            # [B,3]
        return plaque_logit, echo_logit


@torch.no_grad()
def evaluate(model, loader, device, threshold: float, max_batches: int | None = None):
    model.eval()
    y_true_p, y_prob_p = [], []
    y_true_e, y_pred_e = [], []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        if max_batches is not None and step >= max_batches:
            break
        imt_img = batch["imt_img"].to(device)
        cca_imgs = batch["cca_imgs"].to(device)
        cca_mask = batch["cca_mask"].to(device)
        plaque_logit, echo_logit = model(imt_img, cca_imgs, cca_mask)

        # Plaque metrics
        prob = torch.sigmoid(plaque_logit).squeeze(1).cpu().numpy()
        true_p = batch["labels"]["plaque"].squeeze(1).cpu().numpy()
        y_prob_p.extend(prob.tolist())
        y_true_p.extend(true_p.tolist())

        # Echo metrics
        pred_e = echo_logit.argmax(dim=1).cpu().numpy()
        true_e = batch["labels"]["echo"].squeeze(1).cpu().numpy()
        y_pred_e.extend(pred_e.tolist())
        y_true_e.extend(true_e.tolist())

    plaque_metrics = classification_metrics(y_true_p, y_prob_p, threshold=threshold)

    # Echo macro-F1 (ignore -100)
    y_true_e = np.array(y_true_e)
    y_pred_e = np.array(y_pred_e)
    keep = y_true_e != -100
    from sklearn.metrics import f1_score
    echo_f1 = f1_score(y_true_e[keep], y_pred_e[keep], average="macro",
                       zero_division=0) if keep.sum() > 0 else float("nan")

    plaque_metrics["echo_macro_f1"] = round(echo_f1, 4)
    plaque_metrics["n_eval"] = len(y_true_p)
    return plaque_metrics


def train_one_pooling(pooling_name: str, df, folds, cfg, args, device):
    """Train 5-fold cho 1 loại pooling."""
    feat_dim = cfg["vision"]["feat_dim"]
    n_echo = len(cfg["labels"]["echo_classes"])
    fold_results = []

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        df_train = df.iloc[train_idx]
        df_val = df.iloc[val_idx]
        scaler = P.fit_scaler(P.encode_categorical(df_train, cfg), cfg)

        train_ds = CarotidDataset(
            df_train, cfg, scaler, project_root=args.project_root,
            transform=make_transforms(True, cfg["data"]["image_size"]),
        )
        val_ds = CarotidDataset(
            df_val, cfg, scaler, project_root=args.project_root,
            transform=make_transforms(False, cfg["data"]["image_size"]),
        )
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=device.type == "cuda",
            collate_fn=collate_fn,
        )

        vision = VisionBranch(
            encoder=args.encoder, feat_dim=feat_dim,
            pretrained=args.pretrained, dropout=cfg["train"]["dropout"],
            pooling=pooling_name,
        )
        model = AblationModel(vision, feat_dim, n_echo).to(device)

        # Loss
        pos_weight = torch.tensor([P.compute_pos_weight(df_train, cfg)], device=device)
        plaque_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        echo_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        best_metrics = None
        best_epoch = 0
        patience_counter = 0
        history = []

        for epoch in range(1, args.epochs + 1):
            # ---- Train ----
            model.train()
            losses = []
            for step, batch in enumerate(train_loader):
                if args.max_train_batches is not None and step >= args.max_train_batches:
                    break
                imt_img = batch["imt_img"].to(device)
                cca_imgs = batch["cca_imgs"].to(device)
                cca_mask = batch["cca_mask"].to(device)
                plaque_label = batch["labels"]["plaque"].to(device)
                echo_label = batch["labels"]["echo"].squeeze(1).to(device)

                optimizer.zero_grad(set_to_none=True)
                plaque_logit, echo_logit = model(imt_img, cca_imgs, cca_mask)

                loss_p = plaque_loss_fn(plaque_logit, plaque_label)
                loss_e = echo_loss_fn(echo_logit, echo_label)
                loss = loss_p + 0.5 * loss_e  # multi-task weight
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))

            scheduler.step()
            train_loss = float(np.mean(losses))

            # ---- Evaluate ----
            metrics = evaluate(model, val_loader, device,
                               threshold=cfg["eval"]["decision_threshold"],
                               max_batches=args.max_val_batches)
            metrics["train_loss"] = round(train_loss, 4)
            history.append({"epoch": epoch, **metrics})

            score = metrics.get("pr_auc", -1.0)
            best_score = best_metrics.get("pr_auc", -1.0) if best_metrics else -1.0

            if score > best_score:
                best_metrics = metrics.copy()
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"  [{pooling_name}] fold={fold_id} epoch={epoch} "
                f"loss={train_loss:.4f} pr_auc={metrics['pr_auc']} "
                f"f1={metrics['f1']} echo_f1={metrics['echo_macro_f1']}"
            )

            if args.early_stop and patience_counter >= args.early_stop_patience:
                print(f"  -> Early stop at epoch {epoch}")
                break

        fold_results.append({
            "fold": fold_id,
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": history,
        })

    return fold_results


def summarize(fold_results):
    metric_keys = ["sensitivity", "specificity", "f1", "auc_roc", "pr_auc", "echo_macro_f1"]
    summary = {}
    for key in metric_keys:
        vals = np.array([r["best_metrics"].get(key, float("nan")) for r in fold_results], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
            continue
        summary[key] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="M3 Ablation: Attention vs Mean Pooling")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--encoder", default="custom_cnn", choices=["custom_cnn", "resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop", action="store_true")
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--output", default="m3_vision/results/ablation_pooling_metrics.json")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = P.load_config(args.config)
    df = P.load_dataframe(cfg, project_root=args.project_root)
    folds = stratified_folds(df, cfg)[:args.folds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} encoder={args.encoder} folds={len(folds)} epochs={args.epochs}")

    results = {}
    for pooling_name in ["attention", "mean"]:
        print(f"\n{'='*60}")
        print(f"Training with pooling: {pooling_name}")
        print(f"{'='*60}")
        fold_results = train_one_pooling(pooling_name, df, folds, cfg, args, device)
        summary = summarize(fold_results)
        results[pooling_name] = {
            "encoder": args.encoder,
            "pooling": pooling_name,
            "epochs": args.epochs,
            "folds": len(folds),
            "summary": summary,
            "fold_results": fold_results,
        }
        print(f"\n{pooling_name} summary:")
        for k, v in summary.items():
            print(f"  {k}: {v['mean']:.4f} ± {v['std']:.4f}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved to {output}")


if __name__ == "__main__":
    main()
