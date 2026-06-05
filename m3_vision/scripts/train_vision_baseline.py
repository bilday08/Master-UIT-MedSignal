from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.eval.metrics import classification_metrics
from src.models.vision import VisionPlaqueClassifier


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


def run_epoch(model, loader, criterion, optimizer, device, max_batches: int | None = None) -> float:
    model.train()
    losses: list[float] = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        imt_img = batch["imt_img"].to(device)
        labels = batch["labels"]["plaque"].to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imt_img)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate(model, loader, device, threshold: float, max_batches: int | None = None) -> dict:
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        logits = model(batch["imt_img"].to(device))
        prob = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        true = batch["labels"]["plaque"].squeeze(1).cpu().numpy()
        y_prob.extend(prob.tolist())
        y_true.extend(true.tolist())

    metrics = classification_metrics(y_true, y_prob, threshold=threshold)
    metrics["n_eval"] = len(y_true)
    return metrics


def metric_score(metrics: dict) -> float:
    pr_auc = metrics.get("pr_auc", float("nan"))
    if not np.isnan(pr_auc):
        return float(pr_auc)
    f1 = metrics.get("f1", float("nan"))
    return float(f1) if not np.isnan(f1) else -1.0


def train_one_fold(df, train_idx, val_idx, cfg, args, device, fold_id: int) -> dict:
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]
    scaler = P.fit_scaler(P.encode_categorical(df_train, cfg), cfg)

    train_ds = CarotidDataset(
        df_train,
        cfg,
        scaler,
        project_root=args.project_root,
        transform=make_transforms(True, cfg["data"]["image_size"]),
    )
    val_ds = CarotidDataset(
        df_val,
        cfg,
        scaler,
        project_root=args.project_root,
        transform=make_transforms(False, cfg["data"]["image_size"]),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
    )

    model = VisionPlaqueClassifier(
        encoder=args.encoder,
        feat_dim=cfg["vision"]["feat_dim"],
        pretrained=args.pretrained,
        dropout=cfg["train"]["dropout"],
    ).to(device)

    pos_weight = torch.tensor([P.compute_pos_weight(df_train, cfg)], device=device)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_metrics: dict | None = None
    best_state: dict | None = None
    best_epoch = 0
    patience_counter = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            max_batches=args.max_train_batches,
        )
        metrics = evaluate(
            model,
            val_loader,
            device,
            threshold=cfg["eval"]["decision_threshold"],
            max_batches=args.max_val_batches,
        )
        metrics["train_loss"] = round(train_loss, 4)
        metrics["lr"] = round(float(optimizer.param_groups[0]["lr"]), 6)
        history.append({"epoch": epoch, **metrics})

        scheduler.step()

        if best_metrics is None or metric_score(metrics) > metric_score(best_metrics):
            best_metrics = metrics.copy()
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"fold={fold_id} epoch={epoch} "
            f"loss={train_loss:.4f} pr_auc={metrics['pr_auc']} "
            f"auc={metrics['auc_roc']} f1={metrics['f1']} "
            f"sens={metrics['sensitivity']} spec={metrics['specificity']} "
            f"lr={metrics['lr']}"
        )

        if args.early_stop and patience_counter >= args.early_stop_patience:
            print(f"  -> Early stopping at epoch {epoch} (patience={args.early_stop_patience})")
            break

    assert best_metrics is not None
    checkpoint_path = None
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{args.encoder}_fold{fold_id}_best.pt"
        torch.save({
            "model_state": best_state,
            "encoder": args.encoder,
            "feat_dim": cfg["vision"]["feat_dim"],
            "pretrained": args.pretrained,
            "dropout": cfg["train"]["dropout"],
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "total_epochs_trained": epoch,
        }, checkpoint_path)

    return {
        "fold": fold_id,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "history": history,
    }


def summarize_folds(fold_results: list[dict]) -> dict:
    metric_keys = ["sensitivity", "specificity", "f1", "auc_roc", "pr_auc"]
    summary = {}
    for key in metric_keys:
        vals = np.array([r["best_metrics"][key] for r in fold_results], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
            continue
        summary[key] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="M3 IMT-only vision baseline for Plaque_present.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--encoder", default="custom_cnn", choices=["custom_cnn", "resnet18"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--early-stop", action="store_true", help="Enable early stopping")
    parser.add_argument("--early-stop-patience", type=int, default=7, help="Patience epochs for early stopping")
    parser.add_argument("--output", default="m3_vision/results/vision_baseline_metrics.json")
    parser.add_argument("--checkpoint-dir", default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = P.load_config(args.config)
    df = P.load_dataframe(cfg, project_root=args.project_root)
    folds = stratified_folds(df, cfg)[:args.folds]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} encoder={args.encoder} folds={len(folds)} epochs={args.epochs}")

    fold_results = []
    for fold_id, (train_idx, val_idx) in enumerate(folds):
        fold_results.append(train_one_fold(df, train_idx, val_idx, cfg, args, device, fold_id))

    result = {
        "task": "M3 IMT-only Plaque_present baseline",
        "encoder": args.encoder,
        "epochs": args.epochs,
        "folds": len(folds),
        "summary": summarize_folds(fold_results),
        "fold_results": fold_results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))
    print(f"saved={output}")


if __name__ == "__main__":
    main()
