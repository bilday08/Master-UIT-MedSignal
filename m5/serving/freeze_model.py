"""Freeze model v3 cho demo live (M5).

Train MultimodalFusion (cau hinh v3: Focal Loss + WeightedSampler) tren CA 300 ca
-> luu bundle artifact ma FastAPI load:

  m5/serving/artifacts/
    model.pth        # state_dict
    scaler.joblib    # StandardScaler fit tren toan 300 ca
    threshold.json   # youden threshold + feature_names + cfg snapshot
    meta.json        # version, frozen_at, reference_metrics (tu 5-fold)

LUU Y TRUNG THUC: model nay train tren ca 300 ca CHI de demo tuong tac.
Moi metric tren dashboard lay tu ket qua 5-fold cua M2/M4 (m4_fusion/*/results.json),
KHONG tu model nay (tranh leakage train=test).

Chay 1 lan:  .venv/bin/python -m m5.serving.freeze_model
"""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn, make_weighted_sampler
from src.models.fusion import MultimodalFusion

from m5.serving.common import (
    ARTIFACT_DIR,
    PROJECT_ROOT,
    FocalMultiTaskLoss,
    load_v3_config,
    make_transforms,
    select_device,
    youden_threshold,
)


def _predict_all(model, dl, device) -> tuple[np.ndarray, np.ndarray]:
    """Chay model tren loader (val-transform) -> (y_true, y_prob) cho plaque."""
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for batch in dl:
            out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                        batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
            probs.append(torch.sigmoid(out["plaque"]).cpu())
            trues.append(batch["labels"]["plaque"])
    return (torch.cat(trues).squeeze(1).numpy(), torch.cat(probs).squeeze(1).numpy())


def _reference_metrics() -> dict:
    """Doc summary 5-fold cua v3 (so lieu trung thuc cho dashboard)."""
    path = PROJECT_ROOT / "m4_fusion" / "v3_focal_loss" / "results.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    summary = data.get("summary", {})
    keys = ["auc_roc", "pr_auc", "f1", "sensitivity", "specificity", "macro_f1", "mae", "r2"]
    return {k: summary[k] for k in keys if k in summary}


def freeze(epochs: int | None = None, seed: int = 42,
           device: torch.device | None = None, workers: int = 0) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    cfg = load_v3_config()
    if epochs is not None:
        cfg["train"]["epochs"] = epochs
    device = device or select_device()
    img_size = cfg["data"]["image_size"]
    print(f"[freeze] device={device} | epochs={cfg['train']['epochs']} | v3 FocalLoss")

    # --- Data: CA 300 ca ---
    df = P.load_dataframe(cfg, str(PROJECT_ROOT))
    scaler = P.fit_scaler(P.encode_categorical(df, cfg), cfg)  # fit tren toan bo (model deploy)

    ds_train = CarotidDataset(df, cfg, scaler, str(PROJECT_ROOT),
                              transform=make_transforms(True, img_size),
                              cca_transform=make_transforms(True, img_size))
    ds_eval = CarotidDataset(df, cfg, scaler, str(PROJECT_ROOT),
                             transform=make_transforms(False, img_size),
                             cca_transform=make_transforms(False, img_size))
    dl_train = DataLoader(ds_train, batch_size=cfg["train"]["batch_size"],
                          sampler=make_weighted_sampler(df, cfg),
                          collate_fn=collate_fn, num_workers=workers)
    dl_eval = DataLoader(ds_eval, batch_size=cfg["train"]["batch_size"],
                         shuffle=False, collate_fn=collate_fn, num_workers=workers)

    # --- Model + loss ---
    model = MultimodalFusion(cfg, in_tab=len(P.feature_columns(cfg))).to(device)
    pos_weight = P.compute_pos_weight(df, cfg) * cfg["train"]["pos_weight_scale"]
    criterion = FocalMultiTaskLoss(cfg["train"]["loss_weights"], pos_weight,
                                   gamma=cfg["train"]["focal_gamma"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    # --- Train ---
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in dl_train:
            optimizer.zero_grad()
            out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                        batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            loss, _ = criterion(out, labels)
            loss.backward()
            optimizer.step()
            running += float(loss.detach())
        print(f"[freeze] epoch {epoch + 1:>3}/{cfg['train']['epochs']} | loss {running / len(dl_train):.4f}")

    # --- Threshold Youden tren toan tap (chi de cat nhan demo) ---
    y_true, y_prob = _predict_all(model, dl_eval, device)
    threshold = youden_threshold(y_true, y_prob)
    print(f"[freeze] youden threshold = {threshold:.4f}")

    # --- Luu artifacts ---
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ARTIFACT_DIR / "model.pth")
    joblib.dump(scaler, ARTIFACT_DIR / "scaler.joblib")
    (ARTIFACT_DIR / "threshold.json").write_text(json.dumps({
        "threshold": threshold,
        "feature_names": P.feature_columns(cfg),
        "echo_classes": cfg["labels"]["echo_classes"],
        "cfg": cfg,
    }, indent=2))
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps({
        "version": "v3_focal_loss",
        "note": "Train tren ca 300 ca CHI de demo. Metric bao cao lay tu 5-fold.",
        "device": str(device),
        "epochs": cfg["train"]["epochs"],
        "reference_metrics_5fold": _reference_metrics(),
    }, indent=2))
    print(f"[freeze] saved artifacts -> {ARTIFACT_DIR}")
    return {"threshold": threshold, "artifact_dir": str(ARTIFACT_DIR)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=None, help="override so epoch (mac dinh 30)")
    ap.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None,
                    help="ep device; mac dinh auto (cuda>mps>cpu)")
    ap.add_argument("--workers", type=int, default=0, help="num_workers DataLoader (song song augmentation)")
    args = ap.parse_args()
    dev = torch.device(args.device) if args.device else None
    freeze(epochs=args.epochs, device=dev, workers=args.workers)
