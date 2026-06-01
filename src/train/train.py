# [M4] Vong train multimodal multi-task (5-fold). KHUNG - dien phan TODO.
from __future__ import annotations

import argparse

import torch
from torch.utils.data import DataLoader

from ..data import preprocess as P
from ..data.dataset import CarotidDataset, collate_fn
from ..data.splits import stratified_folds
from ..models.fusion import MultimodalFusion
from .losses import MultiTaskLoss


def train_one_fold(df, train_idx, val_idx, cfg, project_root=".", device="cpu"):
    """
    Train 1 fold. KHUNG: M4 hoan thien phan training loop + early stopping.
    """
    df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]

    # Scaler fit CHI tren train (tranh leakage) — du lieu da encode Sex truoc khi fit.
    scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)

    ds_tr = CarotidDataset(df_tr, cfg, scaler, project_root)
    ds_va = CarotidDataset(df_va, cfg, scaler, project_root)
    dl_tr = DataLoader(ds_tr, batch_size=cfg["train"]["batch_size"], shuffle=True,
                       collate_fn=collate_fn)
    dl_va = DataLoader(ds_va, batch_size=cfg["train"]["batch_size"], shuffle=False,
                       collate_fn=collate_fn)

    in_tab = len(P.feature_columns(cfg))
    model = MultimodalFusion(cfg, in_tab=in_tab).to(device)

    pos_weight = P.compute_pos_weight(df_tr, cfg) if cfg["train"]["pos_weight_auto"] else None
    criterion = MultiTaskLoss(cfg["train"]["loss_weights"], pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                  weight_decay=cfg["train"]["weight_decay"])

    # M4 TODO: vong epoch + early stopping theo val PR-AUC; goi src.eval.metrics.
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        for batch in dl_tr:
            optimizer.zero_grad()
            out = model(batch["tabular"].to(device), batch["imt_img"].to(device),
                        batch["cca_imgs"].to(device), batch["cca_mask"].to(device))
            labels = {k: v.to(device) for k, v in batch["labels"].items()}
            loss, parts = criterion(out, labels)
            loss.backward()
            optimizer.step()
        # M4 TODO: validate(dl_va) + log + checkpoint.
        break  # KHUNG: chi chay 1 vong de smoke test; xoa khi train that.

    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--project-root", default=".")
    args = ap.parse_args()

    cfg = P.load_config(args.config)
    df = P.load_dataframe(cfg, args.project_root)
    folds = stratified_folds(df, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for i, (tr, va) in enumerate(folds):
        print(f"=== Fold {i} ===")
        train_one_fold(df, tr, va, cfg, args.project_root, device)
        # M4 TODO: thu thap metrics moi fold, in mean +/- std cuoi cung.


if __name__ == "__main__":
    main()
