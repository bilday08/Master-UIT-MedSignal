"""M4 — Xuất Out-of-Fold predictions từ v3 (Focal Loss) cho phân tích discordance.

Nguyên tắc OOF: case P_i chỉ được predict bởi fold mà P_i nằm trong VAL set.
=> Không có data leakage: model chưa từng thấy P_i lúc training.

Output:
  m4_fusion/v3_focal_loss/oof_predictions.csv   — 300 dòng (toàn bộ ca)
  m4_fusion/v3_focal_loss/discordance_oof.csv   — 18 ca (LDL<130 & Lp(a)>=50)

Chạy:
  python3 m4_fusion/scripts/export_oof_predictions.py
  python3 m4_fusion/scripts/export_oof_predictions.py --version v3_focal_loss
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.models.fusion import MultimodalFusion

ECHO_CLASSES = ["Low", "Intermediate", "High"]


def val_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])


def load_v3_config(cfg: dict) -> dict:
    """Override config với đúng siêu tham số của v3 (khớp train.py v3)."""
    cfg = {k: (v.copy() if isinstance(v, dict) else v) for k, v in cfg.items()}
    cfg["train"]["pos_weight_scale"] = 1.2
    cfg["train"]["batch_size"] = 32
    cfg["train"]["focal_gamma"] = 2.0
    return cfg


def load_model_from_checkpoint(ckpt_path: Path, cfg: dict, device: torch.device) -> torch.nn.Module:
    n_tab = len(P.feature_columns(cfg))
    model = MultimodalFusion(cfg, in_tab=n_tab).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


@torch.no_grad()
def predict_fold(model: torch.nn.Module, dl: DataLoader, threshold: float,
                 device: torch.device) -> list[dict]:
    records = []
    for batch in dl:
        out = model(
            batch["tabular"].to(device),
            batch["imt_img"].to(device),
            batch["cca_imgs"].to(device),
            batch["cca_mask"].to(device),
        )
        plaque_prob = torch.sigmoid(out["plaque"]).squeeze(1).cpu().numpy()
        echo_logits = out["echo"].cpu().numpy()           # [B, 3]
        risk_pred = out["risk"].squeeze(1).cpu().numpy()

        true_plaque = batch["labels"]["plaque"].squeeze(1).numpy()
        true_echo = batch["labels"]["echo"].squeeze(1).numpy()   # -100 = ca âm
        true_risk = batch["labels"]["risk"].squeeze(1).numpy()

        for i, pid in enumerate(batch["patient_id"]):
            echo_idx = int(np.argmax(echo_logits[i]))
            # Ca âm (true_echo == -100): echo prediction vô nghĩa — ghi None
            echo_label = ECHO_CLASSES[echo_idx] if int(true_echo[i]) != -100 else None
            records.append({
                "patient_id": pid,
                "plaque_prob": round(float(plaque_prob[i]), 4),
                "plaque_pred": int(plaque_prob[i] >= threshold),
                "plaque_true": int(true_plaque[i]),
                "echo_pred": echo_label,
                "echo_true": (ECHO_CLASSES[int(true_echo[i])] if int(true_echo[i]) != -100 else None),
                "risk_pred": round(float(risk_pred[i]), 4),
                "risk_true": round(float(true_risk[i]), 4),
            })
    return records


def run_oof(version: str = "v3_focal_loss") -> pd.DataFrame:
    cfg_path = PROJECT_ROOT / "configs/config.yaml"
    cfg = load_v3_config(P.load_config(str(cfg_path)))

    ckpt_dir = PROJECT_ROOT / "m4_fusion" / version / "checkpoints"
    results_path = PROJECT_ROOT / "m4_fusion" / version / "results.json"

    with open(results_path) as f:
        results = json.load(f)
    fold_thresholds = [r["best_threshold"] for r in results["fold_results"]]

    df = P.load_dataframe(cfg, str(PROJECT_ROOT))
    folds = stratified_folds(df, cfg)   # seed=42, deterministic

    device = (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("mps") if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Device: {device} | Version: {version}")

    img_size = cfg["data"]["image_size"]
    all_records: list[dict] = []

    for fold_id, (train_idx, val_idx) in enumerate(folds):
        df_tr = df.iloc[train_idx]
        df_va = df.iloc[val_idx]
        threshold = fold_thresholds[fold_id]

        # Refit scaler CHỈ trên train — tránh leakage từ val
        scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)

        ds_va = CarotidDataset(
            df_va, cfg, scaler, str(PROJECT_ROOT),
            transform=val_transform(img_size),
            cca_transform=val_transform(img_size),
        )
        dl_va = DataLoader(
            ds_va, batch_size=cfg["train"]["batch_size"],
            shuffle=False, collate_fn=collate_fn, num_workers=0,
        )

        ckpt_path = ckpt_dir / f"fold{fold_id}_best.pt"
        model = load_model_from_checkpoint(ckpt_path, cfg, device)

        records = predict_fold(model, dl_va, threshold, device)
        for r in records:
            r["fold"] = fold_id
            r["threshold_used"] = threshold

        print(f"  Fold {fold_id}: {len(records)} ca | threshold={threshold:.4f} | "
              f"pred_pos={sum(r['plaque_pred'] for r in records)}")
        all_records.extend(records)

    df_oof = pd.DataFrame(all_records)
    assert len(df_oof) == 300, f"Kỳ vọng 300 ca, thực tế {len(df_oof)}"
    assert df_oof["patient_id"].nunique() == 300, "Có ca bị predict 2 lần!"

    return df_oof


def add_clinical_columns(df_oof: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Ghép lại các cột lâm sàng cần thiết cho phân tích discordance."""
    df_raw = P.load_dataframe(cfg, str(PROJECT_ROOT))
    keep_cols = [
        cfg["columns"]["id"],
        "LDL_C_mg_dL",
        "Lp(a)_mg_dL",
        "Age",
        "Sex",
        "IMT_mm",
        cfg["columns"]["target_plaque"],
        cfg["columns"]["target_echo"],
        cfg["columns"]["target_risk"],
    ]
    df_raw = df_raw[[c for c in keep_cols if c in df_raw.columns]]
    df_raw = df_raw.rename(columns={cfg["columns"]["id"]: "patient_id"})
    return df_oof.merge(df_raw, on="patient_id", how="left")


def main():
    parser = argparse.ArgumentParser(description="Xuất OOF predictions từ v3 fusion")
    parser.add_argument("--version", default="v3_focal_loss",
                        help="Thư mục version trong m4_fusion/ (mặc định: v3_focal_loss)")
    args = parser.parse_args()

    # --- OOF inference ---
    df_oof = run_oof(version=args.version)

    # --- Ghép cột lâm sàng ---
    cfg = P.load_config(str(PROJECT_ROOT / "configs/config.yaml"))
    df_oof = add_clinical_columns(df_oof, cfg)

    # --- Lưu toàn bộ 300 ca ---
    out_dir = PROJECT_ROOT / "m4_fusion" / args.version
    oof_path = out_dir / "oof_predictions.csv"
    df_oof.to_csv(oof_path, index=False)
    print(f"\nSaved 300-ca OOF → {oof_path}")

    # --- Lọc 18 ca discordance ---
    ldl_max = cfg["eval"]["discordance_ldl_max"]   # 130
    lpa_min = cfg["eval"]["discordance_lpa_min"]   # 50
    mask = (df_oof["LDL_C_mg_dL"] < ldl_max) & (df_oof["Lp(a)_mg_dL"] >= lpa_min)
    df_disc = df_oof[mask].copy()

    disc_path = out_dir / "discordance_oof.csv"
    df_disc.to_csv(disc_path, index=False)

    n_disc_pos = int(df_disc["plaque_true"].sum())
    n_disc_pred_pos = int(df_disc["plaque_pred"].sum())
    n_disc_tp = int(((df_disc["plaque_true"] == 1) & (df_disc["plaque_pred"] == 1)).sum())
    n_disc_fp = int(((df_disc["plaque_true"] == 0) & (df_disc["plaque_pred"] == 1)).sum())
    sensitivity_disc = n_disc_tp / max(n_disc_pos, 1)

    print(f"Saved {len(df_disc)} ca discordance → {disc_path}")
    print(f"\n=== Discordance Summary (LDL<{ldl_max} & Lp(a)>={lpa_min}) ===")
    print(f"  Tổng:              {len(df_disc)} ca")
    print(f"  Plaque dương thực: {n_disc_pos} ca")
    print(f"  Model predict pos: {n_disc_pred_pos} ca (TP={n_disc_tp}, FP={n_disc_fp})")
    print(f"  Sensitivity(disc): {sensitivity_disc:.3f} "
          f"({n_disc_tp}/{n_disc_pos} ca dương được detect)")

    print("\nDanh sách 18 ca discordance:")
    cols_show = ["patient_id", "fold", "LDL_C_mg_dL", "Lp(a)_mg_dL",
                 "plaque_true", "plaque_prob", "plaque_pred", "echo_pred", "risk_pred"]
    cols_show = [c for c in cols_show if c in df_disc.columns]
    print(df_disc[cols_show].sort_values("patient_id").to_string(index=False))


if __name__ == "__main__":
    main()
