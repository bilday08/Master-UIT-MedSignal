# [M5] Metrics y te + phan tich subgroup discordance.
from __future__ import annotations

import numpy as np
import pandas as pd


def classification_metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    """
    Tinh Sensitivity, Specificity, F1, AUC-ROC, PR-AUC cho bai toan nhi phan.
    y_true: [N] 0/1; y_prob: [N] xac suat duong.
    """
    from sklearn.metrics import (average_precision_score, f1_score,
                                 roc_auc_score, confusion_matrix)

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / max(tp + fn, 1)     # recall lop duong
    spec = tn / max(tn + fp, 1)
    return {
        "sensitivity": round(sens, 4),
        "specificity": round(spec, 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else float("nan"),
        "pr_auc": round(average_precision_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else float("nan"),
    }


def echo_metrics(y_true, y_pred) -> dict:
    """Macro-F1 cho echogenicity 3 lop (bo qua nhan -100)."""
    from sklearn.metrics import f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    keep = y_true != -100
    if keep.sum() == 0:
        return {"macro_f1": float("nan")}
    return {"macro_f1": round(f1_score(y_true[keep], y_pred[keep],
                                       average="macro", zero_division=0), 4)}


def risk_metrics(y_true, y_pred) -> dict:
    """MAE + R2 cho hoi quy risk score."""
    from sklearn.metrics import mean_absolute_error, r2_score

    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "r2": round(r2_score(y_true, y_pred), 4),
    }


def aggregate_folds(fold_metrics: list[dict]) -> dict:
    """Gop metrics nhieu fold -> mean +/- std cho moi khoa."""
    keys = fold_metrics[0].keys()
    out = {}
    for k in keys:
        vals = np.array([m[k] for m in fold_metrics], dtype=float)
        out[k] = {"mean": round(float(np.nanmean(vals)), 4),
                  "std": round(float(np.nanstd(vals)), 4)}
    return out


def discordance_subgroup(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Tach nhom Discordance: LDL-C < nguong VA Lp(a) >= nguong (giu nguong goc).
    THUC TE: 18 ca, 6 duong -> n nho, bao cao trung thuc.
    """
    ldl_max = cfg["eval"]["discordance_ldl_max"]
    lpa_min = cfg["eval"]["discordance_lpa_min"]
    mask = (df["LDL_C_mg_dL"] < ldl_max) & (df["Lp(a)_mg_dL"] >= lpa_min)
    sub = df[mask].copy()
    # M5 TODO: so sanh Sensitivity cua "LDL-only" vs "Multimodal" tren chinh nhom nay.
    return sub
