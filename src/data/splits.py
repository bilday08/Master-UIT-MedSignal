# [M1] Chia fold Stratified K-Fold.
# Stratify theo Plaque_present, long them tin hieu echogenicity de can bang nhom duong.
from __future__ import annotations

import numpy as np
import pandas as pd


def make_stratify_key(df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Tao khoa stratify ket hop: plaque + echogenicity.
    - Ca am (Plaque=0): khoa = "0"
    - Ca duong (Plaque=1): khoa = "1_<echo>"  (Low/Intermediate/High)
    => dam bao moi fold can bang ca ti le plaque lan phan bo echo trong nhom duong.
    """
    plaque = df[cfg["columns"]["target_plaque"]].astype(int).astype(str)
    echo = df[cfg["columns"]["target_echo"]].astype(str)
    key = np.where(plaque.values == "1", "1_" + echo.values, "0")
    return key


def stratified_folds(df: pd.DataFrame, cfg: dict) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Tra ve list (train_idx, val_idx) cho moi fold.
    Dung StratifiedKFold cua sklearn voi khoa ket hop o tren.
    """
    from sklearn.model_selection import StratifiedKFold

    n_folds = cfg["split"]["n_folds"]
    seed = cfg["split"]["seed"]
    key = make_stratify_key(df, cfg)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(df)), key))


def fold_summary(df: pd.DataFrame, folds, cfg: dict) -> pd.DataFrame:
    """Bang tom tat ti le plaque tung fold (de kiem tra can bang)."""
    tcol = cfg["columns"]["target_plaque"]
    rows = []
    for i, (tr, va) in enumerate(folds):
        rows.append({
            "fold": i,
            "n_train": len(tr),
            "n_val": len(va),
            "val_pos_rate": round(float(df.iloc[va][tcol].mean()), 3),
        })
    return pd.DataFrame(rows)
