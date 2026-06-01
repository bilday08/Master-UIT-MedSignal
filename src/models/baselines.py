# [M2] Baseline ML truyen thong (khong deep learning) — mốc so sanh.
# Bao gom: XGBoost/LightGBM cho plaque, va baseline "LDL-only"/"lipid panel" cho discordance.
from __future__ import annotations

import numpy as np
import pandas as pd


def build_tree_classifier(kind: str = "xgboost", **kwargs):
    """
    Tao classifier cay (XGBoost hoac LightGBM) cho Plaque_present.
    M2 TODO: tune scale_pos_weight (= n_neg/n_pos) de xu ly lech lop.
    """
    if kind == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr", **kwargs,
        )
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, **kwargs,
        )
    raise ValueError(f"kind khong ho tro: {kind}")


def build_risk_regressor(kind: str = "xgboost", **kwargs):
    """Baseline hoi quy Baseline_Risk_Score."""
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, **kwargs)
    from lightgbm import LGBMRegressor
    return LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, **kwargs)


def ldl_only_features(df: pd.DataFrame) -> np.ndarray:
    """Baseline doi chung: CHI dung LDL-C (mo phong sang loc lipid truyen thong)."""
    return df[["LDL_C_mg_dL"]].values


def lipid_panel_features(df: pd.DataFrame) -> np.ndarray:
    """Baseline doi chung: panel lipid day du (KHONG co Lp(a) de lam noi vai tro Lp(a))."""
    cols = ["LDL_C_mg_dL", "ApoB_mg_dL", "Triglyceride_mg_dL",
            "Total_Cholesterol_mg_dL", "Non_HDL_mg_dL"]
    return df[cols].values

# M2 TODO: viet ham fit/predict tren 5-fold tra ve dict metrics (goi src.eval.metrics).
