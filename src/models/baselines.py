# [M2] Baseline ML truyen thong (khong deep learning) — mốc so sanh.
# Bao gom: XGBoost/LightGBM cho plaque, va baseline "LDL-only"/"lipid panel" cho discordance.
from __future__ import annotations

import numpy as np
import pandas as pd
# torch import lazy (chi khi can) de tranh crash khi chi dung tree/logistic models


# ─────────────────────────────────────────────────────────────
# Tree model factories
# ─────────────────────────────────────────────────────────────

def build_tree_classifier(kind: str = "xgboost", scale_pos_weight: float | None = None, **kwargs):
    """
    Tao classifier cay cho Plaque_present.
    scale_pos_weight = n_neg/n_pos truyen vao tu run_cv_classifier.

    XGBoost  : dung tham so 'scale_pos_weight' chinh xac.
    LightGBM : chuyen sang 'scale_pos_weight' cung co nhung de an toan hon
               dung 'class_weight' map sang {0:1, 1:scale_pos_weight}.
    """
    if kind == "xgboost":
        from xgboost import XGBClassifier
        spw = {"scale_pos_weight": scale_pos_weight} if scale_pos_weight is not None else {}
        return XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
            random_state=42, **spw, **kwargs,
        )
    elif kind == "lightgbm":
        from lightgbm import LGBMClassifier
        # LightGBM: class_weight={0:1, 1:scale_pos_weight} tuong duong scale_pos_weight
        # nhung an toan hon voi moi phien ban
        cw = {0: 1.0, 1: float(scale_pos_weight)} if scale_pos_weight is not None else None
        return LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, subsample_freq=5,   # subsample can subsample_freq > 0
            colsample_bytree=0.8, verbose=-1,
            class_weight=cw, random_state=42, **kwargs,
        )
    raise ValueError(f"kind khong ho tro: {kind}")


def build_risk_regressor(kind: str = "xgboost", **kwargs):
    """Baseline hoi quy Baseline_Risk_Score."""
    if kind == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, **kwargs,
        )
    from lightgbm import LGBMRegressor
    return LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, subsample_freq=5,
        colsample_bytree=0.8, verbose=-1,
        random_state=42, **kwargs,
    )


def ldl_only_features(df: pd.DataFrame) -> np.ndarray:
    """Baseline doi chung: CHI dung LDL-C (mo phong sang loc lipid truyen thong)."""
    return df[["LDL_C_mg_dL"]].values


def lipid_panel_features(df: pd.DataFrame) -> np.ndarray:
    """Baseline doi chung: panel lipid day du (KHONG co Lp(a) de lam noi vai tro Lp(a))."""
    cols = ["LDL_C_mg_dL", "ApoB_mg_dL", "Triglyceride_mg_dL",
            "Total_Cholesterol_mg_dL", "Non_HDL_mg_dL"]
    return df[cols].values


# ─────────────────────────────────────────────────────────────
# Reusable 5-fold CV wrappers
# ─────────────────────────────────────────────────────────────

def run_cv_classifier(
    kind: str,
    df: pd.DataFrame,
    folds: list,
    cfg: dict,
    feature_fn=None,
) -> list[dict]:
    """
    Run a 5-fold CV for the plaque_present classification task.

    Args:
        kind       : "xgboost" | "lightgbm" | "logistic"
        df         : DataFrame with encoded Sex (scaler applied internally)
        folds      : list (train_idx, val_idx) from stratified_folds()
        cfg        : config dict
        feature_fn : function (df) -> np.ndarray; if None use all feature_columns

    Returns: list[dict] metrics each fold.
    """
    from sklearn.linear_model import LogisticRegression
    from src.data import preprocess as P
    from src.eval.metrics import classification_metrics

    feat_cols = P.feature_columns(cfg)
    ycol = cfg["columns"]["target_plaque"]
    fold_metrics = []

    for tr_idx, va_idx in folds:
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]

        # Scaler fit CHI tren train
        scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
        df_tr_sc = P.apply_scaler(P.encode_categorical(df_tr, cfg), scaler, cfg)
        df_va_sc = P.apply_scaler(P.encode_categorical(df_va, cfg), scaler, cfg)

        if feature_fn is not None:
            X_tr = feature_fn(df_tr_sc)
            X_va = feature_fn(df_va_sc)
        else:
            X_tr = df_tr_sc[feat_cols].values
            X_va = df_va_sc[feat_cols].values

        y_tr = df_tr[ycol].values.astype(int)
        y_va = df_va[ycol].values.astype(int)
        pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

        if kind == "logistic":
            clf = LogisticRegression(
                class_weight="balanced", max_iter=1000, random_state=42,
            )
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_va)[:, 1]
        else:
            # scale_pos_weight truyen truc tiep (da xu ly dung cho tung loai trong factory)
            clf = build_tree_classifier(kind, scale_pos_weight=pos_w)
            clf.fit(X_tr, y_tr)
            prob = clf.predict_proba(X_va)[:, 1]

        fold_metrics.append(classification_metrics(y_va, prob))

    return fold_metrics


def run_cv_regressor(
    kind: str,
    df: pd.DataFrame,
    folds: list,
    cfg: dict,
) -> list[dict]:
    """
    Run a 5-fold CV for the Baseline_Risk_Score regression task.

    Args:
        kind       : "xgboost" | "lightgbm" | "mlp"
        df         : DataFrame with encoded categorical features
        folds      : list (train_idx, val_idx) from stratified_folds()
        cfg        : config dict

    Returns: list[dict] metrics each fold.
    """
    from src.data import preprocess as P
    from src.eval.metrics import risk_metrics

    feat_cols = P.feature_columns(cfg)
    ycol = cfg["columns"]["target_risk"]
    fold_metrics = []

    for tr_idx, va_idx in folds:
        df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]

        scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
        df_tr_sc = P.apply_scaler(P.encode_categorical(df_tr, cfg), scaler, cfg)
        df_va_sc = P.apply_scaler(P.encode_categorical(df_va, cfg), scaler, cfg)

        X_tr = df_tr_sc[feat_cols].values.astype(np.float32)
        X_va = df_va_sc[feat_cols].values.astype(np.float32)
        y_tr = df_tr[ycol].values.astype(np.float32)
        y_va = df_va[ycol].values.astype(np.float32)

        if kind == "mlp":
            model = TabularRiskRegressor(in_dim=X_tr.shape[1])
            preds = _train_mlp_regressor(model, X_tr, y_tr, X_va)
        else:
            reg = build_risk_regressor(kind)
            reg.fit(X_tr, y_tr)
            preds = reg.predict(X_va)

        fold_metrics.append(risk_metrics(y_va, preds))

    return fold_metrics


# ─────────────────────────────────────────────────────────────
# MLP Regressor for Baseline_Risk_Score
# ─────────────────────────────────────────────────────────────

def _get_torch():
    """Lazy import torch — tranh crash khi chi dung tree/logistic models."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


class TabularRiskRegressor:
    """
    MLP hoi quy Baseline_Risk_Score.
    Kien truc [64,32] -> 1 gia tri lien tuc (khong sigmoid).
    Ke thua nn.Module khi torch co san (lazy import).
    """

    def __new__(cls, *args, **kwargs):
        torch, nn, _, _ = _get_torch()

        class _TabularRiskRegressorImpl(nn.Module):
            def __init__(self, in_dim=9, hidden=(64, 32), dropout=0.3):
                super().__init__()
                dims = [in_dim, *hidden]
                layers = []
                for a, b in zip(dims[:-1], dims[1:]):
                    layers += [nn.Linear(a, b), nn.BatchNorm1d(b), nn.ReLU(), nn.Dropout(dropout)]
                self.backbone = nn.Sequential(*layers)
                self.head = nn.Linear(dims[-1], 1)

            def forward(self, x):
                return self.head(self.backbone(x)).squeeze(-1)  # [B]

        return _TabularRiskRegressorImpl(*args, **kwargs)


def _train_mlp_regressor(
    model,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    epochs: int = 60,
    lr: float = 3e-4,
    batch_size: int = 32,
    weight_decay: float = 1e-4,
) -> np.ndarray:
    """Train TabularRiskRegressor 1 fold, tra ve predictions tren val set."""
    torch, nn, DataLoader, TensorDataset = _get_torch()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.SmoothL1Loss()

    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_va).to(device)).cpu().numpy()
    return preds
