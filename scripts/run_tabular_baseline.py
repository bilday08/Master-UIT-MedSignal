"""M2 tabular baseline runner: XGBoost, LightGBM, TabularMLP, LDL-only — 5-fold."""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.splits import stratified_folds
from src.eval.metrics import aggregate_folds, classification_metrics
from src.models.baselines import (build_tree_classifier, build_risk_regressor,
                                   ldl_only_features, lipid_panel_features)
from src.models.tabular import TabularClassifier


def run_tree_cv(df, folds, cfg, kind="xgboost", feature_fn=None):
    results = []
    for train_idx, val_idx in folds:
        df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]
        scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
        if feature_fn:
            X_tr = feature_fn(df_tr)
            X_va = feature_fn(df_va)
        else:
            feat_cols = P.feature_columns(cfg)
            df_tr_enc = P.apply_scaler(P.encode_categorical(df_tr, cfg), scaler, cfg)
            df_va_enc = P.apply_scaler(P.encode_categorical(df_va, cfg), scaler, cfg)
            X_tr = df_tr_enc[feat_cols].values
            X_va = df_va_enc[feat_cols].values
        y_tr = df_tr[cfg["columns"]["target_plaque"]].values
        y_va = df_va[cfg["columns"]["target_plaque"]].values

        n_neg, n_pos = (y_tr == 0).sum(), (y_tr == 1).sum()
        spw = n_neg / max(n_pos, 1)
        clf = build_tree_classifier(kind, scale_pos_weight=spw)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_va)[:, 1]
        results.append(classification_metrics(y_va, prob, threshold=0.3))
    return results


def run_mlp_cv(df, folds, cfg, epochs=50, lr=3e-4, device="cpu"):
    results = []
    feat_cols = P.feature_columns(cfg)
    in_dim = len(feat_cols)
    for train_idx, val_idx in folds:
        df_tr, df_va = df.iloc[train_idx], df.iloc[val_idx]
        scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
        df_tr_enc = P.apply_scaler(P.encode_categorical(df_tr, cfg), scaler, cfg)
        df_va_enc = P.apply_scaler(P.encode_categorical(df_va, cfg), scaler, cfg)
        X_tr = torch.tensor(df_tr_enc[feat_cols].values, dtype=torch.float32).to(device)
        X_va = torch.tensor(df_va_enc[feat_cols].values, dtype=torch.float32).to(device)
        y_tr = torch.tensor(df_tr[cfg["columns"]["target_plaque"]].values,
                            dtype=torch.float32).unsqueeze(1).to(device)
        y_va = df_va[cfg["columns"]["target_plaque"]].values

        n_neg = int((y_tr == 0).sum()); n_pos = int((y_tr == 1).sum())
        pw = torch.tensor([n_neg / max(n_pos, 1) * 2.0], device=device)
        model = TabularClassifier(in_dim=in_dim, hidden=(64, 32), dropout=0.3).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        crit = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

        best_prob, best_score = None, -1
        patience, patience_counter = 7, 0
        for _ in range(epochs):
            model.train(); opt.zero_grad()
            loss = crit(model(X_tr), y_tr)
            loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                prob = torch.sigmoid(model(X_va)).squeeze(1).cpu().numpy()
            m = classification_metrics(y_va, prob, threshold=0.3)
            score = m.get("pr_auc", 0.0)
            if score > best_score:
                best_score = score; best_prob = prob; patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        results.append(classification_metrics(y_va, best_prob, threshold=0.3))
    return results


def main():
    cfg = P.load_config(str(PROJECT_ROOT / "configs/config.yaml"))
    df = P.load_dataframe(cfg, project_root=str(PROJECT_ROOT))
    folds = stratified_folds(df, cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    results = {}

    print("\n--- XGBoost (all features) ---")
    r = run_tree_cv(df, folds, cfg, "xgboost")
    results["xgboost_all"] = aggregate_folds(r)
    print(json.dumps(results["xgboost_all"], indent=2))

    print("\n--- LightGBM (all features) ---")
    r = run_tree_cv(df, folds, cfg, "lightgbm")
    results["lightgbm_all"] = aggregate_folds(r)
    print(json.dumps(results["lightgbm_all"], indent=2))

    print("\n--- LDL-only (XGBoost) ---")
    r = run_tree_cv(df, folds, cfg, "xgboost", feature_fn=ldl_only_features)
    results["xgboost_ldl_only"] = aggregate_folds(r)
    print(json.dumps(results["xgboost_ldl_only"], indent=2))

    print("\n--- Lipid panel (XGBoost, no Lp(a)) ---")
    r = run_tree_cv(df, folds, cfg, "xgboost", feature_fn=lipid_panel_features)
    results["xgboost_lipid_panel"] = aggregate_folds(r)
    print(json.dumps(results["xgboost_lipid_panel"], indent=2))

    print("\n--- TabularMLP ---")
    r = run_mlp_cv(df, folds, cfg, epochs=100, lr=3e-4, device=device)
    results["tabular_mlp"] = aggregate_folds(r)
    print(json.dumps(results["tabular_mlp"], indent=2))

    out = PROJECT_ROOT / "notebooks/tabular_baseline_metrics.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nsaved={out}")


if __name__ == "__main__":
    main()
