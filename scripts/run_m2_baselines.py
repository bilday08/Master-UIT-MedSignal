"""
[M2] run_m2_baselines.py — Chạy toàn bộ M2 experiments và lưu kết quả.

Output: results/m2_baselines.json  (M4/M5 đọc file này cho ablation table)

Chạy trên Colab (đầy đủ):
    python scripts/run_m2_baselines.py

Chạy local (chỉ sklearn, không cần xgboost/lightgbm/torch):
    python scripts/run_m2_baselines.py --sklearn-only
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Project root detection ────────────────────────────────────────────────────
_here = Path(__file__).resolve().parent
for _p in [_here.parent, *_here.parents]:
    if (_p / "configs" / "config.yaml").exists():
        PROJECT_ROOT = str(_p); break
else:
    raise RuntimeError("Không tìm thấy project root (configs/config.yaml)")
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.data.preprocess import (
    load_config, load_dataframe, encode_categorical,
    fit_scaler, apply_scaler, feature_columns,
)
from src.data.splits import stratified_folds
from src.eval.metrics import classification_metrics, risk_metrics, aggregate_folds, discordance_subgroup

# ── Dependency detection ──────────────────────────────────────────────────────
def _has(*names):
    for n in names:
        try: __import__(n)
        except ImportError: return False
    return True

HAS_XGB   = _has("xgboost")
HAS_LGB   = _has("lightgbm")
HAS_TORCH = _has("torch")


# ── Scaler helpers (df đã encode — KHÔNG gọi encode_categorical lại) ──────────
def _scale(df_enc, scaler, cfg):
    """Apply scaler lên df đã encode."""
    return apply_scaler(df_enc, scaler, cfg)

def _fit_scale(df_enc, cfg):
    """Fit scaler và apply, trả về (scaler, df_scaled)."""
    scaler = fit_scaler(df_enc, cfg)
    return scaler, apply_scaler(df_enc, scaler, cfg)


# ── Model factories ───────────────────────────────────────────────────────────
def _clf(kind, pos_w):
    if kind == "logistic":
        return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    if kind == "xgboost" and HAS_XGB:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8, eval_metric="aucpr",
                             scale_pos_weight=pos_w, random_state=42, verbosity=0)
    if kind == "lightgbm" and HAS_LGB:
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                              subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                              class_weight={0: 1.0, 1: pos_w}, verbose=-1, random_state=42)
    # Fallback: sklearn GBM
    from sklearn.ensemble import GradientBoostingClassifier
    sw = np.where  # dùng sample_weight trong fit
    return ("gbm_sklearn", pos_w)  # sentinel tuple

def _reg(kind):
    if kind == "xgboost" and HAS_XGB:
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0)
    if kind == "lightgbm" and HAS_LGB:
        from lightgbm import LGBMRegressor
        return LGBMRegressor(n_estimators=300, max_depth=6, learning_rate=0.05,
                             subsample=0.8, subsample_freq=5, colsample_bytree=0.8,
                             verbose=-1, random_state=42)
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                     learning_rate=0.05, subsample=0.8, random_state=42)


def _fit_clf(clf_or_sentinel, X_tr, y_tr):
    """Fit classifier; xử lý sentinel cho sklearn GBM (cần sample_weight)."""
    if isinstance(clf_or_sentinel, tuple) and clf_or_sentinel[0] == "gbm_sklearn":
        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=300, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42)
        pos_w = clf_or_sentinel[1]
        sw = np.where(y_tr == 1, pos_w, 1.0)
        clf.fit(X_tr, y_tr, sample_weight=sw)
        return clf
    clf_or_sentinel.fit(X_tr, y_tr)
    return clf_or_sentinel


def _clf_label(kind):
    if kind == "xgboost":  return "XGBoost" if HAS_XGB else "GBM/sklearn (proxy)"
    if kind == "lightgbm": return "LightGBM" if HAS_LGB else "GBM/sklearn (proxy)"
    if kind == "mlp":      return "MLP/PyTorch" if HAS_TORCH else "Logistic (proxy)"
    return "Logistic"


# ── OOF predictions helper ────────────────────────────────────────────────────
def oof_classify(kind, df_enc, folds, cfg, feat_cols, ycol, feature_fn=None):
    """
    Sinh Out-of-Fold predictions cho toàn bộ 300 ca.
    Dùng cho discordance + Lp(a) stratified (tránh leakage train→test).
    """
    n = len(df_enc)
    oof_prob = np.full(n, np.nan)

    for tr_idx, va_idx in folds:
        df_tr = df_enc.iloc[tr_idx]
        df_va = df_enc.iloc[va_idx]
        _, df_tr_sc = _fit_scale(df_tr, cfg)
        scaler = fit_scaler(df_tr, cfg)
        df_va_sc = _scale(df_va, scaler, cfg)

        X_tr = feature_fn(df_tr_sc) if feature_fn else df_tr_sc[feat_cols].values
        X_va = feature_fn(df_va_sc) if feature_fn else df_va_sc[feat_cols].values
        y_tr = df_tr[ycol].values.astype(int)

        pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

        if kind == "mlp" and HAS_TORCH:
            prob = _train_mlp_clf(X_tr, y_tr, X_va, pos_w)
        else:
            clf = _clf(kind, pos_w)
            clf = _fit_clf(clf, X_tr, y_tr)
            prob = clf.predict_proba(X_va)[:, 1]

        oof_prob[va_idx] = prob

    return oof_prob


# ── CV classification ─────────────────────────────────────────────────────────
def cv_classify(kind, df_enc, folds, cfg, feature_fn=None):
    feat_cols = feature_columns(cfg)
    ycol = cfg["columns"]["target_plaque"]
    fold_metrics = []

    for tr_idx, va_idx in folds:
        df_tr = df_enc.iloc[tr_idx]
        df_va = df_enc.iloc[va_idx]
        scaler, df_tr_sc = _fit_scale(df_tr, cfg)
        df_va_sc = _scale(df_va, scaler, cfg)

        X_tr = feature_fn(df_tr_sc) if feature_fn else df_tr_sc[feat_cols].values
        X_va = feature_fn(df_va_sc) if feature_fn else df_va_sc[feat_cols].values
        y_tr = df_tr[ycol].values.astype(int)
        y_va = df_va[ycol].values.astype(int)
        pos_w = float((y_tr == 0).sum()) / max(float((y_tr == 1).sum()), 1)

        if kind == "mlp" and HAS_TORCH:
            prob = _train_mlp_clf(X_tr, y_tr, X_va, pos_w)
        else:
            clf = _clf(kind if kind != "mlp" else "logistic", pos_w)
            clf = _fit_clf(clf, X_tr, y_tr)
            prob = clf.predict_proba(X_va)[:, 1]

        fold_metrics.append(classification_metrics(y_va, prob))

    return aggregate_folds(fold_metrics)


def _train_mlp_clf(X_tr, y_tr, X_va, pos_w, epochs=80, lr=3e-4, bs=32):
    import torch, torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    from src.models.tabular import TabularClassifier

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TabularClassifier(in_dim=X_tr.shape[1], norm="batch").to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
    cw = np.where(y_tr == 1, pos_w, 1.0)
    sampler = WeightedRandomSampler(torch.tensor(cw, dtype=torch.double), len(cw), replacement=True)
    ds = TensorDataset(torch.from_numpy(X_tr.astype("float32")),
                       torch.from_numpy(y_tr.astype("float32")).unsqueeze(1))
    dl = DataLoader(ds, batch_size=bs, sampler=sampler)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); criterion(model(xb), yb).backward(); opt.step()
    model.eval()
    with torch.no_grad():
        prob = torch.sigmoid(model(torch.from_numpy(X_va.astype("float32")).to(device)))
        return prob.squeeze(-1).cpu().numpy()


# ── CV regression ─────────────────────────────────────────────────────────────
def cv_regress(kind, df_enc, folds, cfg):
    feat_cols = feature_columns(cfg)
    ycol = cfg["columns"]["target_risk"]
    fold_metrics = []

    for tr_idx, va_idx in folds:
        df_tr = df_enc.iloc[tr_idx]
        df_va = df_enc.iloc[va_idx]
        scaler, df_tr_sc = _fit_scale(df_tr, cfg)
        df_va_sc = _scale(df_va, scaler, cfg)

        X_tr = df_tr_sc[feat_cols].values.astype("float32")
        X_va = df_va_sc[feat_cols].values.astype("float32")
        y_tr = df_tr[ycol].values.astype("float32")
        y_va = df_va[ycol].values.astype("float32")

        if kind == "mlp" and HAS_TORCH:
            from src.models.baselines import TabularRiskRegressor, _train_mlp_regressor
            preds = _train_mlp_regressor(TabularRiskRegressor(in_dim=X_tr.shape[1]), X_tr, y_tr, X_va)
        else:
            reg = _reg(kind)
            reg.fit(X_tr, y_tr)
            preds = reg.predict(X_va)

        fold_metrics.append(risk_metrics(y_va, preds))

    return aggregate_folds(fold_metrics)


# ── Discordance analysis (dùng OOF) ──────────────────────────────────────────
def run_discordance(df_enc, folds, cfg) -> dict:
    """
    So sánh Sensitivity LDL-only vs Full-tabular trên nhóm discordance (n=18, 6 dương).
    Dùng OOF predictions để tránh leakage.
    ⚠️ n=18 — chỉ mang tính minh họa định tính.
    """
    feat_cols = feature_columns(cfg)
    ycol = cfg["columns"]["target_plaque"]
    df_discord = discordance_subgroup(df_enc, cfg)
    disc_idx_positions = [df_enc.index.get_loc(i) for i in df_discord.index]

    y_all = df_enc[ycol].values.astype(int)
    y_disc = y_all[disc_idx_positions]

    # OOF LDL-only
    feat_fn_ldl = lambda d: d[["LDL_C_mg_dL"]].values
    oof_ldl = oof_classify("logistic", df_enc, folds, cfg,
                            feat_cols, ycol, feature_fn=feat_fn_ldl)
    prob_ldl_disc = oof_ldl[disc_idx_positions]

    # OOF Full tabular (XGBoost > sklearn GBM)
    kind_full = "xgboost" if HAS_XGB else "logistic"
    oof_full = oof_classify(kind_full, df_enc, folds, cfg, feat_cols, ycol)
    prob_full_disc = oof_full[disc_idx_positions]

    result = {
        "n_total": len(df_discord),
        "n_positive": int(y_disc.sum()),
        "method": "OOF (out-of-fold) predictions — không leakage",
        "warning": "n=18 quá nhỏ — kết quả chỉ mang tính minh họa định tính.",
        "full_model_used": _clf_label(kind_full),
    }

    if len(set(y_disc)) < 2:
        result["note"] = "Không đủ cả 2 lớp trong discordance group để tính AUC."
    else:
        m_ldl  = classification_metrics(y_disc, prob_ldl_disc)
        m_full = classification_metrics(y_disc, prob_full_disc)
        result["ldl_only_logistic"]  = m_ldl
        result["full_tabular"]       = m_full
        result["sensitivity_lift"]   = round(m_full["sensitivity"] - m_ldl["sensitivity"], 4)
        result["pr_auc_lift"]        = round(m_full["pr_auc"] - m_ldl["pr_auc"], 4)

    # Case listing
    id_col = cfg["columns"]["id"]
    result["cases"] = df_enc.loc[df_discord.index, [
        id_col, "LDL_C_mg_dL", "Lp(a)_mg_dL", ycol, cfg["columns"]["target_echo"]
    ]].to_dict(orient="records")

    return result


# ── Lp(a) stratified AUC (dùng OOF) ─────────────────────────────────────────
def run_lpa_stratified(df_enc, folds, cfg) -> list[dict]:
    """AUC phân tầng theo Lp(a) tertile — dùng OOF predictions."""
    feat_cols = feature_columns(cfg)
    ycol = cfg["columns"]["target_plaque"]
    lpa_col = "Lp(a)_mg_dL"

    kind = "xgboost" if HAS_XGB else "logistic"
    oof_prob = oof_classify(kind, df_enc, folds, cfg, feat_cols, ycol)
    y_all = df_enc[ycol].values.astype(int)

    t33, t67 = np.percentile(df_enc[lpa_col], [33.3, 66.7])
    tiers = [
        (f"Low  Lp(a) < {t33:.0f} mg/dL",
         df_enc[lpa_col].values <  t33),
        (f"Mid  {t33:.0f} ≤ Lp(a) < {t67:.0f} mg/dL",
         (df_enc[lpa_col].values >= t33) & (df_enc[lpa_col].values < t67)),
        (f"High Lp(a) ≥ {t67:.0f} mg/dL",
         df_enc[lpa_col].values >= t67),
    ]

    rows = []
    for label, mask in tiers:
        y_t = y_all[mask]; p_t = oof_prob[mask]
        n_pos = int(y_t.sum())
        if len(set(y_t)) < 2:
            rows.append({"tier": label, "n": int(mask.sum()), "n_pos": n_pos,
                         "auc_roc": None, "pr_auc": None,
                         "note": "Không đủ 2 lớp để tính AUC"})
        else:
            m = classification_metrics(y_t, p_t)
            rows.append({"tier": label, "n": int(mask.sum()), "n_pos": n_pos,
                         "auc_roc": m["auc_roc"], "pr_auc": m["pr_auc"],
                         "sensitivity": m["sensitivity"], "specificity": m["specificity"]})
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────
def main(sklearn_only: bool = False):
    global HAS_XGB, HAS_LGB, HAS_TORCH
    if sklearn_only:
        HAS_XGB = HAS_LGB = HAS_TORCH = False
        print("[INFO] sklearn-only mode")

    cfg = load_config("configs/config.yaml")
    df_enc = encode_categorical(load_dataframe(cfg), cfg)  # encode 1 lần duy nhất
    folds  = stratified_folds(df_enc, cfg)

    feat_fn_ldl   = lambda d: d[["LDL_C_mg_dL"]].values
    feat_fn_lipid = lambda d: d[["LDL_C_mg_dL","ApoB_mg_dL","Triglyceride_mg_dL",
                                  "Total_Cholesterol_mg_dL","Non_HDL_mg_dL"]].values

    results: dict = {
        "meta": {},
        "classification": {},
        "regression": {},
        "discordance": {},
        "lpa_stratified": [],
    }

    # ── Classification ────────────────────────────────────────────────────────
    print("\n=== Classification (Plaque_present, 5-fold) ===")
    clf_tasks = [
        ("ldl_only_logistic",    "logistic",  feat_fn_ldl,   "LDL-C only"),
        ("lipid_panel_logistic", "logistic",  feat_fn_lipid, "Lipid panel (no Lp(a))"),
        ("full_logistic",        "logistic",  None,          "Full features (logistic)"),
        ("xgboost",              "xgboost",   None,          _clf_label("xgboost")),
        ("lightgbm",             "lightgbm",  None,          _clf_label("lightgbm")),
        ("mlp",                  "mlp",       None,          _clf_label("mlp")),
    ]
    for key, kind, ffn, label in clf_tasks:
        print(f"  [{label}]...", end=" ", flush=True)
        agg = cv_classify(kind, df_enc, folds, cfg, feature_fn=ffn)
        results["classification"][key] = {"label": label, **agg}
        print(f"PR-AUC={agg['pr_auc']['mean']:.3f}±{agg['pr_auc']['std']:.3f}  "
              f"Sens={agg['sensitivity']['mean']:.3f}  AUC={agg['auc_roc']['mean']:.3f}")

    # ── Regression ────────────────────────────────────────────────────────────
    print("\n=== Regression (Baseline_Risk_Score, 5-fold) ===")
    reg_tasks = [
        ("xgboost",  "xgboost",  _clf_label("xgboost")),
        ("lightgbm", "lightgbm", _clf_label("lightgbm")),
        ("mlp",      "mlp",      _clf_label("mlp")),
    ]
    for key, kind, label in reg_tasks:
        print(f"  [{label}]...", end=" ", flush=True)
        agg = cv_regress(kind, df_enc, folds, cfg)
        results["regression"][key] = {"label": label, **agg}
        print(f"MAE={agg['mae']['mean']:.4f}±{agg['mae']['std']:.4f}  "
              f"R²={agg['r2']['mean']:.4f}±{agg['r2']['std']:.4f}")

    # ── Discordance ───────────────────────────────────────────────────────────
    print("\n=== Discordance Analysis (OOF, không leakage) ===")
    disc = run_discordance(df_enc, folds, cfg)
    results["discordance"] = disc
    print(f"  n={disc['n_total']} ca, {disc['n_positive']} dương")
    if "ldl_only_logistic" in disc:
        print(f"  LDL-only  Sens={disc['ldl_only_logistic']['sensitivity']:.3f}  "
              f"PR-AUC={disc['ldl_only_logistic']['pr_auc']:.3f}")
        print(f"  Full-tab  Sens={disc['full_tabular']['sensitivity']:.3f}  "
              f"PR-AUC={disc['full_tabular']['pr_auc']:.3f}  "
              f"(Sensitivity lift={disc['sensitivity_lift']:+.3f})")

    # ── Lp(a) stratified ─────────────────────────────────────────────────────
    print("\n=== Lp(a) Stratified AUC (OOF) ===")
    strat = run_lpa_stratified(df_enc, folds, cfg)
    results["lpa_stratified"] = strat
    for row in strat:
        auc = f"{row['auc_roc']:.3f}" if row.get("auc_roc") else "N/A"
        pra = f"{row['pr_auc']:.3f}"  if row.get("pr_auc")  else "N/A"
        print(f"  {row['tier']}: n={row['n']} n_pos={row['n_pos']}  "
              f"AUC-ROC={auc}  PR-AUC={pra}")

    # ── Meta ──────────────────────────────────────────────────────────────────
    results["meta"] = {
        "has_xgboost":  HAS_XGB,
        "has_lightgbm": HAS_LGB,
        "has_torch":    HAS_TORCH,
        "n_folds":      cfg["split"]["n_folds"],
        "n_samples":    len(df_enc),
        "note": "Chạy lại trên Colab với XGBoost+LightGBM+Torch để có kết quả đầy đủ." if not (HAS_XGB and HAS_LGB and HAS_TORCH) else "Full run.",
    }

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path("results") / "m2_baselines.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n✓ Saved → {out_path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sklearn-only", action="store_true")
    args = ap.parse_args()
    main(sklearn_only=args.sklearn_only)
