"""Doc ca mau that tu dataset cho demo (nut 'Tai ca mau') + bang ablation.

Tach rieng phan doc du lieu tinh khoi app.py.
"""
from __future__ import annotations

import json
from functools import lru_cache

from src.data import preprocess as P

from m5.serving.common import PROJECT_ROOT, load_v3_config

NUMERIC = [
    "Age", "Lp(a)_mg_dL", "ApoB_mg_dL", "LDL_C_mg_dL", "Triglyceride_mg_dL",
    "Total_Cholesterol_mg_dL", "Non_HDL_mg_dL", "IMT_mm",
]


@lru_cache(maxsize=1)
def _df():
    cfg = load_v3_config()
    return P.load_dataframe(cfg, str(PROJECT_ROOT)), cfg


def images_dir() -> str:
    _, cfg = _df()
    return str(PROJECT_ROOT / cfg["data"]["images_dir"])


def _case(row, cfg) -> dict:
    names = P.parse_associated_images(row[cfg["columns"]["images"]])
    imt, cca = P.split_imt_cca(names)
    tabular = {c: (float(row[c]) if c != "Age" else int(row[c])) for c in NUMERIC}
    tabular["Sex"] = str(row["Sex"]).strip().capitalize()
    return {
        "patient_id": str(row[cfg["columns"]["id"]]),
        "tabular": tabular,
        "imt_image": imt,
        "cca_images": cca,
        "ground_truth": {
            "plaque": int(row[cfg["columns"]["target_plaque"]]),
            "echo": str(row[cfg["columns"]["target_echo"]]),
            "risk": float(row[cfg["columns"]["target_risk"]]),
        },
    }


def sample_cases(n_pos: int = 3, n_neg: int = 3) -> list[dict]:
    """Vai ca duong + am, tron deu, deterministic (khong random) de demo on dinh."""
    df, cfg = _df()
    tcol = cfg["columns"]["target_plaque"]
    pos = df[df[tcol] == 1].head(n_pos)
    neg = df[df[tcol] == 0].head(n_neg)
    cases = [_case(r, cfg) for _, r in pos.iterrows()]
    cases += [_case(r, cfg) for _, r in neg.iterrows()]
    return cases


def list_cases() -> list[dict]:
    """Danh sach NHE toan bo 300 ca (id + plaque + so anh CCA) cho dropdown chon."""
    df, cfg = _df()
    tcol = cfg["columns"]["target_plaque"]
    out = []
    for _, row in df.iterrows():
        names = P.parse_associated_images(row[cfg["columns"]["images"]])
        _, cca = P.split_imt_cca(names)
        out.append({
            "patient_id": str(row[cfg["columns"]["id"]]),
            "plaque": int(row[tcol]),
            "n_cca": len(cca),
        })
    return out


def get_case(pid: str) -> dict | None:
    """Chi tiet day du 1 ca theo patient_id (tabular + ten anh + ground truth)."""
    df, cfg = _df()
    match = df[df[cfg["columns"]["id"]].astype(str) == str(pid)]
    if match.empty:
        return None
    return _case(match.iloc[0], cfg)


def ablation_table() -> dict:
    """Tong hop cac model x 3 task tu JSON M2/M3/M4 (so lieu 5-fold that, KHONG tu model demo).

    Tra ve: { plaque: [{model, auc_roc, pr_auc, ...}], echo: [...], risk: [...] }
    """
    rows_tab = _read_tabular_baselines()
    rows_vision = _read_vision()
    rows_fusion = _read_fusion()
    models = rows_tab + rows_vision + rows_fusion

    plaque, echo, risk = [], [], []
    for m in models:
        s = m["summary"]
        plaque.append({
            "model": m["model"],
            "auc_roc": _mean(s, "auc_roc"),
            "pr_auc": _mean(s, "pr_auc"),
            "sensitivity": _mean(s, "sensitivity"),
            "specificity": _mean(s, "specificity"),
            "f1": _mean(s, "f1"),
        })
        echo.append({"model": m["model"], "macro_f1": _mean(s, "macro_f1")})
        risk.append({"model": m["model"], "mae": _mean(s, "mae"), "r2": _mean(s, "r2")})
    return {"plaque": plaque, "echo": echo, "risk": risk}


def _mean(summary: dict, key: str):
    v = summary.get(key)
    if isinstance(v, dict):
        return v.get("mean")
    return v


def _m2_baselines() -> dict:
    path = PROJECT_ROOT / "m2" / "m2_baselines.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _read_tabular_baselines() -> list[dict]:
    """3 model tabular chinh cho ablation: plaque tu classification, risk tu regression."""
    data = _m2_baselines()
    cls = data.get("classification", {})
    reg = data.get("regression", {})
    label = {
        "xgboost": "Tabular-XGBoost",
        "lightgbm": "Tabular-LightGBM",
        "mlp": "Tabular-MLP",
    }
    out = []
    for key, name in label.items():
        if key not in cls:
            continue
        summary = dict(cls[key])  # plaque metrics (sens/spec/f1/auc/pr-auc)
        if key in reg:  # them risk mae/r2
            summary["mae"] = reg[key].get("mae")
            summary["r2"] = reg[key].get("r2")
        out.append({"model": name, "summary": summary})
    return out


def _multimodal_discordance() -> dict | None:
    """Sens/Spec/F1 cua Multimodal (v3) tren 18 ca discordance, tu OOF cua M4."""
    path = PROJECT_ROOT / "m4_fusion" / "v3_focal_loss" / "discordance_oof.csv"
    if not path.exists():
        return None
    import csv
    rows = list(csv.DictReader(path.open()))
    pos = [r for r in rows if r["plaque_true"] == "1"]
    neg = [r for r in rows if r["plaque_true"] == "0"]
    tp = sum(1 for r in pos if r["plaque_pred"] == "1")
    fp = sum(1 for r in neg if r["plaque_pred"] == "1")
    tn = len(neg) - fp
    sens = tp / len(pos) if pos else None
    spec = tn / len(neg) if neg else None
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * prec * sens / (prec + sens)) if sens and (prec + sens) else 0.0
    return {
        "sensitivity": round(sens, 4) if sens is not None else None,
        "specificity": round(spec, 4) if spec is not None else None,
        "f1": round(f1, 4),
    }


def _discordance_cases() -> list[dict]:
    """18 ca discordance kem du doan Multimodal (tu OOF csv)."""
    path = PROJECT_ROOT / "m4_fusion" / "v3_focal_loss" / "discordance_oof.csv"
    if not path.exists():
        return []
    import csv
    out = []
    for r in csv.DictReader(path.open()):
        out.append({
            "patient_id": r["patient_id"],
            "ldl": float(r["LDL_C_mg_dL"]),
            "lpa": float(r["Lp(a)_mg_dL"]),
            "plaque_true": int(r["plaque_true"]),
            "plaque_pred": int(r["plaque_pred"]),
            "plaque_prob": round(float(r["plaque_prob"]), 4),
        })
    return out


def discordance_data() -> dict:
    """Phan tich discordance (LDL<130 & Lp(a)>=50): LDL-only vs Tabular vs Multimodal.

    Nguon: m2_baselines.json (M2) + discordance_oof.csv (M4). Trung thuc voi n nho.
    """
    m2 = _m2_baselines().get("discordance", {})
    comparison = []
    if "ldl_only_logistic" in m2:
        comparison.append({"model": "LDL-C only", **_pick(m2["ldl_only_logistic"])})
    if "full_tabular" in m2:
        comparison.append({"model": "Tabular (XGBoost)", **_pick(m2["full_tabular"])})
    mm = _multimodal_discordance()
    if mm:
        comparison.append({"model": "Multimodal", **mm})
    return {
        "n_total": m2.get("n_total"),
        "n_positive": m2.get("n_positive"),
        "warning": m2.get("warning"),
        "method": m2.get("method"),
        "comparison": comparison,
        "lpa_stratified": _m2_baselines().get("lpa_stratified", []),
        "cases": _discordance_cases(),
    }


def _pick(d: dict) -> dict:
    """Lay sens/spec/f1 (so phang) tu object metric cua M2."""
    return {k: d.get(k) for k in ("sensitivity", "specificity", "f1")}


def _read_vision() -> list[dict]:
    """Dong Vision-IMT-CNN (chi co task plaque; echo/risk de trong)."""
    path = PROJECT_ROOT / "notebooks" / "vision_custom_cnn_30epoch_metrics.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [{"model": "Vision-IMT-CNN", "summary": data.get("summary", {})}]


def _read_fusion() -> list[dict]:
    """Lay v3 (model chinh) lam dong 'Multimodal'."""
    path = PROJECT_ROOT / "m4_fusion" / "v3_focal_loss" / "results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [{"model": "Multimodal", "summary": data.get("summary", {})}]
