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
    """Tong hop 4 model x 3 task tu JSON M2/M4 (so lieu 5-fold that, KHONG tu model demo).

    Tra ve: { plaque: [{model, auc_roc, pr_auc, ...}], echo: [...], risk: [...] }
    """
    rows_tab = _read_tabular_baselines()
    rows_fusion = _read_fusion()
    models = rows_tab + rows_fusion

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


def _read_tabular_baselines() -> list[dict]:
    path = PROJECT_ROOT / "notebooks" / "tabular_baseline_metrics.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    # Chi lay 3 model chinh cho bang ablation. ldl_only/lipid_panel thuoc discordance.
    label = {
        "xgboost_all": "Tabular-XGBoost",
        "lightgbm_all": "Tabular-LightGBM",
        "tabular_mlp": "Tabular-MLP",
    }
    out = []
    for key, name in label.items():
        if key in data:
            out.append({"model": name, "summary": data[key]})
    return out


def _read_fusion() -> list[dict]:
    """Lay v3 (model chinh) lam dong 'Multimodal'."""
    path = PROJECT_ROOT / "m4_fusion" / "v3_focal_loss" / "results.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    return [{"model": "Multimodal", "summary": data.get("summary", {})}]
