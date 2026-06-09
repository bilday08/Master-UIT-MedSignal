# [M1] Tien xu ly du lieu bang (CSV) — lam sach, encode, chuan hoa.
# Module nay FUNCTIONAL (chay duoc) de lam nen cho ca nhom.
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    """Doc file cau hinh trung tam."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dataframe(cfg: dict, project_root: str = ".") -> pd.DataFrame:
    """Doc CSV 300 ca. Tra ve DataFrame nguyen ban (chua scale)."""
    csv_path = os.path.join(project_root, cfg["data"]["csv"])
    df = pd.read_csv(csv_path)
    assert len(df) == 300, f"Ky vong 300 ca, thuc te {len(df)}"
    return df


def encode_categorical(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Encode Sex: Male->1, Female->0. Tra ve df moi (khong sua tai cho)."""
    df = df.copy()
    df["Sex"] = (df["Sex"].astype(str).str.strip().str.lower() == "male").astype(float)
    return df


def encode_echo_label(value, cfg: dict) -> int:
    """Echogenicity -> nhan int. NaN hoac 'None' (ca am) -> -100 (ignore_index cho CrossEntropy)."""
    import math
    classes = cfg["labels"]["echo_classes"]          # ["Low","Intermediate","High"]
    none_token = cfg["labels"]["echo_none"]           # "None"
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return -100
    v = str(value).strip()
    if v == none_token or v == "nan":
        return -100
    return classes.index(v)


def feature_columns(cfg: dict) -> list[str]:
    """Danh sach cot dac trung dua vao model (numeric da scale + Sex)."""
    return list(cfg["columns"]["numeric"]) + list(cfg["columns"]["categorical"])


def fit_scaler(df_train: pd.DataFrame, cfg: dict):
    """
    Fit StandardScaler CHI tren tap train (tranh ro ri).
    Tra ve scaler da fit; chi scale cac cot numeric (Sex giu nguyen 0/1).
    """
    from sklearn.preprocessing import StandardScaler

    numeric = cfg["columns"]["numeric"]
    scaler = StandardScaler()
    scaler.fit(df_train[numeric].values)
    return scaler


def apply_scaler(df: pd.DataFrame, scaler, cfg: dict) -> pd.DataFrame:
    """Ap scaler len cot numeric, tra ve df moi da scale."""
    df = df.copy()
    numeric = cfg["columns"]["numeric"]
    df[numeric] = scaler.transform(df[numeric].values)
    return df


def parse_associated_images(value: str) -> list[str]:
    """'P003_IMT.png,P003_CCA_L1.png,...' -> list ten file (strip khoang trang)."""
    return [s.strip() for s in str(value).split(",") if s.strip()]


def split_imt_cca(image_names: list[str]) -> tuple[Optional[str], list[str]]:
    """Tach anh IMT (1 anh) va danh sach CCA (0 hoac 4 anh)."""
    imt = next((n for n in image_names if "_IMT" in n), None)
    cca = [n for n in image_names if "_CCA_" in n]
    return imt, cca


def compute_pos_weight(df: pd.DataFrame, cfg: dict) -> float:
    """pos_weight = n_neg / n_pos cho BCE (xu ly lech lop 205/95)."""
    y = df[cfg["columns"]["target_plaque"]].astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    return n_neg / max(n_pos, 1)
