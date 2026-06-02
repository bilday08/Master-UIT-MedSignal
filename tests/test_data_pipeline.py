"""
Test suite tu dong cho luong du lieu M1 (Data Engineering & Pipeline).

Chay:
    pytest tests/test_data_pipeline.py -v
hoac:
    python -m pytest tests -v

Bao phu:
  1. Preprocessing  : encode Sex / echo, scaler khong dong vao Sex, pos_weight.
  2. Stratification : 5 folds, ti le plaque val <=2%, echo can bang, khong trung index.
  3. Dataset/Collate: data-contract (shape/dtype), ca Control vs Target, mask CCA.
  4. Chong leakage  : Dataset chi APPLY scaler, khong fit lai.
  5. Tien ich nang cao: WeightedRandomSampler, quet toan ven anh thuc te.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

# Cho phep `import src...` khi chay pytest tu bat ky dau.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data import preprocess as P
from src.data.dataset import (
    CarotidDataset,
    collate_fn,
    make_weighted_sampler,
    scan_image_integrity,
)
from src.data.splits import stratified_folds
from torch.utils.data import DataLoader

CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "config.yaml")


# ============================================================ fixtures
@pytest.fixture(scope="module")
def cfg():
    return P.load_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def df(cfg):
    return P.load_dataframe(cfg, PROJECT_ROOT)


@pytest.fixture(scope="module")
def folds(df, cfg):
    return stratified_folds(df, cfg)


@pytest.fixture(scope="module")
def train_scaler(df, cfg, folds):
    """Scaler fit CHI tren fold-0 train (dung cach chong leakage)."""
    tr_idx, _ = folds[0]
    df_tr = df.iloc[tr_idx]
    return P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg), tr_idx


# ============================================================ 1. PREPROCESSING
def test_load_300_cases(df):
    assert len(df) == 300


def test_encode_sex(df, cfg):
    enc = P.encode_categorical(df, cfg)
    assert set(np.unique(enc["Sex"].values)).issubset({0.0, 1.0})
    # Male -> 1.0, Female -> 0.0 (kiem tra dung ngu nghia tren vai dong dau).
    for raw, val in zip(df["Sex"].astype(str).str.lower(), enc["Sex"].values):
        assert val == (1.0 if raw.strip() == "male" else 0.0)


@pytest.mark.parametrize("raw,expected", [
    ("Low", 0), ("Intermediate", 1), ("High", 2),
    ("None", -100), ("", -100), ("nan", -100),
])
def test_encode_echo_label_strings(cfg, raw, expected):
    assert P.encode_echo_label(raw, cfg) == expected


def test_encode_echo_label_nan_and_none(cfg):
    # Phong thu: nhan NaN (float) / None (Python) deu -> -100 (khong ValueError).
    assert P.encode_echo_label(float("nan"), cfg) == -100
    assert P.encode_echo_label(None, cfg) == -100


def test_echo_distribution_matches_reality(df, cfg):
    enc = df[cfg["columns"]["target_echo"]].apply(lambda v: P.encode_echo_label(v, cfg))
    counts = {int(k): int(v) for k, v in enc.value_counts().items()}
    assert counts[-100] == 205        # ca am (None)
    assert counts[0] == 28            # Low
    assert counts[1] == 40            # Intermediate
    assert counts[2] == 27            # High


def test_scaler_does_not_touch_sex(df, cfg):
    enc = P.encode_categorical(df, cfg)
    scaler = P.fit_scaler(enc, cfg)
    scaled = P.apply_scaler(enc, scaler, cfg)
    # Sex KHONG nam trong numeric -> phai giu nguyen 0/1.
    assert np.array_equal(scaled["Sex"].values, enc["Sex"].values)
    assert set(np.unique(scaled["Sex"].values)).issubset({0.0, 1.0})


def test_scaler_standardizes_numeric(df, cfg):
    enc = P.encode_categorical(df, cfg)
    scaler = P.fit_scaler(enc, cfg)
    scaled = P.apply_scaler(enc, scaler, cfg)
    num = cfg["columns"]["numeric"]
    means = scaled[num].mean().values
    stds = scaled[num].std(ddof=0).values
    assert np.allclose(means, 0.0, atol=1e-6)
    assert np.allclose(stds, 1.0, atol=1e-6)


def test_pos_weight(df, cfg):
    pw = P.compute_pos_weight(df, cfg)
    assert pw == pytest.approx(205 / 95, rel=1e-6)   # ~2.16


# ============================================================ 2. STRATIFICATION
def test_five_folds(folds):
    assert len(folds) == 5


def test_folds_cover_all_no_overlap(folds):
    all_val = np.concatenate([va for _, va in folds])
    assert len(all_val) == 300
    assert len(np.unique(all_val)) == 300          # val folds doi ngau, phu kin 300 ca
    for tr, va in folds:
        assert len(np.intersect1d(tr, va)) == 0    # train/val khong giao nhau


def test_val_pos_rate_within_2pct(df, cfg, folds):
    tcol = cfg["columns"]["target_plaque"]
    y = df[tcol].astype(int).values
    overall = y.mean()
    for _, va in folds:
        assert abs(y[va].mean() - overall) <= 0.02


def test_echo_balanced_across_folds(df, cfg, folds):
    tcol = cfg["columns"]["target_plaque"]
    ecol = cfg["columns"]["target_echo"]
    per_class_counts = {c: [] for c in cfg["labels"]["echo_classes"]}
    for _, va in folds:
        sub = df.iloc[va]
        pos = sub[sub[tcol].astype(int) == 1]
        vc = pos[ecol].value_counts()
        for c in per_class_counts:
            per_class_counts[c].append(int(vc.get(c, 0)))
    # Moi lop chenh lech giua cac fold khong qua 2 ca (du lieu nho).
    for c, lst in per_class_counts.items():
        assert max(lst) - min(lst) <= 2, f"Echo '{c}' mat can bang giua folds: {lst}"


# ============================================================ 3. DATASET & COLLATE
def _control_target_ids(df, cfg):
    tcol = cfg["columns"]["target_plaque"]
    control_idx = df.index[df[tcol].astype(int) == 0][0]
    target_idx = df.index[df[tcol].astype(int) == 1][0]
    return int(control_idx), int(target_idx)


def test_sample_contract_shapes_dtypes(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    s = ds[0]
    assert set(s.keys()) == {"patient_id", "tabular", "imt_img", "cca_imgs", "labels"}
    assert isinstance(s["patient_id"], str)
    assert s["tabular"].shape == (9,) and s["tabular"].dtype == torch.float32
    assert s["imt_img"].shape == (1, 256, 256) and s["imt_img"].dtype == torch.float32
    assert s["labels"]["plaque"].dtype == torch.float32
    assert s["labels"]["echo"].dtype == torch.long
    assert s["labels"]["risk"].dtype == torch.float32
    # Anh nam trong [0,1].
    assert float(s["imt_img"].min()) >= 0.0 and float(s["imt_img"].max()) <= 1.0


def test_control_has_zero_cca(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    control_idx, _ = _control_target_ids(df, cfg)
    s = ds[control_idx]
    assert s["cca_imgs"].shape == (0, 1, 256, 256)   # Control: K=0


def test_target_has_four_cca(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    _, target_idx = _control_target_ids(df, cfg)
    s = ds[target_idx]
    assert s["cca_imgs"].shape == (4, 1, 256, 256)   # Target: K=4


def test_collate_batch_shapes_and_mask(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    control_idx, target_idx = _control_target_ids(df, cfg)
    batch = collate_fn([ds[control_idx], ds[target_idx]])

    assert batch["tabular"].shape == (2, 9)
    assert batch["imt_img"].shape == (2, 1, 256, 256)
    assert batch["cca_imgs"].shape == (2, 4, 1, 256, 256)
    assert batch["cca_mask"].shape == (2, 4) and batch["cca_mask"].dtype == torch.bool
    assert batch["labels"]["plaque"].shape == (2, 1)
    assert batch["labels"]["echo"].shape == (2, 1)
    assert batch["labels"]["risk"].shape == (2, 1)

    # Control (phan tu 0): mask toan False + vung pad bang 0.
    assert batch["cca_mask"][0].sum().item() == 0
    assert torch.count_nonzero(batch["cca_imgs"][0]).item() == 0
    # Target (phan tu 1): dung 4 True.
    assert batch["cca_mask"][1].sum().item() == 4


def test_dataloader_end_to_end(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    dl = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    batch = next(iter(dl))
    assert batch["tabular"].shape == (16, 9)
    assert batch["cca_imgs"].shape == (16, 4, 1, 256, 256)
    # mask True <=> dong do la ca duong (co CCA).
    tcol = cfg["columns"]["target_plaque"]
    y0 = df[tcol].astype(int).values[:16]
    has_cca = batch["cca_mask"].any(dim=1).numpy().astype(int)
    assert np.array_equal(has_cca, y0)


# ============================================================ 4. CHONG LEAKAGE
def test_dataset_applies_provided_scaler(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    assert ds.scaler is scaler                      # giu dung scaler duoc truyen vao
    expected = P.apply_scaler(P.encode_categorical(df, cfg), scaler, cfg)
    num = cfg["columns"]["numeric"]
    assert np.allclose(ds.df_scaled[num].values, expected[num].values)


def test_dataset_never_fits_scaler(df, cfg, train_scaler, monkeypatch):
    """Chong leakage: Dataset KHONG duoc goi fit_scaler luc khoi tao/lay mau."""
    scaler, _ = train_scaler

    def _boom(*a, **k):
        raise AssertionError("CarotidDataset KHONG duoc fit scaler (ro ri du lieu)!")

    monkeypatch.setattr(P, "fit_scaler", _boom)
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)   # khong duoc raise
    _ = ds[0]                                            # lay mau cung khong fit


# ============================================================ 5. TIEN ICH NANG CAO
def test_weighted_sampler_balances_classes(df, cfg):
    torch.manual_seed(0)
    sampler = make_weighted_sampler(df, cfg)
    idx = list(sampler)
    assert len(idx) == len(df)
    y = df[cfg["columns"]["target_plaque"]].astype(int).to_numpy()
    pos_frac = y[np.array(idx)].mean()
    # Lop goc 32% duong -> sau sampler phai xap xi 50/50.
    assert 0.42 <= pos_frac <= 0.58, f"pos_frac={pos_frac:.3f} chua can bang"


def test_image_error_policy_raise(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT, image_error_policy="raise")
    with pytest.raises(RuntimeError):
        ds._load_image("KHONG_TON_TAI_P999_IMT.png", None)


def test_image_error_policy_zero(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT, image_error_policy="zero")
    t = ds._load_image("KHONG_TON_TAI_P999_IMT.png", None)
    assert t.shape == (1, 256, 256)
    assert torch.count_nonzero(t).item() == 0           # tra ve anh 0, khong crash


def test_missing_imt_name_raises(df, cfg, train_scaler):
    scaler, _ = train_scaler
    ds = CarotidDataset(df, cfg, scaler, PROJECT_ROOT)
    with pytest.raises(ValueError):
        ds._load_image(None, None)


def test_scan_image_integrity_all_clean(df, cfg):
    report = scan_image_integrity(df, cfg, PROJECT_ROOT)
    assert report["total"] == 680                       # 205*1 + 95*5
    assert report["missing"] == []
    assert report["corrupt"] == []
    assert report["wrong_size"] == []
    assert report["ok"] == report["total"]
