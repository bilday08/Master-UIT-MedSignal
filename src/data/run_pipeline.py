# [M1] Chay TOAN BO pipeline du lieu dau-cuoi theo dung thu tu:
#   preprocess (lam sach) -> splits (chia 5 fold) -> dataset (Tensor + batch)
#
# Muc dich: 1 lenh duy nhat de kiem chung ca pipeline M1 chay thong,
# in ro tung buoc + cong bo data-contract cho M2/M3/M4.
#
# Chay:
#     python -m src.data.run_pipeline
#     python -m src.data.run_pipeline --fold 0 --batch-size 16 --sampler --scan-images
from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

# Goc project = 3 cap tren file nay (src/data/run_pipeline.py -> src/data -> src -> ROOT).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Cho phep chay CA HAI cach:
#   1) python -m src.data.run_pipeline   (tu goc project — khuyen nghi)
#   2) python3 run_pipeline.py           (truc tiep trong src/data/)
try:
    from . import preprocess as P
    from .dataset import (
        CarotidDataset,
        collate_fn,
        make_weighted_sampler,
        scan_image_integrity,
    )
    from .splits import fold_summary, stratified_folds
except ImportError:
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.data import preprocess as P
    from src.data.dataset import (
        CarotidDataset,
        collate_fn,
        make_weighted_sampler,
        scan_image_integrity,
    )
    from src.data.splits import fold_summary, stratified_folds


def _hr(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


# --------------------------------------------------------------- BUOC 1
def step_preprocess(cfg, project_root):
    """Doc & lam sach CSV; encode Sex; cong bo cot dac trung."""
    _hr("BUOC 1 — PREPROCESS (doc & lam sach CSV)")
    df = P.load_dataframe(cfg, project_root)
    enc = P.encode_categorical(df, cfg)

    tcol = cfg["columns"]["target_plaque"]
    print(f"• So ca               : {len(df)}  (ky vong 300)")
    print(f"• Cot CSV             : {len(df.columns)}")
    print(f"• Plaque_present      : {dict(df[tcol].astype(int).value_counts())}  (0=Control,1=Target)")
    print(f"• Sex (Male->1/Female->0): {dict(enc['Sex'].astype(int).value_counts())}")
    print(f"• 'None' giu nguyen chuoi: {(df[cfg['columns']['target_echo']] == 'None').sum()} ca am")
    print(f"• Dac trung tabular ({len(P.feature_columns(cfg))}): {P.feature_columns(cfg)}")
    print(f"• pos_weight (n_neg/n_pos): {P.compute_pos_weight(df, cfg):.4f}")
    return df


# --------------------------------------------------------------- BUOC 2
def step_splits(df, cfg):
    """StratifiedKFold(5) theo Plaque_present, long can bang echogenicity."""
    _hr("BUOC 2 — SPLITS (StratifiedKFold 5, chong leakage)")
    folds = stratified_folds(df, cfg)
    print(fold_summary(df, folds, cfg).to_string(index=False))

    # Kiem tra phu kin + khong trung.
    import numpy as np
    all_val = np.concatenate([va for _, va in folds])
    assert len(all_val) == len(np.unique(all_val)) == len(df), "Fold bi trung/thieu index!"
    print(f"\n✓ 5 fold phu kin {len(df)} ca, val khong trung nhau, train∩val = ∅")
    return folds


# --------------------------------------------------------------- BUOC 3
def step_dataset(df, cfg, folds, project_root, fold_id, batch_size, use_sampler):
    """Fit scaler tren train fold -> CarotidDataset -> DataLoader -> 1 batch."""
    _hr(f"BUOC 3 — DATASET & COLLATE (fold {fold_id}, batch={batch_size})")
    tr_idx, va_idx = folds[fold_id]
    df_tr, df_va = df.iloc[tr_idx], df.iloc[va_idx]

    # Scaler fit CHI tren train (chong leakage).
    scaler = P.fit_scaler(P.encode_categorical(df_tr, cfg), cfg)
    print(f"• Scaler fit tren {len(df_tr)} ca train (val {len(df_va)} ca chi APPLY).")

    ds_tr = CarotidDataset(df_tr, cfg, scaler, project_root)
    ds_va = CarotidDataset(df_va, cfg, scaler, project_root)

    if use_sampler:
        sampler = make_weighted_sampler(df_tr, cfg)
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, sampler=sampler, collate_fn=collate_fn)
        print(f"• DataLoader train dung WeightedRandomSampler (can bang lop).")
    else:
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        print(f"• DataLoader train dung shuffle=True.")
    _ = DataLoader(ds_va, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    batch = next(iter(dl_tr))
    print("\n• Mot batch tra ve (data-contract):")
    print(f"    patient_id : list[str] (len={len(batch['patient_id'])})  vd {batch['patient_id'][:3]}")
    print(f"    tabular    : {tuple(batch['tabular'].shape)}  {batch['tabular'].dtype}")
    print(f"    imt_img    : {tuple(batch['imt_img'].shape)}  {batch['imt_img'].dtype}  range[{batch['imt_img'].min():.2f},{batch['imt_img'].max():.2f}]")
    print(f"    cca_imgs   : {tuple(batch['cca_imgs'].shape)}  {batch['cca_imgs'].dtype}")
    print(f"    cca_mask   : {tuple(batch['cca_mask'].shape)}  {batch['cca_mask'].dtype}")
    for k in ("plaque", "echo", "risk"):
        t = batch["labels"][k]
        print(f"    labels.{k:<6}: {tuple(t.shape)}  {t.dtype}")

    # Kiem chung chong leakage: co CCA <=> ca duong.
    has_cca = batch["cca_mask"].any(dim=1).int()
    plaque = batch["labels"]["plaque"].squeeze(1).int()
    assert torch.equal(has_cca, plaque), "cca_mask khong khop nhan plaque!"
    print("\n✓ cca_mask khop chinh xac nhan Plaque (co CCA <=> ca duong) — chong leakage OK.")
    return batch


# --------------------------------------------------------------- BUOC 4
def step_contract():
    """Cong bo data-contract chinh thuc."""
    _hr("BUOC 4 — CONG BO DATA-CONTRACT (cho M2/M3/M4)")
    print("""    batch = {
      "patient_id": list[str],
      "tabular":   Tensor[B, 9]            float32   (8 numeric da scale + Sex)
      "imt_img":   Tensor[B, 1, 256, 256]  float32   [0,1]   -> task PLAQUE
      "cca_imgs":  Tensor[B, 4, 1, 256, 256] float32 [0,1]   -> task ECHO
      "cca_mask":  Tensor[B, 4]            bool      (True=anh that)
      "labels": {
        "plaque": Tensor[B,1] float32  (0./1.)
        "echo":   Tensor[B,1] int64    (0/1/2 hoac -100=ignore)
        "risk":   Tensor[B,1] float32  (Baseline_Risk_Score)
      }
    }
    Chi tiet: xem src/data/README.md""")


def main():
    ap = argparse.ArgumentParser(description="Chay pipeline du lieu M1 dau-cuoi.")
    ap.add_argument("--config", default=os.path.join(_PROJECT_ROOT, "configs/config.yaml"))
    ap.add_argument("--project-root", default=_PROJECT_ROOT)
    ap.add_argument("--fold", type=int, default=0, help="Fold de demo (0..4)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--sampler", action="store_true", help="Dung WeightedRandomSampler")
    ap.add_argument("--scan-images", action="store_true", help="Quet toan ven 680 anh")
    args = ap.parse_args()

    cfg = P.load_config(args.config)

    df = step_preprocess(cfg, args.project_root)
    folds = step_splits(df, cfg)

    if args.scan_images:
        _hr("(tuy chon) QUET TOAN VEN ANH")
        rep = scan_image_integrity(df, cfg, args.project_root)
        print(f"• ok={rep['ok']}/{rep['total']} | missing={len(rep['missing'])} "
              f"| corrupt={len(rep['corrupt'])} | wrong_size={len(rep['wrong_size'])}")

    step_dataset(df, cfg, folds, args.project_root, args.fold, args.batch_size, args.sampler)
    step_contract()

    _hr("PIPELINE M1 HOAN CHINH")


if __name__ == "__main__":
    main()
