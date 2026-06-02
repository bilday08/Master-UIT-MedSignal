# [M1] Xuat CSV da lam sach de KIEM TRA TRUC QUAN (khong dung cho training).
#
# Chay:
#     python -m src.data.export_clean
#     python -m src.data.export_clean --out outputs/clean_preview.csv
#
# LUU Y QUAN TRONG (chong leakage):
#   File nay CHI encode an toan (Sex -> 0/1, echo -> nhan int) + co tinh san pos_weight.
#   KHONG ap StandardScaler — vi scaler phai fit RIENG tung fold; mot file "da scale"
#   chung cho 300 ca se ro ri thong ke val sang train. Day chi la ban xem truc quan.
from __future__ import annotations

import argparse
import os
import sys

# Goc project = 3 cap tren file nay (src/data/export_clean.py -> src/data -> src -> ROOT).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chay duoc ca: `python -m src.data.export_clean` lan `python3 export_clean.py`.
try:
    from . import preprocess as P
except ImportError:
    if _PROJECT_ROOT not in sys.path:
        sys.path.insert(0, _PROJECT_ROOT)
    from src.data import preprocess as P


def build_clean_frame(cfg: dict, project_root: str = "."):
    """Tra ve DataFrame: cot goc + 2 cot encode (Sex_encoded, Echo_label)."""
    df = P.load_dataframe(cfg, project_root)
    out = df.copy()

    # Sex -> 0/1 (cot moi, giu nguyen cot goc de doi chieu).
    out["Sex_encoded"] = P.encode_categorical(df, cfg)["Sex"].astype(int)

    # Echogenicity -> nhan int (-100 = ca am / ignore khi tinh CrossEntropy).
    echo_col = cfg["columns"]["target_echo"]
    out["Echo_label"] = df[echo_col].apply(lambda v: P.encode_echo_label(v, cfg))

    return out


def main():
    ap = argparse.ArgumentParser(description="Xuat CSV da lam sach (xem truc quan).")
    ap.add_argument("--config", default=os.path.join(_PROJECT_ROOT, "configs/config.yaml"))
    ap.add_argument("--project-root", default=_PROJECT_ROOT)
    ap.add_argument("--out", default="outputs/clean_preview.csv")
    args = ap.parse_args()

    cfg = P.load_config(args.config)
    clean = build_clean_frame(cfg, args.project_root)

    out_path = os.path.join(args.project_root, args.out)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    clean.to_csv(out_path, index=False)

    # Tom tat nhanh de kiem tra.
    tcol = cfg["columns"]["target_plaque"]
    print(f"Da ghi {len(clean)} ca -> {out_path}")
    print(f"   - Cot: {len(clean.columns)} (goc + Sex_encoded + Echo_label)")
    print(f"   - Sex_encoded : {dict(clean['Sex_encoded'].value_counts())}")
    print(f"   - Plaque      : {dict(clean[tcol].astype(int).value_counts())}")
    print(f"   - Echo_label  : {dict(clean['Echo_label'].value_counts())}")
    print(f"   - pos_weight  : {P.compute_pos_weight(clean, cfg):.4f}")
    print("\n File chua scale (co chu dich) — scaler fit rieng tung fold khi train.")


if __name__ == "__main__":
    main()
