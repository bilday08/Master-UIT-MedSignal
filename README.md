# Master-UIT-MedSignal — Multimodal Carotid Atherosclerosis

Mô hình đa phương thức (tabular lâm sàng + ảnh siêu âm) chẩn đoán mảng xơ vữa động mạch cảnh và phân tầng nguy cơ, trên dataset giả lập `clinical_carotid_dataset_v3` (300 ca, 680 ảnh PNG 256×256 grayscale).

> 📋 Kế hoạch chi tiết, phân vai 5 thành viên, lộ trình 3 tuần: xem **[PROJECT_PLAN.md](PROJECT_PLAN.md)**.

## Dữ liệu (đã verify)

| | Giá trị thật |
|---|---|
| Số ca | 300 |
| Plaque_present | 0: **205** / 1: **95** (lệch lớp ~32% dương) |
| Plaque_echogenicity | None=205, Intermediate=40, Low=28, High=27 |
| Baseline_Risk_Category | Low=293, Moderate=7 → **suy biến, không dùng**; thay bằng hồi quy `Baseline_Risk_Score` |
| Ảnh | Control: 1 ảnh IMT/ca · Target: 5 ảnh (IMT + 4 CCA) → 680 ảnh |

⚠️ **Chống leakage:** số lượng ảnh = nhãn hoàn hảo (1↔Control, 5↔Target). Nhánh Vision chẩn đoán plaque **chỉ dùng ảnh IMT** (ai cũng có đúng 1); 4 ảnh CCA chỉ dùng cho task echogenicity.

## Cấu trúc

```
configs/config.yaml      # đường dẫn + siêu tham số (nguồn cấu hình duy nhất)
src/
  data/    preprocess · splits · dataset(+collate)   [M1]
  models/  tabular · baselines · vision · fusion      [M2/M3/M4]
  train/   losses · train                             [M4]
  eval/    metrics · explain                          [M5]
  demo/    app (Streamlit)                            [M5]
notebooks/ 00_setup → 04_eval (Colab, theo giai đoạn)
```

## Chạy trên Google Colab (khuyến nghị)

1. Mở `notebooks/00_setup_colab.ipynb` trong Colab.
2. Đặt `SOURCE = "drive"` (mount Google Drive) **hoặc** `"git"` (clone từ GitHub).
3. Runtime → Change runtime type → **GPU (T4)** → Run all.
4. Cell smoke test sẽ load thử 1 batch để xác nhận pipeline OK.

Sau đó chạy lần lượt `01_baseline_tabular` → `02_baseline_vision` → `03_fusion_multitask` → `04_eval_explain_demo`.

## Chạy local (tuỳ chọn)

```bash
pip install -r requirements.txt
python -c "from src.data.dataset import CarotidDataset; print('OK')"
python -m src.train.train --config configs/config.yaml      # khi M4 hoàn thiện
```

## Phân công

| Thành viên | Trách nhiệm chính |
|---|---|
| M1 | Data engineering, Dataset/DataLoader, fold split |
| M2 | Tabular branch (MLP) + baseline XGBoost/LightGBM |
| M3 | Vision branch (CNN encoder + attention pooling CCA) |
| M4 | Fusion + multi-task joint training |
| M5 | Đánh giá, Grad-CAM/SHAP, demo Streamlit |

Mỗi file `src/*.py` ghi rõ `# [Mx]` ở đầu và các `TODO` theo người phụ trách.
