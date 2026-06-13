# Hướng dẫn chạy M3 Colab Notebook

## Chuẩn bị

1. Nén project thành `.zip` (chỉ cần `configs/`, `src/`, `m3_vision/scripts/`, `clinical_carotid_dataset_v3/`):

```bash
# Từ repo root
zip -r Master-UIT-MedSignal.zip \
  configs/ src/ m3_vision/scripts/ \
  m3_vision/notebooks/M3_Vision_Colab_Training.ipynb \
  clinical_carotid_dataset_v3/
```

2. Mở [Google Colab](https://colab.research.google.com/) → chọn **Runtime → Change runtime type → T4 GPU**.

---

## Các bước chạy trong notebook

### Bước 1 — Kiểm tra GPU (Cell 0)
Chạy cell `!nvidia-smi` → xác nhận có Tesla T4.

### Bước 2 — Upload project (Cell 1–2)
Upload file `.zip` → giải nén:

```python
from google.colab import files
uploaded = files.upload()
!unzip -q Master-UIT-MedSignal.zip -d /content/
%cd /content/Master-UIT-MedSignal
```

### Bước 3 — Cài dependency (Cell 2)
```
!pip install -q scikit-learn torchvision pandas pyyaml Pillow
```

### Bước 4 — Sanity check (Cell 3)
Import + load 1 batch → xác nhận data pipeline OK.

### Bước 5 — Train Custom CNN 30 epochs (Cell 4–5)
- Thời gian: ~10-15 min trên T4.
- Output: `m3_vision/results/vision_custom_cnn_30epoch_metrics.json`
- Checkpoints: `m3_vision/checkpoints/custom_cnn_30epoch/`

### Bước 6 — Train ResNet-18 30 epochs (Cell 6–7)
- Thời gian: ~15-25 min trên T4.
- Output: `m3_vision/results/vision_resnet18_30epoch_metrics.json`
- Checkpoints: `m3_vision/checkpoints/resnet18_30epoch/`

### Bước 7 — Ablation Attention vs Mean Pooling (Cell 8–9)
- Thời gian: ~20-30 min trên T4.
- Output: `m3_vision/results/ablation_pooling_metrics.json`

### Bước 8 — Tổng hợp kết quả (Cell 10)
Bảng so sánh tất cả model đã chạy.

### Bước 9 — Download kết quả (Cell 11)
Tải JSON + checkpoints về máy → copy vào `m3_vision/results/` và `m3_vision/checkpoints/` trên local.

---

## Sau khi chạy xong

1. Copy JSON results vào `m3_vision/results/` trên local.
2. Copy checkpoints vào `m3_vision/checkpoints/` trên local.
3. Mở `m3_vision/docs/M3_VISION_REPORT.md` → điền kết quả vào các bảng placeholder (tìm `<!-- Điền bảng`).

## Lưu ý

- Nếu Colab ngắt kết nối giữa chừng → chạy lại từ Bước 2 (re-upload zip).
- Checkpoints được save mỗi fold → không mất tiến độ nếu ngắt giữa fold.
- Có thể giảm `--folds` từ 5 xuống 3 để test nhanh hơn.
