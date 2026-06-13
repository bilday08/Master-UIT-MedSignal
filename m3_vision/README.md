# M3 Vision Branch

Nhánh Vision — mã hóa ảnh siêu âm carotid thành đặc trưng cho plaque & echogenicity.

## Cấu trúc

```
m3_vision/
├── README.md                         ← Bạn đang ở đây
├── CHECKLIST.md                      Task tracking
├── notebooks/                        Colab notebook + hướng dẫn
│   ├── M3_Vision_Colab_Training.ipynb  Colab GPU training
│   └── HOW_TO_RUN.md                   Các bước chạy notebook
│
├── scripts/                          Scripts thực thi
│   ├── train_vision_baseline.py      Train 5-fold IMT-only baseline
│   ├── ablation_pooling.py           Ablation: Attention vs Mean Pooling
│   ├── validate_m3_pipeline.py       Validate data/mask/leakage
│   ├── generate_gradcam_examples.py  Tạo Grad-CAM heatmaps
│   └── preview_augmentations.py      Preview augmentation
│
├── docs/                             Tài liệu
│   ├── README.md                     Hướng dẫn đọc docs
│   ├── FILE_STRUCTURE.md             Sơ đồ cấu trúc chi tiết
│   ├── WORK_DONE.md                  Nhật ký công việc
│   ├── A_TO_Z_GUIDE.md               Hướng dẫn chạy lại từ A→Z
│   ├── RESULTS_SUMMARY.md            Tóm tắt kết quả
│   ├── REPORT_NOTES.md               Ghi chú cho report
│   ├── M3_VISION_REPORT.md           Report chính thức
│   └── M3_VISION_PRESENTATION.md     Nội dung slides (10 slides)
│
├── results/                          Outputs (JSON, PNG)
└── checkpoints/                      Model checkpoints từng fold
```

## Shared source (M3 đã chỉnh)

```
src/models/vision.py    ImageEncoder, VisionPlaqueClassifier, AttentionPool, VisionBranch
src/data/preprocess.py  Fix NaN handling (keep_default_na=False)
```

## Chạy nhanh

```bash
# Validate pipeline + anti-leakage
python3 m3_vision/scripts/validate_m3_pipeline.py

# Train baseline (smoke test)
python3 m3_vision/scripts/train_vision_baseline.py \
  --encoder custom_cnn --epochs 1 --folds 1 --batch-size 8 \
  --max-train-batches 1 --max-val-batches 1 \
  --output m3_vision/results/smoke_metrics.json

# Train chính thức (30 epochs + early stopping, dùng Colab)
python3 m3_vision/scripts/train_vision_baseline.py \
  --encoder custom_cnn --epochs 30 --folds 5 \
  --early-stop --early-stop-patience 7 \
  --checkpoint-dir m3_vision/checkpoints/custom_cnn_30epoch \
  --output m3_vision/results/vision_custom_cnn_30epoch_metrics.json
```

## Kết quả hiện tại

| Encoder | Epochs | AUC-ROC | PR-AUC | Ghi chú |
|---|---|---|---|---|
| Custom CNN | 5 | 0.707 ± 0.04 | 0.625 ± 0.03 | CPU, preliminary |
| ResNet-18 | 1 | 0.706 ± 0.03 | 0.614 ± 0.04 | CPU, quick check |

→ **Kết quả chính thức 30 epochs chờ chạy Colab.**
