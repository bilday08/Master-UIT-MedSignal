# v2 — WeightedSampler + Optimal Threshold + pos_weight_scale=1.2

## Thay đổi so với v1
- `pos_weight_scale`: 2.0 → **1.2** (tránh collapse)
- `DataLoader`: `shuffle=True` → **`WeightedRandomSampler`**
- `threshold`: fixed 0.3 → **Youden Index per fold** (max Sensitivity + Specificity - 1)

## Kết quả (5-fold mean ± std)
| Metric | v1 | v2 | Δ |
|---|---|---|---|
| AUC-ROC | 0.729 ± 0.049 | **0.732 ± 0.037** | +0.003 |
| PR-AUC | 0.673 ± 0.041 | 0.660 ± 0.043 | -0.013 |
| Sensitivity | 1.000 ± 0.000 | 0.537 ± 0.117 | -0.463 |
| Specificity | 0.000 ± 0.000 | **0.868 ± 0.093** | +0.868 |
| F1 | 0.481 ± 0.000 | **0.584 ± 0.056** | +0.103 |
| Avg threshold | 0.300 | **0.732 ± 0.130** | — |

## Nhận xét
- Model không còn collapse — học được phân biệt 2 lớp thực sự
- Specificity tăng từ 0→0.868: bây giờ phân biệt được ca âm
- Threshold trung bình ~0.73 rất cao → model cần confidence cao mới predict dương
- PR-AUC giảm nhẹ do model predict thận trọng hơn (ít false positive)
