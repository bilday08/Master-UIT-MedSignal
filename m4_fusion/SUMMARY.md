# M4 Fusion — Báo cáo thực nghiệm

**Dataset:** 300 ca (205 âm tính / 95 dương tính với plaque)
**Encoder ảnh:** ResNet-18 (v1-v4) / custom_cnn (v5)
**Đánh giá:** 5-fold StratifiedKFold, kết quả mean +/- std
**Máy chạy:** RTX 3050 Ti Laptop, ngày 2026-06-09

---

## Cấu trúc thư mục

```
m4_fusion/
├── SUMMARY.md
├── v1_baseline/
├── v2_sampler_threshold/
├── v3_focal_loss/
├── v4_uncertainty/
└── v5_all_improvements/
```

Mỗi thư mục chứa: `config.yaml`, `train.py`, `results.json`, `summary.md`, `checkpoints/`.

---

## Vấn đề gốc (v1 baseline)

Model fusion gốc (`src/train/train.py`) được chạy với `pos_weight_scale=2.0` và threshold cố định `0.3`. Kết quả: model predict toàn dương tính — Sensitivity=1.0, Specificity=0.0, F1=0.481. AUC-ROC=0.729 cao nhưng vô nghĩa vì model không phân biệt được ca âm tính.

---

## Chi tiết từng version

### v1 — Baseline

**Điều chỉnh:** Không thay đổi so với script gốc. Chạy lại để có baseline đối chiếu.

**Cấu hình:**
- Encoder: ResNet-18 (22.5M params)
- Loss: BCE với pos_weight_scale=2.0
- Threshold: 0.3 cố định
- DataLoader: shuffle=True

**Kết quả:**
- AUC-ROC: 0.729 +/- 0.049
- F1: 0.481
- Sensitivity: 1.000 / Specificity: 0.000 (model collapse)
- R2: -0.063

**Kết luận:** pos_weight_scale=2.0 quá lớn, buộc model luôn predict dương tính. Cần sửa.

---

### v2 — WeightedSampler + Youden Threshold

**Điều chỉnh:**
1. `pos_weight_scale` giảm từ 2.0 xuống 1.2 — tránh collapse
2. Thay `shuffle=True` bằng `WeightedRandomSampler` — đảm bảo mỗi batch có đủ ca dương tính
3. Bỏ threshold cố định, thay bằng Youden Index tự động tính trên val set mỗi fold: `threshold = argmax(Sensitivity + Specificity - 1)`

**Kết quả:**
- AUC-ROC: 0.732 +/- 0.037
- F1: 0.584 +/- 0.056
- Sensitivity: 0.537 / Specificity: 0.868
- Threshold trung bình: 0.73
- R2: -0.068

**Kết luận:** Fix được model collapse. Specificity tăng từ 0 lên 0.868. Tuy nhiên Sensitivity chỉ 0.537 — model thiên về âm tính. Threshold 0.73 quá cao, model chưa tự tin với ca dương tính.

---

### v3 — Focal Loss (gamma=2.0)

**Điều chỉnh (thêm vào v2):**
1. Thay BCE bằng Focal Loss cho head plaque:
   - `p_t = prob * targets + (1-prob) * (1-targets)`
   - `focal_weight = (1 - p_t) ** gamma`
   - `loss = focal_weight * BCE` — downweight easy negatives, tập trung hard cases
2. gamma=2.0 (giá trị chuẩn từ RetinaNet paper)

**Kết quả:**
- AUC-ROC: 0.736 +/- 0.036 (tốt nhất mọi version)
- PR-AUC: 0.669 +/- 0.035 (tốt nhất mọi version)
- F1: 0.613 +/- 0.041 (tốt nhất cùng v5)
- Sensitivity: 0.663 / Specificity: 0.761
- Threshold trung bình: 0.552
- R2: -0.019

**Kết luận:** Focal Loss cải thiện cả AUC-ROC và cân bằng Sensitivity/Specificity. Threshold về mức tự nhiên hơn (0.55). Tuy nhiên val_loss có std=1.69 rất cao — do ResNet-18 (22.5M params) quá lớn cho 240 training samples, training không ổn định.

---

### v4 — Uncertainty Weighting (Kendall 2018)

**Điều chỉnh (thay v2, không dùng Focal Loss):**
1. Thay trọng số loss cố định [1, 0.5, 0.5] bằng learnable uncertainty weighting theo Kendall et al. 2018:
   - Mỗi task có `log_var` learnable
   - `loss_task = (1 / exp(log_var)) * loss + 0.5 * log_var`
   - Optimizer cập nhật cả model params lẫn log_var

**Kết quả:**
- AUC-ROC: 0.720 +/- 0.035
- F1: 0.604 +/- 0.038
- R2: +0.014 (duy nhất dương trong v1-v4)
- log_var học được dao động +/-0.04 (gần như không thay đổi so với cố định)

**Kết luận:** Uncertainty weighting không giúp nhiều khi model còn underfitting. log_var hội tụ về 0 nên gần bằng trọng số cố định. R2 dương nhẹ nhưng AUC thấp hơn v3. Không chọn version này.

---

### v5 — Tất cả cải tiến kết hợp

**Điều chỉnh (toàn bộ so với v3):**
1. **Encoder: ResNet-18 -> custom_cnn** (22.5M -> 272K params, giảm 99.8%)
   - Phù hợp hơn với dataset nhỏ (240 training samples/fold)
   - custom_cnn: 3x Conv2d(BN+ReLU) + AdaptiveAvgPool, feat_dim=128
2. **tabular.feat_dim: 32 -> 128**
   - Trước đây tabular (32d) bị lấn át bởi vision (128d) trong concat fusion
   - Cân bằng tỉ lệ 1:1 để fusion thực sự tận dụng cả hai modality
3. **CrossAttentionFusion thay concat đơn giản**
   - `nn.MultiheadAttention(dim=128, num_heads=4)` cho cả tab->img và img->tab
   - LayerNorm + residual connection
   - Output: cat(tab_out, img_out) -> [B, 256] -> Linear(256,128) -> head
4. **SMOTE trên tabular training fold**
   - imbalanced-learn SMOTE: 240 -> 328 samples/fold (+88 synthetic positives)
   - Ảnh cho synthetic samples: lấy ngẫu nhiên từ ca dương tính thực (xấp xỉ, không có ảnh synthetic thực sự)
5. **CosineAnnealingLR** (T_max=50, eta_min=1e-5)
6. **Gradient clipping** max_norm=1.0
7. Giữ nguyên Focal Loss gamma=2.0, WeightedRandomSampler, Youden threshold từ v3

**Kết quả:**
- AUC-ROC: 0.711 +/- 0.029
- PR-AUC: 0.614 +/- 0.040
- F1: 0.613 +/- 0.042
- Sensitivity: 0.611 / Specificity: 0.824
- Threshold: 0.632 +/- 0.022 (ổn định nhất)
- R2: +0.085 (tốt nhất mọi version)
- MAE: 0.492 (tốt nhất mọi version)
- val_loss std: 0.19 (ổn định hơn v3 rất nhiều, v3 là 1.69)

**Kết luận:** Switching sang custom_cnn giải quyết instability hoàn toàn. R2 lần đầu dương đáng kể nhờ tab_dim cân bằng + CrossAttention. AUC-ROC thấp hơn v3 một chút (0.711 vs 0.736) — SMOTE + ảnh xấp xỉ có thể thêm noise cho vision branch. Bù lại training ổn định và regression tốt hơn rõ rệt.

---

## Bảng so sánh tổng hợp

### Task: Plaque_present

| Version | AUC-ROC | PR-AUC | Sensitivity | Specificity | F1 | Threshold |
|---|---|---|---|---|---|---|
| v1 baseline | 0.729 +/- 0.049 | 0.673 +/- 0.041 | 1.000 | 0.000 | 0.481 | 0.300 (cố định) |
| v2 sampler+thr | 0.732 +/- 0.037 | 0.660 +/- 0.043 | 0.537 +/- 0.117 | 0.868 +/- 0.093 | 0.584 +/- 0.056 | 0.73 +/- 0.08 |
| v3 focal loss | 0.736 +/- 0.036 | 0.669 +/- 0.035 | 0.663 +/- 0.118 | 0.761 +/- 0.153 | 0.613 +/- 0.041 | 0.552 +/- 0.044 |
| v4 uncertainty | 0.720 +/- 0.035 | 0.658 +/- 0.039 | 0.590 +/- 0.070 | 0.829 +/- 0.094 | 0.604 +/- 0.038 | 0.581 +/- 0.047 |
| v5 all improv. | 0.711 +/- 0.029 | 0.614 +/- 0.040 | 0.611 +/- 0.127 | 0.824 +/- 0.121 | 0.613 +/- 0.042 | 0.632 +/- 0.022 |

### Task: Echogenicity + Risk Score

| Task | Metric | v1 | v3 | v5 |
|---|---|---|---|---|
| Echogenicity | Macro-F1 | 0.670 | 0.658 | 0.488 |
| Risk Score | MAE | 0.525 | 0.509 | 0.492 |
| Risk Score | R2 | -0.063 | -0.019 | +0.085 |

---

## Kết luận

**Best cho plaque classification (AUC-ROC, PR-AUC): v3**
Focal Loss cải thiện rõ khả năng phân biệt ca plaque. AUC-ROC 0.736, PR-AUC 0.669, F1 0.613. Dùng version này khi ưu tiên detect plaque chính xác.

**Best cho risk regression (R2, MAE): v5**
Architecture overhaul (custom_cnn + tab_dim=128 + CrossAttention) giải quyết vấn đề imbalance giữa tabular và vision. R2=+0.085 là duy nhất dương đáng kể. Training ổn định nhất (val_loss std 0.19).

**Cho báo cáo luận văn:**
Dùng v3 làm model chính của M4 (best plaque AUC). Report v5 như ablation study về architecture, chứng minh tầm quan trọng của cân bằng tabular/vision dim và lựa chọn encoder phù hợp với dataset nhỏ.

**Vấn đề còn mở:**
- Echogenicity Macro-F1 giảm ở v5 (0.488 vs 0.658) — CrossAttention chưa tối ưu cho task này
- SMOTE với ảnh xấp xỉ có thể gây noise cho vision branch — cần thử v5 không có SMOTE để cô lập ảnh hưởng
