# v5 — All Improvements Combined

## Thay đổi so với v3 (previous best)

| # | Cải tiến | Chi tiết |
|---|---|---|
| 1 | **custom_cnn** | 22.5M → **272,902** params (giảm 99.8%) |
| 2 | **tab_feat_dim** | 32 → **128** (cân bằng với imt_feat 128d) |
| 3 | **CrossAttention** | `concat(tab, img)` → `tab↔img cross-attn` + LayerNorm + residual |
| 4 | **SMOTE** | 240 → ~328 training samples/fold (+88 synthetic positives) |
| 5 | **CosineAnnealingLR** | lr: 3e-4 → 1e-5 trong 50 epochs |
| 6 | **Gradient clipping** | max_norm=1.0 |
| 7 | Focal Loss γ=2 | Giữ nguyên từ v3 |
| 8 | WeightedSampler | Giữ nguyên từ v2 |
| 9 | Youden threshold | Giữ nguyên từ v2 |

## Kết quả (5-fold mean ± std)

| Metric | v1 baseline | v3 (prev best) | **v5 all** | Δ vs v3 |
|---|---|---|---|---|
| AUC-ROC | 0.729 ± 0.049 | 0.736 ± 0.036 | 0.711 ± 0.029 | -0.025 |
| PR-AUC | 0.673 ± 0.041 | 0.669 ± 0.035 | 0.614 ± 0.040 | -0.055 |
| Sensitivity | 1.000 ± 0.000 | 0.663 ± 0.118 | 0.611 ± 0.127 | -0.052 |
| Specificity | 0.000 ± 0.000 | 0.761 ± 0.153 | **0.824 ± 0.121** | +0.063 |
| **F1** | 0.481 ± 0.000 | 0.613 ± 0.041 | **0.613 ± 0.042** | ≈0 |
| Threshold | 0.300 | 0.552 ± 0.044 | 0.632 ± 0.022 | — |

## Task khác

| Task | Metric | v1 | v3 | **v5** |
|---|---|---|---|---|
| Echogenicity | Macro-F1 | 0.670 | 0.658 | 0.488 |
| Risk Score | MAE | 0.525 | 0.509 | **0.492** |
| Risk Score | R² | -0.063 | -0.019 | **+0.085** |

## Nhận xét

**Điểm mạnh của v5:**
- **R² = +0.085** (tốt nhất mọi version, duy nhất có R²>0 đáng kể) — cải thiện task risk regression rõ rệt nhờ tabular dim cân bằng và CrossAttention
- **MAE = 0.492** (tốt nhất)
- **Specificity 0.824** (cân bằng tốt hơn v3: 0.761)
- **F1 = 0.613** (ngang v3, tốt nhất)
- Model params giảm từ 22.5M → 272K: training ổn định hơn (loss val std = 0.19 vs 1.69 ở v3)
- Threshold ổn định hơn: std 0.022 vs 0.044 ở v3

**Điểm yếu:**
- AUC-ROC 0.711 thấp hơn v3 (0.736) — CrossAttention chưa tận dụng được signal khi dataset quá nhỏ
- PR-AUC 0.614 thấp hơn v3 — SMOTE synthetic samples có thể gây noise cho vision branch (ảnh lấy từ random positive, không thực sự tương ứng tabular)
- Echogenicity Macro-F1 giảm (0.488 vs 0.658) — cần điều tra thêm

**Kết luận:**
- Nếu ưu tiên **plaque classification** (AUC-ROC, PR-AUC): dùng **v3** (Focal Loss)
- Nếu ưu tiên **risk regression** (R², MAE) + **balanced Sens/Spec**: dùng **v5**
- Có thể thử hybrid: v3 architecture + tab_dim=128 + CrossAttention nhưng bỏ SMOTE
