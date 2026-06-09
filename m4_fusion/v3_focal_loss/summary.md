# v3 — Focal Loss (γ=2.0) ✅ Best Version

## Thay đổi so với v2
- Loss plaque: `BCEWithLogitsLoss` → **`FocalLoss(gamma=2.0)`**
- Focal Loss: `L = (1-p_t)^γ × BCE` — downweight easy negatives (ca âm dễ phân biệt), focus vào hard cases

## Kết quả (5-fold mean ± std)
| Metric | v1 | v2 | **v3** | Δ vs v2 |
|---|---|---|---|---|
| AUC-ROC | 0.729 | 0.732 | **0.736 ± 0.036** | +0.004 |
| PR-AUC | 0.673 | 0.660 | **0.669 ± 0.035** | +0.009 |
| Sensitivity | 1.000 | 0.537 | **0.663 ± 0.118** | +0.126 |
| Specificity | 0.000 | 0.868 | 0.761 ± 0.153 | -0.107 |
| F1 | 0.481 | 0.584 | **0.613 ± 0.041** | +0.029 |
| Avg threshold | 0.300 | 0.732 | **0.552 ± 0.044** | -0.180 |

## Task khác
| Task | Metric | v3 |
|---|---|---|
| Echogenicity | Macro-F1 | 0.658 ± 0.378 |
| Risk Score | MAE | 0.509 ± 0.065 |
| Risk Score | R² | -0.019 ± 0.109 |

## Nhận xét
- **AUC-ROC và F1 tốt nhất** trong tất cả versions
- Sensitivity tăng +12.6% so v2 trong khi Specificity chỉ giảm 10.7% — trade-off tốt hơn cho bài toán sàng lọc y tế (ưu tiên không bỏ sót ca bệnh)
- Threshold ~0.55 tự nhiên hơn, gần điểm 0.5
- Loss giảm nhanh và ổn định hơn v2 (Focal học tốt hơn trên hard samples)
- **Khuyến nghị dùng v3 làm model chính trong báo cáo**
