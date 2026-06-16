# v4 — Uncertainty Weighting (Kendall et al. 2018)

## Thay đổi so với v2
- Loss weights: fixed `[plaque=1.0, echo=0.5, risk=0.5]` → **learnable `log_var` per task**
- Formula: `L = Σ [ L_i / (2·exp(log_var_i)) + 0.5·log_var_i ]`
- `log_var` được optimize cùng model parameters

## Kết quả (5-fold mean ± std)
| Metric | v2 | v3 | **v4** | Δ vs v2 |
|---|---|---|---|---|
| AUC-ROC | 0.732 | **0.736** | 0.720 ± 0.035 | -0.012 |
| PR-AUC | 0.660 | **0.669** | 0.658 ± 0.039 | -0.002 |
| Sensitivity | 0.537 | 0.663 | 0.590 ± 0.070 | +0.053 |
| Specificity | **0.868** | 0.761 | 0.829 ± 0.094 | -0.039 |
| F1 | 0.584 | **0.613** | 0.604 ± 0.038 | +0.020 |

## Task Risk Score
| Metric | v1 | v2 | v3 | **v4** |
|---|---|---|---|---|
| MAE | 0.525 | 0.526 | 0.509 | **0.507** |
| R² | -0.063 | -0.147 | -0.019 | **+0.014** |

## Learned log_var (trung bình cuối training)
- `log_var_plaque`: ~+0.04 → weight ≈ 0.96× (giảm nhẹ)
- `log_var_echo`: ~-0.04 → weight ≈ 1.04× (tăng nhẹ)
- `log_var_risk`: ~-0.04 → weight ≈ 1.04× (tăng nhẹ)

## Nhận xét
- `log_var` học được rất nhỏ (~±0.04) — uncertainty weighting không thay đổi nhiều so với fixed weights ở số epoch này
- AUC-ROC thấp hơn v3 — khi model còn underfitting, uncertainty weighting không giúp ích
- **Điểm sáng:** R²=+0.014 (duy nhất >0) — uncertainty weighting cải thiện task risk regression rõ rệt
- Cần train nhiều epoch hơn để log_var có thời gian hội tụ
