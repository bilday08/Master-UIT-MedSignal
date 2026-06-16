# M2 — Tabular Branch & ML Baselines: Summary

> Dành cho các thành viên M3/M4/M5 cần nắm nhanh đầu ra của M2.

---

## Tổng quan

M2 xây dựng toàn bộ nhánh **dữ liệu bảng (tabular)** — chỉ từ 9 chỉ số xét nghiệm, không dùng ảnh. Mục tiêu kép:

1. **Mốc baseline** để M4 (multimodal) phải vượt qua.
2. **Phân tích lâm sàng**: chứng minh Lp(a) bổ sung thông tin mà LDL-C truyền thống bỏ sót.

Dataset: 300 ca, 95 dương (32%) — mất cân bằng lớp, dùng `scale_pos_weight` / `class_weight` bù trừ.

---

## Đầu vào

| Nhóm | Features |
|---|---|
| Lipid | LDL-C, ApoB, Triglyceride, Total Cholesterol, Non-HDL |
| Lp(a) | Lp(a) |
| Lâm sàng | Age, Sex (encoded: Male=1, Female=0) |
| Hình ảnh | IMT (đo số, không phải ảnh) |

**Config:** `configs/config.yaml` — tất cả tên cột, đường dẫn CSV, số fold đều lấy từ đây.

---

## Kiến trúc & Files

```
src/
├── models/
│   ├── tabular.py      — TabularMLP, TabularClassifier, TabularRiskRegressor
│   └── baselines.py    — XGBoost/LightGBM factories, run_cv_classifier(), run_cv_regressor()
├── eval/
│   └── explain.py      — shap_on_tabular(), shap_importance_df()  [dùng cho M5]
notebooks/
└── 01_baseline_tabular.ipynb   — chạy end-to-end toàn bộ M2
m2/
├── run_tabular_baseline.py     — script standalone (không cần Jupyter)
└── m2_baselines.json           — kết quả đã chạy (dùng cho M4/M5)
```

### TabularMLP (`src/models/tabular.py`)

```
[9 features] → Linear(64) → Norm → ReLU → Dropout
             → Linear(32) → Norm → ReLU → Dropout
             → Linear(feat_dim=32)   ← feature vector cho M4 Fusion
```

- `norm="layer"` (default) → an toàn cho batch_size=1 (Streamlit demo).
- `sex_embed_dim>0` → dùng `nn.Embedding` thay float 0/1 nếu cần.
- `TabularClassifier`: gắn thêm 1 linear → logit cho Plaque_present.
- `TabularRiskRegressor`: đầu ra 1 giá trị liên tục cho Baseline_Risk_Score.

### CV Protocol

5-fold **Stratified** (stratify theo plaque + echogenicity key) — scaler fit trên train fold, không leakage. Discordance và Lp(a) stratified dùng **OOF (out-of-fold) predictions** để tránh train-on-all → test-on-subset.

---

## Kết quả

### Classification — Plaque_present (5-fold, n=300)

| Model | Sensitivity | Specificity | AUC-ROC | PR-AUC |
|---|---|---|---|---|
| LDL-C only (Logistic) | 0.505 ±0.079 | 0.502 ±0.025 | 0.472 ±0.042 | 0.341 ±0.043 |
| Lipid panel, no Lp(a) | 0.495 ±0.079 | 0.546 ±0.040 | 0.549 ±0.032 | 0.386 ±0.042 |
| Full Logistic (9 features) | 0.632 ±0.067 | 0.673 ±0.091 | 0.714 ±0.035 | 0.606 ±0.070 |
| XGBoost | 0.411 ±0.052 | 0.824 ±0.073 | 0.656 ±0.035 | 0.608 ±0.048 |
| LightGBM | 0.432 ±0.077 | 0.805 ±0.087 | 0.628 ±0.048 | 0.550 ±0.051 |
| **MLP (TabularClassifier)** | **0.758 ±0.071** | 0.390 ±0.101 | **0.704 ±0.031** | **0.627 ±0.030** |

**Nhận xét:**
- LDL-C only AUC=0.472 ≈ random → khẳng định LDL đơn độc không đủ để sàng lọc.
- Thêm Lp(a) (Full Logistic) đẩy AUC-ROC từ 0.549 → 0.714 (+0.165).
- MLP dẫn đầu PR-AUC (0.627) — phù hợp dataset mất cân bằng.
- XGBoost conservative hơn (specificity 0.82, sensitivity 0.41) — ít bỏ sót người khoẻ.

### Regression — Baseline_Risk_Score (5-fold)

| Model | MAE | R² |
|---|---|---|
| XGBoost | 0.440 ±0.072 | 0.161 ±0.088 |
| LightGBM | 0.440 ±0.070 | 0.170 ±0.098 |
| MLP | 0.439 ±0.066 | 0.161 ±0.126 |

R²≈0.16 là thực tế — Risk Score là biến tổng hợp, tương quan thấp với từng chỉ số riêng lẻ.

### Discordance Analysis (OOF, n=18, 6 dương)

> **Nhóm Discordance** = LDL-C < 130 mg/dL VÀ Lp(a) ≥ 50 mg/dL — bệnh nhân "an toàn theo LDL" nhưng thực ra vẫn có nguy cơ.

| Model | Sensitivity | PR-AUC |
|---|---|---|
| LDL-only Logistic | 0.167 (1/6 ca) | 0.426 |
| XGBoost full-tabular | 0.333 (2/6 ca) | 0.570 |
| **Lift** | **+0.167 (+1 ca)** | **+0.144** |

Model có Lp(a) phát hiện gấp đôi ca dương trong nhóm này.
⚠️ n=18 quá nhỏ — chỉ dùng để minh hoạ định tính, không suy diễn thống kê.

### Lp(a) Stratified AUC (XGBoost OOF, n=300)

| Dải Lp(a) | n | n_pos | AUC-ROC | PR-AUC |
|---|---|---|---|---|
| Low < 21 mg/dL | 99 | 30 | 0.663 | 0.633 |
| Mid 21–34 mg/dL | 101 | 28 | 0.691 | 0.551 |
| High ≥ 34 mg/dL | 100 | 37 | 0.608 | 0.600 |

AUC ổn định ~0.61–0.69 ở cả 3 dải — model không bị overfit vào một vùng Lp(a) cụ thể.

---

## Đầu ra cho M4/M5

### Cho M4 (Fusion)

| Item | Mô tả |
|---|---|
| `TabularMLP` | Encoder xuất vector 32-d, dùng làm nhánh tabular trong Fusion |
| `m2/m2_baselines.json` | Số baseline đầy đủ — M4 cần vượt AUC-ROC 0.714, PR-AUC 0.627 |
| Feature contract | 9 features, thứ tự theo `feature_columns(cfg)`, đã StandardScale |

**Mốc M4 cần beat:** AUC-ROC > 0.714, PR-AUC > 0.627 (MLP tabular-only).

### Cho M5 (SHAP / Explainability)

```python
from src.eval.explain import shap_on_tabular, shap_importance_df

# Tự detect model type: XGBoost/LightGBM → TreeExplainer (nhanh)
#                        MLP PyTorch     → KernelExplainer (chậm hơn)
shap_vals, explainer = shap_on_tabular(
    clf,
    background=X[:50],
    samples=X,
    feature_names=feat_cols,
    plot=True,           # hiện summary plot ngay
)
df_importance = shap_importance_df(shap_vals, feat_cols)
```

---

## Chạy lại (nếu cần)

```bash
# Trên Google Colab (GPU):
# 1. Clone repo + git pull
# 2. Mở notebooks/01_baseline_tabular.ipynb
# 3. Runtime → Change runtime type → GPU
# 4. Chạy từ cell 1 xuống cuối (không cần bước preprocessing riêng)
# Kết quả tự lưu vào results/m2_baselines.json

# Hoặc script standalone:
python m2/run_tabular_baseline.py
```
