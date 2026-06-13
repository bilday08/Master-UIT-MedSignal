# M5 — Next.js + shadcn Dashboard (Eval, Explainability & Demo)

**Ngày:** 2026-06-13 · **Owner:** Quang (M5) · **Branch:** `feat/m5`
**Thay thế:** Streamlit demo trong task brief gốc

---

## 1. Mục tiêu

Biến toàn bộ deliverable M5 (eval / ablation / discordance / explainability / demo) thành **một web dashboard** thay cho Streamlit:

- **Demo inference live** — nhập 9 chỉ số + upload ảnh IMT/CCA → plaque prob, echogenicity, risk score, Grad-CAM heatmap.
- **Results** — bảng ablation 4 model × 3 task, ROC/PR curves, discordance Lp(a).
- **Explainability** — SHAP (global + local), Grad-CAM gallery.

Phạm vi đã chốt: **full dashboard** (cả 3 trang), chấp nhận rủi ro scope với phương án cắt giảm ở Tuần 3 (xem §8).

## 2. Ràng buộc & quyết định kiến trúc

| Quyết định | Lựa chọn | Lý do |
|---|---|---|
| Serving model PyTorch | **FastAPI backend + Next.js frontend** (REST) | Next.js không chạy PyTorch; giữ Grad-CAM/SHAP ở Python |
| Model cho demo **live** | **v3 (Focal Loss)** | Import sạch `MultimodalFusion`; plaque AUC tốt nhất (0.736); là "model chính" theo `m4_fusion/SUMMARY.md` |
| Số liệu tĩnh | **Precompute → JSON/PNG** | Không tốn GPU lúc demo; frontend chỉ fetch |
| Deploy | **Local only** (`next dev` + `uvicorn`) | Demo luận văn — không cần auth/DB/Docker |

**Trung thực số liệu (bắt buộc):** model freeze train trên cả 300 ca **chỉ để demo tương tác**. Mọi metric trên dashboard (AUC, PR-AUC, mean±std) lấy từ **kết quả 5-fold đã có sẵn** của M2/M4, KHÔNG từ model demo (tránh leakage train=test). Discordance giữ nguyên cảnh báo n nhỏ (6 ca dương).

## 3. Thực trạng repo (đã verify trên `feat/m5`)

`feat/m5` = `main` + m2 (tabular/baselines) + m4 (fusion v1–v5, đã gồm m3 vision). Merge sạch trừ 1 conflict ở `encode_echo_label` (đã giải, giữ bản robust của main/m2).

**Chặn cứng cần giải trong Tuần 1:**
- **Không có checkpoint** nào trên đĩa (`.pth`, `checkpoints/` bị gitignore).
- M4 **không lưu scaler** (`scaler_tab` fit per-fold rồi vứt).
- M4 **không cung cấp** hàm `predict(...)`. Điểm chặn #4 trong task brief → M5 tự làm.
- Checkpoint M4 là **per-fold** (5 cái), không có model "final".

→ M5 sở hữu bước **freeze model** (§5).

## 4. Cấu trúc thư mục (thêm mới, không đụng M1–M4)

```
Master-UIT-MedSignal/
├── src/ ...                      # giữ nguyên
├── m5/serving/                   # Backend Python (FastAPI)
│   ├── freeze_model.py           #   train v3 trên 300 ca → bundle artifact
│   ├── artifacts/                #   model.pth + scaler.joblib + threshold.json + meta.json (gitignored)
│   ├── inference.py              #   predict(tabular_dict, imt_img, cca_imgs)
│   ├── explain.py                #   Grad-CAM + SHAP
│   ├── app.py                    #   FastAPI endpoints
│   ├── precompute/               #   script + output JSON/PNG cho số liệu tĩnh
│   └── requirements.txt          #   fastapi, uvicorn, shap, pytorch-grad-cam, joblib...
└── m5/web/                       # Frontend Next.js + shadcn
    ├── app/(demo|results|explain)/page.tsx
    ├── components/{ui,PredictionForm,ResultCard,AblationTable,MetricCurve,ShapPlot}.tsx
    └── lib/api.ts
```

## 5. Freeze model (m5/serving/freeze_model.py)

Chạy 1 lần, sinh `m5/serving/artifacts/`:

```
model.pth          # state_dict MultimodalFusion (v3 cfg)
scaler.joblib      # StandardScaler fit trên toàn 300 ca
threshold.json     # { threshold: <youden>, feature_names: [9 tên đúng thứ tự], cfg_snapshot: {...} }
meta.json          # { version: "v3", frozen_at, reference_metrics: {auc, pr_auc, ...} }
```

**Logic:**
1. Load cfg (v3 config). Dựng `MultimodalFusion(cfg, in_tab=9)` từ `src/models/fusion.py`.
2. Load 300 ca, `fit_scaler` trên toàn bộ, train tới hội tụ (early stopping theo train loss — đây là model deploy, không phải để đo).
3. Tính Youden threshold trên toàn tập (chỉ phục vụ cắt nhãn demo; threshold "thật" cho báo cáo lấy từ 5-fold).
4. Lưu `state_dict` + scaler + threshold + 9 feature names + cfg snapshot.

## 6. inference.py — hàm predict (bù điểm chặn M4 #4)

```python
predict(tabular_dict: dict, imt_img: PIL.Image, cca_imgs: list[PIL.Image] | None)
  -> { plaque_prob: float, plaque_label: 0|1, echo_class: str, risk_score: float, threshold: float }
```

- Tái dựng model từ cfg snapshot + load `model.pth`.
- Tabular: lấy đúng thứ tự `feature_names`, encode Sex, áp `scaler.joblib`.
- Ảnh: **val-transform** = grayscale 'L' → resize 256 → ToTensor (KHÔNG augment). CCA: `collate_fn` cho 1 ca (pad về K=4 + `cca_mask`); nếu không có CCA → K=0 mask toàn False.
- Forward → sigmoid(plaque), argmax(echo), risk thẳng. Cắt nhãn plaque theo `threshold`.

## 7. API & data flow

**Động (live, cần artifact):**
| Endpoint | Vào | Ra |
|---|---|---|
| `POST /predict` | 9 chỉ số + ảnh | `{plaque_prob, plaque_label, echo_class, risk_score, threshold}` |
| `POST /gradcam` | ảnh IMT/CCA | PNG heatmap overlay (hook conv cuối: v3 resnet18 `layer4`) |
| `POST /shap/local` | 9 chỉ số | SHAP values cho ca vừa nhập |

**Tĩnh (precompute → JSON/PNG, frontend chỉ GET):**
| Endpoint | Nguồn |
|---|---|
| `GET /ablation` | `notebooks/tabular_baseline_metrics.json` + `m4_fusion/*/results.json` → 4 model × 3 task |
| `GET /curves` | điểm ROC/PR precompute |
| `GET /shap/global` | SHAP summary (TreeExplainer cho XGB/LGBM, KernelExplainer cho MLP) |
| `GET /discordance` | ca LDL<130 & Lp(a)≥50, kèm cảnh báo n nhỏ |

`precompute/` đọc các JSON M2/M4 + dataset CSV, ghi ra `precompute/*.json` và `*.png`. FastAPI serve file tĩnh + 3 endpoint live.

## 8. Frontend — 3 trang + phasing

**Stack:** Next.js 14 (App Router) + TS + Tailwind + shadcn/ui + Recharts.

| Trang | Nội dung | Component chính |
|---|---|---|
| `/demo` | form 9 chỉ số + dropzone ảnh → ResultCard + Grad-CAM | `PredictionForm`, `ResultCard` |
| `/results` | ablation 4×3, ROC/PR, discordance | `AblationTable`, `MetricCurve` |
| `/explain` | SHAP global/local, Grad-CAM gallery | `ShapPlot` |

**Phasing 3 tuần:**

| Tuần | Backend | Frontend |
|---|---|---|
| 1 — Foundation | `freeze_model.py` + `inference.py` + `/predict` | scaffold + **Demo page** gọi `/predict` thật |
| 2 — Eval/ablation | `precompute/` + `/ablation` `/curves` `/discordance` | **Results page** |
| 3 — Explain + chốt | `explain.py` Grad-CAM + SHAP | **Explain page** + Grad-CAM trên Demo + polish |

**YAGNI / phương án cắt:** không auth/DB/Docker/SSR. Nếu Tuần 3 hụt giờ → Explain page fallback về **ảnh Grad-CAM/SHAP tĩnh precompute** thay vì interactive.

## 9. Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Freeze v3 train lâu / không hội tụ trên máy | early stopping; nếu cần, freeze 1 fold checkpoint thay vì train-all |
| Scope full dashboard vượt 3 tuần | phasing có thứ tự ưu tiên; Explain page cắt được |
| SHAP KernelExplainer chậm | precompute global; local giới hạn nsamples |
| Grad-CAM layer khác giữa v3/v5 | freeze cố định v3 → `layer4` xác định |
| Ảnh upload sai định dạng/kích thước | validate ở frontend + guard ở `inference.py` |

## 10. Ngoài phạm vi

Train lại model (M4 sở hữu); tối ưu kiến trúc fusion; hosting/CI/CD; mobile; multi-user. Báo cáo/slide cuối là deliverable riêng, không thuộc app này.
