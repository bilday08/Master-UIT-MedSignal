# M5 — Eval, Explainability & Demo (Next.js + FastAPI)

Lớp tiêu thụ cuối của pipeline M1–M4: đo, giải thích, demo. Thay cho Streamlit.
Thiết kế: [docs/superpowers/specs/2026-06-13-m5-nextjs-dashboard-design.md](../docs/superpowers/specs/2026-06-13-m5-nextjs-dashboard-design.md).

```
m5/serving/   # Backend Python (FastAPI) — load model M4, inference, explain
m5/web/       # Frontend Next.js + shadcn — demo / results / explain
```

## Yêu cầu môi trường (1 lần)

```bash
# Backend: venv python 3.12 (torch chua co wheel cho 3.14)
python3.12 -m venv .venv
.venv/bin/pip install -r m5/serving/requirements.txt

# Frontend
cd m5/web && npm install
```

## Bước 1 — Freeze model (1 lần, sinh artifacts/)

```bash
# Nhanh (~12 phut tren CPU) — model demo dung duoc ngay:
.venv/bin/python -m m5.serving.freeze_model --epochs 6 --device cpu --workers 4

# Chat luong cao (~77 phut, giong bao cao) — chay qua dem:
.venv/bin/python -m m5.serving.freeze_model --epochs 30 --device cpu --workers 4
```

Sinh `m5/serving/artifacts/`: `model.pth`, `scaler.joblib`, `threshold.json`, `meta.json` (đã gitignore).

> ⚠️ Model freeze train trên cả 300 ca **chỉ để demo tương tác**. Metric báo cáo
> (AUC, PR-AUC, mean±std) lấy từ kết quả 5-fold của M2/M4, KHÔNG từ model này.

## Bước 2 — Chạy demo (2 process song song)

```bash
# Terminal 1 — backend (cong 8000)
.venv/bin/uvicorn m5.serving.app:app --reload --port 8000

# Terminal 2 — frontend (cong 3000)
cd m5/web && npm run dev
```

Mở http://localhost:3000 → trang `/demo`.

## Test

```bash
.venv/bin/python -m pytest tests/test_m5_inference.py -v   # can artifact da freeze
```

## API (Phase 1)

| Endpoint | Mô tả |
|---|---|
| `GET /health` | trạng thái + model_ready + meta |
| `POST /predict` | form: `tabular` (JSON 9 feature) + `imt_image` + `cca_images[]` → plaque/echo/risk |

Phase 2–3 (đang làm): `/ablation`, `/curves`, `/discordance`, `/gradcam`, `/shap`.
