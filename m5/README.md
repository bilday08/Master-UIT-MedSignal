# M5: Eval, Explainability & Demo

Lớp tiêu thụ cuối của pipeline (M1 đến M4). M5 không train model chính, chỉ **đo,
giải thích, và demo**: một web dashboard thay cho Streamlit trong task brief gốc.

Thiết kế chi tiết: [DESIGN.md](DESIGN.md).

## Cấu trúc

```
m5/
├── serving/    Backend Python (FastAPI): freeze model, inference, Grad-CAM, SHAP
│   └── artifacts/   Model đã freeze (gitignore, sinh lại bằng freeze_model.py)
└── web/        Frontend Next.js + shadcn: 3 trang Demo / Results / Explain
```

Tách 2 phần vì Next.js không chạy được PyTorch. Mọi thứ nặng (model, Grad-CAM, SHAP)
nằm ở `serving`; `web` chỉ render và gọi REST.

## Quan hệ với M1 đến M4

| Nhận từ | M5 dùng để |
|---|---|
| M1 (data, folds) | eval đúng split, không rò rỉ |
| M2 (tabular + baselines) | ablation, SHAP, đối chứng discordance |
| M3 (vision CNN) | ablation vision, Grad-CAM |
| M4 (fusion 3 head) | đối tượng eval chính + model demo (freeze v3) |

Số liệu đánh giá trên dashboard (AUC, PR-AUC, mean±std) lấy từ kết quả **5-fold**
của M2/M4, không phải từ model demo (tránh leakage train=test).

## Quickstart

Yêu cầu: Python 3.12 (torch chưa có wheel cho 3.14), Node 18+.

```bash
# 1. Môi trường (1 lần)
python3.12 -m venv .venv
.venv/bin/pip install -r m5/serving/requirements.txt
cd m5/web && npm install && cd ../..

# 2. Freeze model demo (1 lần, ~12 phút trên CPU)
.venv/bin/python -m m5.serving.freeze_model --epochs 6 --device cpu --workers 4

# 3. Chạy demo (2 process song song)
.venv/bin/uvicorn m5.serving.app:app --reload --port 8000   # backend
cd m5/web && npm run dev                                     # frontend (terminal khác)
```

Mở http://localhost:3000. Bấm **Tải ca mẫu** để dùng 1 bệnh nhân thật rồi **Dự đoán**.

## Trạng thái

| Phần | Trạng thái |
|---|---|
| Demo (form, inference live, ca mẫu) | Xong |
| Results (bảng ablation 4 model x 3 task) | Xong |
| Explain (Grad-CAM, SHAP) | Placeholder, Phase 3 |
| ROC/PR curves, discordance | Phase 2 |

Hướng dẫn backend chi tiết (API, freeze, test): [serving/README.md](serving/README.md).
