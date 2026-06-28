"""FastAPI backend cho M5 dashboard.

Phase 1: /health, /predict (live inference).
Phase 2-3 se them: /ablation, /curves, /discordance, /gradcam, /shap.

Chay:  .venv/bin/uvicorn m5.serving.app:app --reload --port 8000
"""
from __future__ import annotations

import io
import json
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image, UnidentifiedImageError

from m5.serving.common import ARTIFACT_DIR
from m5.serving.inference import get_predictor
from m5.serving.samples import (
    ablation_table,
    discordance_data,
    get_case,
    images_dir,
    list_cases,
    sample_cases,
)

app = FastAPI(title="MedSignal M5 API", version="0.1.0")

# Cho phep Next.js (localhost:3000) goi truc tiep luc dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 9 feature dau vao (khop feature_columns): 8 numeric + Sex.
NUMERIC_FEATURES = [
    "Age", "Lp(a)_mg_dL", "ApoB_mg_dL", "LDL_C_mg_dL", "Triglyceride_mg_dL",
    "Total_Cholesterol_mg_dL", "Non_HDL_mg_dL", "IMT_mm",
]


def _read_image(upload: UploadFile) -> Image.Image:
    try:
        return Image.open(io.BytesIO(upload.file.read()))
    except (UnidentifiedImageError, OSError) as err:
        raise HTTPException(400, f"Ảnh '{upload.filename}' không đọc được: {err}") from err


@app.get("/health")
def health() -> dict:
    ready = (ARTIFACT_DIR / "model.pth").exists()
    meta = {}
    meta_path = ARTIFACT_DIR / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    return {"status": "ok", "model_ready": ready, "meta": meta}


@app.post("/predict")
async def predict(
    tabular: str = Form(..., description="JSON 9 feature: 8 numeric + Sex (Male/Female)"),
    imt_image: UploadFile = File(..., description="Anh IMT (bat buoc)"),
    cca_images: list[UploadFile] = File(default=[], description="0-4 anh CCA (tuy chon)"),
) -> dict:
    try:
        tab = json.loads(tabular)
    except json.JSONDecodeError as err:
        raise HTTPException(400, f"`tabular` không phải JSON hợp lệ: {err}") from err

    missing = [f for f in NUMERIC_FEATURES + ["Sex"] if f not in tab]
    if missing:
        raise HTTPException(400, f"Thiếu chỉ số: {missing}")

    imt = _read_image(imt_image)
    cca = [_read_image(f) for f in cca_images] if cca_images else None

    try:
        predictor = get_predictor()
    except FileNotFoundError as err:
        raise HTTPException(
            503, "Mô hình chưa sẵn sàng, vui lòng thử lại sau."
        ) from err
    return predictor.predict(tab, imt, cca)


@app.get("/samples")
def samples() -> dict:
    """Vai ca that tu dataset (3 duong + 3 am) cho nut 'Tai ca mau'."""
    return {"cases": sample_cases()}


@app.get("/cases")
def cases() -> dict:
    """Danh sach nhe toan bo 300 ca (id + plaque + n_cca) cho dropdown."""
    return {"cases": list_cases()}


@app.get("/case/{pid}")
def case(pid: str) -> dict:
    """Chi tiet 1 ca theo patient_id (tabular + ten anh + ground truth)."""
    c = get_case(pid)
    if c is None:
        raise HTTPException(404, f"Không tìm thấy ca '{pid}'")
    return c


@app.get("/image/{name}")
def image(name: str) -> FileResponse:
    """Phuc vu 1 anh tu CAROTID_IMAGES (preview + tai ca mau). Chong path traversal."""
    safe = os.path.basename(name)
    path = os.path.join(images_dir(), safe)
    if not os.path.isfile(path):
        raise HTTPException(404, f"Không tìm thấy ảnh '{safe}'")
    return FileResponse(path, media_type="image/png")


@app.get("/ablation")
def ablation() -> dict:
    """Bang ablation cac model x 3 task (so lieu 5-fold tu M2/M3/M4)."""
    return ablation_table()


@app.get("/discordance")
def discordance() -> dict:
    """Phan tich discordance (LDL thap + Lp(a) cao): LDL-only vs Tabular vs Multimodal."""
    return discordance_data()


@app.post("/gradcam")
async def gradcam(
    tabular: str = Form(...),
    imt_image: UploadFile = File(...),
    cca_images: list[UploadFile] = File(default=[]),
) -> Response:
    """Heatmap Grad-CAM tren anh IMT (theo task plaque). Tra PNG."""
    from m5.serving.explain_service import gradcam_png

    try:
        tab = json.loads(tabular)
    except json.JSONDecodeError as err:
        raise HTTPException(400, f"`tabular` không phải JSON hợp lệ: {err}") from err
    imt = _read_image(imt_image)
    cca = [_read_image(f) for f in cca_images] if cca_images else None
    try:
        png = gradcam_png(tab, imt, cca)
    except FileNotFoundError as err:
        raise HTTPException(503, "Mô hình chưa sẵn sàng, vui lòng thử lại sau.") from err
    return Response(content=png, media_type="image/png")


@app.get("/shap/global")
def shap_global_endpoint() -> dict:
    """SHAP global: muc do anh huong cua tung chi so lipid toi du doan plaque."""
    from m5.serving.explain_service import shap_global

    return {"features": shap_global()}


@app.post("/shap/local")
async def shap_local_endpoint(tabular: str = Form(...)) -> dict:
    """SHAP cho 1 ca: chi so nao day du doan plaque len/xuong."""
    from m5.serving.explain_service import shap_local

    try:
        tab = json.loads(tabular)
    except json.JSONDecodeError as err:
        raise HTTPException(400, f"`tabular` không phải JSON hợp lệ: {err}") from err
    return {"features": shap_local(tab)}
