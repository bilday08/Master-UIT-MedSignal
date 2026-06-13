"""Grad-CAM + SHAP cho dashboard, boc helper co san cua M2 (src/eval/explain.py).

- Grad-CAM: heatmap tren anh IMT, giai thich vung anh model nhin khi du doan plaque.
- SHAP: chi so lipid nao day du doan plaque len/xuong (global importance).
"""
from __future__ import annotations

import os

# macOS: torch va xgboost moi cai nap 1 libomp -> segfault khi dung chung.
# Cho phep trung libomp truoc khi xgboost duoc import (trong shap_global).
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import io
from functools import lru_cache

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.data import preprocess as P
from src.eval.explain import gradcam_on_image

from m5.serving.common import PROJECT_ROOT, load_v3_config
from m5.serving.inference import get_predictor


class _ImtPlaqueWrapper(nn.Module):
    """Bien fusion model thanh f(imt) -> plaque logit, co dinh tabular + cca.

    Grad-CAM can mot model 1 input (anh) -> 1 output (diem plaque) de tinh heatmap.
    """

    def __init__(self, fusion, tab, cca, mask):
        super().__init__()
        self.fusion = fusion
        self.tab, self.cca, self.mask = tab, cca, mask

    def forward(self, imt):
        return self.fusion(self.tab, imt, self.cca, self.mask)["plaque"]


def gradcam_png(tabular: dict, imt_img: Image.Image, cca_imgs=None) -> bytes:
    """Sinh PNG overlay heatmap Grad-CAM tren anh IMT (theo task plaque)."""
    pred = get_predictor()
    device = torch.device("cpu")  # gradcam can backward, chay CPU cho on dinh
    model = pred.model.to(device)
    model.eval()

    tab = pred._tabular_tensor(tabular).to(device)
    imt = pred._img_tensor(imt_img).unsqueeze(0).to(device)  # [1,1,H,W]
    cca, mask = pred._cca_tensors(cca_imgs)

    wrapper = _ImtPlaqueWrapper(model, tab, cca.to(device), mask.to(device))
    target_layer = model.vision.imt_encoder.backbone.layer4  # resnet18 conv cuoi (v3)
    cam = gradcam_on_image(wrapper, imt, target_layer, device=str(device))  # [H,W] 0..1

    # Anh goc: bo normalize (mean .5 std .5) -> [0,1], chuyen 3 kenh.
    base = (imt[0, 0].detach().cpu().numpy() * 0.5 + 0.5).clip(0, 1)
    rgb = np.stack([base, base, base], axis=-1).astype(np.float32)
    overlay = show_cam_on_image(rgb, cam, use_rgb=True)  # uint8 [H,W,3]

    model.to(pred.device)  # tra model ve device cu cho /predict
    buf = io.BytesIO()
    Image.fromarray(overlay).save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=1)
def _shap_model():
    """Train + cache XGBoost tren 9 feature (gia tri tho) -> plaque. Tra (clf, feats).

    Dung SHAP goc cua XGBoost (pred_contribs) thay vi thu vien shap (tranh segfault
    voi xgboost 3.x tren macOS).
    """
    import xgboost as xgb

    cfg = load_v3_config()
    df = P.load_dataframe(cfg, str(PROJECT_ROOT))
    df = P.encode_categorical(df, cfg)
    feats = P.feature_columns(cfg)
    X = df[feats].astype(float).values
    y = df[cfg["columns"]["target_plaque"]].astype(int).values

    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.8, eval_metric="logloss",
    )
    clf.fit(X, y)
    return clf, feats


@lru_cache(maxsize=1)
def shap_global() -> list[dict]:
    """SHAP global: muc do anh huong (mean|SHAP|) cua tung chi so toi du doan plaque."""
    import xgboost as xgb

    clf, feats = _shap_model()
    cfg = load_v3_config()
    df = P.encode_categorical(P.load_dataframe(cfg, str(PROJECT_ROOT)), cfg)
    X = df[feats].astype(float).values
    dm = xgb.DMatrix(X, feature_names=feats)
    contribs = clf.get_booster().predict(dm, pred_contribs=True)
    imp = np.abs(contribs[:, :-1]).mean(axis=0)
    rows = [{"feature": f, "importance": round(float(v), 4)} for f, v in zip(feats, imp)]
    rows.sort(key=lambda r: r["importance"], reverse=True)
    return rows


def shap_local(tabular: dict) -> list[dict]:
    """SHAP cho 1 ca: chi so nao day du doan plaque LEN (value>0) hay XUONG (value<0).

    value tinh theo log-odds (SHAP goc XGBoost). Sap xep theo do lon.
    """
    import pandas as pd
    import xgboost as xgb

    clf, feats = _shap_model()
    cfg = load_v3_config()
    row = P.encode_categorical(pd.DataFrame([tabular]), cfg)
    x = row[feats].astype(float).values  # [1, n]
    dm = xgb.DMatrix(x, feature_names=feats)
    contribs = clf.get_booster().predict(dm, pred_contribs=True)[0]  # [n+1], cot cuoi=bias
    rows = [{"feature": f, "value": round(float(v), 4)} for f, v in zip(feats, contribs[:-1])]
    rows.sort(key=lambda r: abs(r["value"]), reverse=True)
    return rows
