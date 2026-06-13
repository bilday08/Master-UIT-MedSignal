"""Inference cho demo live — bu diem chan #4 (M4 khong cung cap predict()).

Tai dung model v3 da freeze, ap CHINH XAC cung tien xu ly nhu luc train
(encode Sex + scale numeric, val-transform anh), tra ve ket qua 3 task cho 1 ca.

    from m5.serving.inference import get_predictor
    pred = get_predictor()
    out = pred.predict(tabular_dict, imt_img, cca_imgs)
"""
from __future__ import annotations

import json
from functools import lru_cache

import joblib
import pandas as pd
import torch
from PIL import Image

from src.data import preprocess as P
from src.models.fusion import MultimodalFusion

from m5.serving.common import ARTIFACT_DIR, make_transforms, select_device

K_MAX = 4  # so anh CCA toi da (khop collate_fn)


class Predictor:
    def __init__(self, artifact_dir=ARTIFACT_DIR):
        self.artifact_dir = artifact_dir
        meta = json.loads((artifact_dir / "threshold.json").read_text())
        self.cfg = meta["cfg"]
        self.feature_names = meta["feature_names"]
        self.echo_classes = meta["echo_classes"]
        self.threshold = float(meta["threshold"])
        self.img_size = int(self.cfg["data"]["image_size"])
        self.img_mode = self.cfg["data"]["image_mode"]

        self.scaler = joblib.load(artifact_dir / "scaler.joblib")
        self.device = select_device()
        self.model = MultimodalFusion(self.cfg, in_tab=len(self.feature_names)).to(self.device)
        state = torch.load(artifact_dir / "model.pth", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        self.transform = make_transforms(False, self.img_size)  # val-transform, KHONG augment

    # ------------------------------------------------------------------ tabular
    def _tabular_tensor(self, tabular: dict) -> torch.Tensor:
        """dict 9 feature tho -> FloatTensor[1,9] (encode Sex + scale numeric, dung thu tu)."""
        row = pd.DataFrame([tabular])
        row = P.encode_categorical(row, self.cfg)
        row = P.apply_scaler(row, self.scaler, self.cfg)
        vec = [float(row.iloc[0][c]) for c in self.feature_names]
        return torch.tensor([vec], dtype=torch.float32)

    # ------------------------------------------------------------------ images
    def _img_tensor(self, img: Image.Image) -> torch.Tensor:
        return self.transform(img.convert(self.img_mode))  # [1,H,W]

    def _cca_tensors(self, cca_imgs):
        """list PIL (0..4) -> (cca [1,4,1,H,W], mask [1,4])."""
        cca = torch.zeros((1, K_MAX, 1, self.img_size, self.img_size), dtype=torch.float32)
        mask = torch.zeros((1, K_MAX), dtype=torch.bool)
        for i, img in enumerate((cca_imgs or [])[:K_MAX]):
            cca[0, i] = self._img_tensor(img)
            mask[0, i] = True
        return cca, mask

    # ------------------------------------------------------------------ predict
    @torch.no_grad()
    def predict(self, tabular: dict, imt_img: Image.Image, cca_imgs=None) -> dict:
        tab = self._tabular_tensor(tabular).to(self.device)
        imt = self._img_tensor(imt_img).unsqueeze(0).to(self.device)  # [1,1,H,W]
        cca, mask = self._cca_tensors(cca_imgs)
        out = self.model(tab, imt, cca.to(self.device), mask.to(self.device))

        plaque_prob = float(torch.sigmoid(out["plaque"]).item())
        echo_idx = int(out["echo"].argmax(dim=1).item())
        risk_score = float(out["risk"].item())
        return {
            "plaque_prob": round(plaque_prob, 4),
            "plaque_label": int(plaque_prob >= self.threshold),
            "threshold": round(self.threshold, 4),
            "echo_class": self.echo_classes[echo_idx],
            "echo_note": "Chỉ có ý nghĩa khi phát hiện mảng xơ vữa.",
            "risk_score": round(risk_score, 4),
        }


@lru_cache(maxsize=1)
def get_predictor() -> Predictor:
    """Singleton — load artifact 1 lan, dung lai cho moi request."""
    return Predictor()
