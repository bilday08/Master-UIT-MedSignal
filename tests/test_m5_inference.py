"""Test inference M5: predict 1 ca tra ve dung schema + kieu, ke ca khi khong co CCA.

Yeu cau artifact da freeze (m5/serving/artifacts/). Neu chua co -> skip.
Chay:  .venv/bin/python -m pytest tests/test_m5_inference.py -v
"""
from __future__ import annotations

import pytest
from PIL import Image

from m5.serving.common import ARTIFACT_DIR

pytestmark = pytest.mark.skipif(
    not (ARTIFACT_DIR / "model.pth").exists(),
    reason="Chua freeze model — chay: .venv/bin/python -m m5.serving.freeze_model",
)

SAMPLE = {
    "Age": 62, "Lp(a)_mg_dL": 80, "ApoB_mg_dL": 110, "LDL_C_mg_dL": 120,
    "Triglyceride_mg_dL": 150, "Total_Cholesterol_mg_dL": 200,
    "Non_HDL_mg_dL": 150, "IMT_mm": 0.9, "Sex": "Male",
}


@pytest.fixture(scope="module")
def predictor():
    from m5.serving.inference import get_predictor
    return get_predictor()


def _img() -> Image.Image:
    return Image.new("L", (256, 256), color=128)


def test_predict_schema_without_cca(predictor):
    out = predictor.predict(SAMPLE, _img(), cca_imgs=None)
    assert set(out) >= {"plaque_prob", "plaque_label", "threshold", "echo_class", "risk_score"}
    assert 0.0 <= out["plaque_prob"] <= 1.0
    assert out["plaque_label"] in (0, 1)
    assert out["echo_class"] in ("Low", "Intermediate", "High")
    assert isinstance(out["risk_score"], float)


def test_predict_with_cca(predictor):
    out = predictor.predict(SAMPLE, _img(), cca_imgs=[_img(), _img()])
    assert 0.0 <= out["plaque_prob"] <= 1.0
    assert out["plaque_label"] == int(out["plaque_prob"] >= out["threshold"])


def test_female_sex_encodes(predictor):
    out = predictor.predict({**SAMPLE, "Sex": "Female"}, _img())
    assert 0.0 <= out["plaque_prob"] <= 1.0
