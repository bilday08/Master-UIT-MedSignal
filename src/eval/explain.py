# [M5] Giai thich mo hinh: Grad-CAM (anh) + SHAP (tabular). KHUNG.


from __future__ import annotations

from typing import Optional
import numpy as np

from pytorch_grad_cam import GradCAM
import shap
import torch
import torch.nn as nn


def gradcam_on_image(model, image_tensor, target_layer, device="cpu"):
    """
    Grad-Cam for Vision branch (IMT/CCA).
    """

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor.to(device))[0]
    return grayscale_cam  # [H,W]


def shap_on_tabular(
        model_or_clf,
        background: np.ndarray,
        samples: np.ndarray,
        feature_names: Optional[list] = None,
        model_type: str = "auto",
        plot: bool = True,
):
    """
    SHAP for Tabular branch (9 feature).
    """

    if model_type == "auto":
        try:
            from xgboost import XGBClassifier, XGBRegressor
            from lightgbm import LGBMClassifier, LGBMRegressor
            _tree_types = (XGBClassifier, XGBRegressor, LGBMClassifier, LGBMRegressor)
        except ImportError:
            _tree_types = ()
        if isinstance(model_or_clf, _tree_types):
            model_type = "tree"
        else:
            model_type = "mlp"

    if model_type == "tree":
        explainer = shap.TreeExplainer(model_or_clf)
        shap_values_raw = explainer.shap_values(samples)

        if isinstance(shap_values_raw, list):
            shap_values = shap_values_raw[1]
        else:
            shap_values = shap_values_raw

    else:

        if isinstance(model_or_clf, nn.Module):
            model_or_clf.eval()

            def predict_fn(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    t = torch.from_numpy(x.astype(np.float32))
                    logits = model_or_clf(t)
                    prob = torch.sigmoid(logits).squeeze(-1).numpy()
                return prob
        else:
            predict_fn = model_or_clf

        explainer = shap.KernelExplainer(predict_fn, background)
        shap_values = explainer.shap_values(samples, nsamples=100)

    if plot:
        shap.summary_plot(
            shap_values,
            samples,
            feature_names=feature_names,
            show=True,
        )

    return shap_values, explainer


def shap_importance_df(shap_values: np.ndarray, feature_names: list) -> "pd.DataFrame":
    """
    Return DataFrame mean |SHAP| sorted descending — for report table and quick print.
    Highlight Lp(a) / ApoB vs LDL-C.
    """
    import pandas as pd

    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": imp})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
