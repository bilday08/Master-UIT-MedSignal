# [M5] Giai thich mo hinh: Grad-CAM (anh) + SHAP (tabular). KHUNG.
from __future__ import annotations


def gradcam_on_image(model, image_tensor, target_layer, device="cpu"):
    """
    Grad-CAM cho nhanh Vision (IMT/CCA).
    Dung thu vien pytorch-grad-cam (grad-cam trong requirements).

    M5 TODO:
      - Chon target_layer = lop conv cuoi cua encoder (vd model.vision.imt_encoder.backbone.layer4).
      - Tra ve heatmap chong len anh goc 256x256.
    """
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image

    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=image_tensor.to(device))[0]
    return grayscale_cam  # [H,W] — M5 overlay len anh goc


def shap_on_tabular(model_or_predict_fn, background, samples):
    """
    SHAP cho nhanh Tabular (hoac XGBoost baseline).

    M5 TODO:
      - Voi XGBoost/LightGBM: dung shap.TreeExplainer.
      - Voi MLP: dung shap.DeepExplainer/KernelExplainer.
      - Ve summary_plot lam noi vai tro Lp(a)/ApoB vs LDL-C.
    """
    import shap

    explainer = shap.Explainer(model_or_predict_fn, background)
    return explainer(samples)
