from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.models.vision import VisionPlaqueClassifier
from m3_vision.train_vision_baseline import make_transforms


def target_layer_for(model: VisionPlaqueClassifier, encoder: str):
    if encoder == "custom_cnn":
        return model.encoder.backbone[6]
    if encoder == "resnet18":
        return model.encoder.backbone.layer4
    raise ValueError(f"encoder khong ho tro: {encoder}")


def load_model(checkpoint_path: str, encoder: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = VisionPlaqueClassifier(
        encoder=encoder,
        feat_dim=int(ckpt.get("feat_dim", 128)),
        pretrained=False,
        dropout=float(ckpt.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def collect_predictions(model, loader, device):
    rows = []
    with torch.no_grad():
        for batch in loader:
            probs = torch.sigmoid(model(batch["imt_img"].to(device))).squeeze(1).cpu()
            labels = batch["labels"]["plaque"].squeeze(1).cpu()
            for i, patient_id in enumerate(batch["patient_id"]):
                rows.append({
                    "patient_id": patient_id,
                    "prob": float(probs[i]),
                    "label": int(labels[i]),
                    "imt_img": batch["imt_img"][i:i + 1],
                })
    return rows


def choose_examples(rows, threshold: float):
    positives = [r for r in rows if r["label"] == 1]
    negatives = [r for r in rows if r["label"] == 0]
    true_pos = [r for r in positives if r["prob"] >= threshold]
    false_pos = [r for r in negatives if r["prob"] >= threshold]
    false_neg = [r for r in positives if r["prob"] < threshold]

    chosen = []
    if true_pos:
        chosen.append(("true_positive", max(true_pos, key=lambda r: r["prob"])))
    elif positives:
        chosen.append(("highest_positive", max(positives, key=lambda r: r["prob"])))

    if false_pos:
        chosen.append(("false_positive", max(false_pos, key=lambda r: r["prob"])))
    elif negatives:
        chosen.append(("highest_negative", max(negatives, key=lambda r: r["prob"])))

    if false_neg:
        chosen.append(("false_negative", min(false_neg, key=lambda r: r["prob"])))
    elif positives:
        chosen.append(("lowest_positive", min(positives, key=lambda r: r["prob"])))

    # Giu toi da 3 case khac patient.
    out = []
    seen = set()
    for name, row in chosen:
        if row["patient_id"] in seen:
            continue
        out.append((name, row))
        seen.add(row["patient_id"])
        if len(out) == 3:
            break
    return out


def compute_gradcam(model, target_layer, image, device):
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        image = image.to(device)
        logit = model(image)[0, 0]
        logit.backward()
    finally:
        h1.remove()
        h2.remove()

    acts = activations[0]            # [1,C,H,W]
    grads = gradients[0]             # [1,C,H,W]
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1)).squeeze(0)
    cam = cam - cam.min()
    cam = cam / cam.max().clamp_min(1e-6)
    cam = torch.nn.functional.interpolate(
        cam[None, None], size=image.shape[-2:], mode="bilinear", align_corners=False
    )[0, 0]
    return cam.cpu().numpy()


def save_overlay(row, cam, output_path: Path):
    img = row["imt_img"][0, 0].numpy()
    img = (img * 0.5) + 0.5
    img = np.clip(img, 0.0, 1.0)

    gray = np.stack([img, img, img], axis=-1)
    heat = np.stack([
        np.clip(1.5 - np.abs(4 * cam - 3), 0, 1),
        np.clip(1.5 - np.abs(4 * cam - 2), 0, 1),
        np.clip(1.5 - np.abs(4 * cam - 1), 0, 1),
    ], axis=-1)
    overlay = (0.55 * gray + 0.45 * heat).clip(0, 1)
    Image.fromarray((overlay * 255).astype(np.uint8)).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM examples for M3 IMT baseline.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--encoder", choices=["custom_cnn", "resnet18"], default="custom_cnn")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--output-dir", default="m3_vision/results/gradcam_examples")
    args = parser.parse_args()

    cfg = P.load_config("configs/config.yaml")
    df = P.load_dataframe(cfg)
    folds = stratified_folds(df, cfg)
    train_idx, val_idx = folds[args.fold]
    scaler = P.fit_scaler(P.encode_categorical(df.iloc[train_idx], cfg), cfg)
    val_ds = CarotidDataset(
        df.iloc[val_idx],
        cfg,
        scaler,
        transform=make_transforms(False, cfg["data"]["image_size"]),
    )
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, args.encoder, device)
    target_layer = target_layer_for(model, args.encoder)
    rows = collect_predictions(model, val_loader, device)
    examples = choose_examples(rows, threshold=cfg["eval"]["decision_threshold"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, row in examples:
        cam = compute_gradcam(model, target_layer, row["imt_img"], device)
        out = output_dir / f"{args.encoder}_fold{args.fold}_{name}_{row['patient_id']}.png"
        save_overlay(row, cam, out)
        print(f"saved={out}")


if __name__ == "__main__":
    main()
