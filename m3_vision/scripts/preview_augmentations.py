from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from m3_vision.train_vision_baseline import make_transforms


def tensor_to_image(tensor) -> Image.Image:
    arr = tensor.squeeze(0).numpy()
    arr = (arr * 0.5) + 0.5
    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray((arr * 255).astype(np.uint8)).convert("RGB")


def main() -> None:
    cfg = P.load_config("configs/config.yaml")
    df = P.load_dataframe(cfg)
    names = P.parse_associated_images(df.loc[0, cfg["columns"]["images"]])
    imt_name, _ = P.split_imt_cca(names)
    image_path = Path(cfg["data"]["images_dir"]) / imt_name
    original = Image.open(image_path).convert(cfg["data"]["image_mode"])

    train_transform = make_transforms(True, cfg["data"]["image_size"])
    val_transform = make_transforms(False, cfg["data"]["image_size"])
    panels = [
        ("val", tensor_to_image(val_transform(original))),
        ("aug1", tensor_to_image(train_transform(original))),
        ("aug2", tensor_to_image(train_transform(original))),
        ("aug3", tensor_to_image(train_transform(original))),
    ]

    cell = cfg["data"]["image_size"]
    label_h = 24
    canvas = Image.new("RGB", (cell * len(panels), cell + label_h), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (label, img) in enumerate(panels):
        canvas.paste(img, (i * cell, label_h))
        draw.text((i * cell + 6, 5), label, fill=(0, 0, 0))

    out = Path("m3_vision/results/augmentation_preview.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    print(f"saved={out}")


if __name__ == "__main__":
    main()
