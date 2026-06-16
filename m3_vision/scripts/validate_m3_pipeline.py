from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import preprocess as P
from src.data.dataset import CarotidDataset, collate_fn
from src.data.splits import stratified_folds
from src.models.fusion import MultimodalFusion
from src.models.vision import AttentionPool, MaskedMeanPool, VisionBranch


def assert_close_zero(x: torch.Tensor, name: str) -> None:
    if not torch.allclose(x, torch.zeros_like(x), atol=1e-6):
        raise AssertionError(f"{name} khong bang 0 nhu ky vong")


def load_batch(cfg: dict):
    df = P.load_dataframe(cfg)
    train_idx, _ = stratified_folds(df, cfg)[0]
    scaler = P.fit_scaler(P.encode_categorical(df.iloc[train_idx], cfg), cfg)
    loader = DataLoader(
        CarotidDataset(df.iloc[train_idx], cfg, scaler),
        batch_size=16,
        shuffle=False,
        collate_fn=collate_fn,
    )
    return next(iter(loader))


def validate_dataset_contract(batch: dict) -> dict:
    assert tuple(batch["tabular"].shape) == (16, 9)
    assert tuple(batch["imt_img"].shape) == (16, 1, 256, 256)
    assert tuple(batch["cca_imgs"].shape) == (16, 4, 1, 256, 256)
    assert tuple(batch["cca_mask"].shape) == (16, 4)
    assert set(batch["labels"].keys()) == {"plaque", "echo", "risk"}
    return {
        "tabular": list(batch["tabular"].shape),
        "imt_img": list(batch["imt_img"].shape),
        "cca_imgs": list(batch["cca_imgs"].shape),
        "cca_mask": list(batch["cca_mask"].shape),
        "cca_true_count": int(batch["cca_mask"].sum()),
    }


def validate_pooling() -> dict:
    feats = torch.randn(3, 4, 128)
    mask = torch.tensor([
        [True, True, True, True],
        [False, False, False, False],
        [True, False, False, False],
    ])

    attn = AttentionPool(128)
    attn_pooled, attn_weights = attn(feats, mask, return_weights=True)
    assert tuple(attn_pooled.shape) == (3, 128)
    assert tuple(attn_weights.shape) == (3, 4)
    assert_close_zero(attn_pooled[1], "attention pooled control")
    assert_close_zero(attn_weights[1], "attention weights control")
    assert_close_zero(attn_weights[2, 1:], "attention weights padded positions")
    if not torch.allclose(attn_weights[0].sum(), torch.tensor(1.0), atol=1e-5):
        raise AssertionError("attention weights ca duong khong sum ve 1")

    mean = MaskedMeanPool()
    mean_pooled, mean_weights = mean(feats, mask, return_weights=True)
    assert tuple(mean_pooled.shape) == (3, 128)
    assert_close_zero(mean_pooled[1], "mean pooled control")
    assert_close_zero(mean_weights[1], "mean weights control")
    assert_close_zero(mean_weights[2, 1:], "mean weights padded positions")

    return {
        "attention_weights_sample": attn_weights.tolist(),
        "mean_weights_sample": mean_weights.tolist(),
    }


@torch.no_grad()
def validate_branch_and_fusion(cfg: dict, batch: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg["vision"]["encoder"] = "custom_cnn"
    cfg["vision"]["pretrained"] = False
    cfg["fusion"]["use_cca_in_fusion"] = False

    branch_attn = VisionBranch("custom_cnn", feat_dim=128, pooling="attention")
    branch_mean = VisionBranch("custom_cnn", feat_dim=128, pooling="mean")
    branch_attn.eval()
    branch_mean.eval()

    imt_feat, cca_feat, weights = branch_attn(
        batch["imt_img"], batch["cca_imgs"], batch["cca_mask"], return_attention=True
    )
    imt_feat_m, cca_feat_m = branch_mean(batch["imt_img"], batch["cca_imgs"], batch["cca_mask"])

    assert tuple(imt_feat.shape) == (16, 128)
    assert tuple(cca_feat.shape) == (16, 128)
    assert tuple(weights.shape) == (16, 4)
    assert tuple(imt_feat_m.shape) == (16, 128)
    assert tuple(cca_feat_m.shape) == (16, 128)

    control_rows = ~batch["cca_mask"].any(dim=1)
    if control_rows.any():
        assert_close_zero(cca_feat[control_rows], "cca_feat attention control rows")
        assert_close_zero(cca_feat_m[control_rows], "cca_feat mean control rows")
        assert_close_zero(weights[control_rows], "attention weights control rows")

    model = MultimodalFusion(cfg, in_tab=9)
    model.eval()
    out_original = model(
        batch["tabular"], batch["imt_img"], batch["cca_imgs"], batch["cca_mask"]
    )
    noisy_cca = torch.randn_like(batch["cca_imgs"])
    out_changed = model(batch["tabular"], batch["imt_img"], noisy_cca, batch["cca_mask"])

    if not torch.allclose(out_original["plaque"], out_changed["plaque"], atol=1e-5):
        raise AssertionError("plaque head bi anh huong boi CCA khi use_cca_in_fusion=false")

    return {
        "imt_feat": list(imt_feat.shape),
        "cca_feat_attention": list(cca_feat.shape),
        "cca_feat_mean": list(cca_feat_m.shape),
        "fusion_outputs": {k: list(v.shape) for k, v in out_original.items()},
        "plaque_ignores_cca_when_disabled": True,
    }


def main() -> None:
    cfg = P.load_config("configs/config.yaml")
    batch = load_batch(cfg)
    result = {
        "dataset_contract": validate_dataset_contract(batch),
        "pooling": validate_pooling(),
        "branch_and_fusion": validate_branch_and_fusion(cfg, batch),
    }
    out = Path("m3_vision/results/m3_pipeline_validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"saved={out}")


if __name__ == "__main__":
    main()
