# [M1] CarotidDataset + collate_fn — DATA CONTRACT cho ca nhom (M2/M3/M4 phu thuoc vao day).
# Module FUNCTIONAL: dung cho smoke test load 1 batch.
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import preprocess as P


class CarotidDataset(Dataset):
    """
    Tra ve moi sample 1 dict (xem data-contract trong PROJECT_PLAN Phan 3.3):
      {
        "patient_id": str,
        "tabular":   FloatTensor[9],            # 8 numeric da scale + Sex
        "imt_img":   FloatTensor[1,256,256],    # luon co
        "cca_imgs":  FloatTensor[K,1,256,256],  # K=0 (Control) hoac 4 (Target)
        "labels": {"plaque": Float[1], "echo": Long[1], "risk": Float[1]}
      }

    LUU Y CHONG LEAKAGE: imt_img dung cho task plaque; cca_imgs CHI cho task echo.
    """

    def __init__(self, df, cfg: dict, scaler, project_root: str = ".",
                 transform=None, cca_transform=None):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.scaler = scaler
        self.project_root = project_root
        self.transform = transform            # augmentation cho IMT (M3 cung cap)
        self.cca_transform = cca_transform    # augmentation cho CCA (M3 cung cap)

        self.images_dir = os.path.join(project_root, cfg["data"]["images_dir"])
        self.feat_cols = P.feature_columns(cfg)
        self.img_size = cfg["data"]["image_size"]

        # Scale truoc toan bo (scaler da fit tren train fold).
        self.df_scaled = P.apply_scaler(P.encode_categorical(self.df, cfg), scaler, cfg)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, name: str, transform) -> torch.Tensor:
        """Doc 1 anh grayscale -> FloatTensor[1,H,W] trong [0,1]."""
        path = os.path.join(self.images_dir, name)
        img = Image.open(path).convert(self.cfg["data"]["image_mode"])  # 'L'
        if transform is not None:
            return transform(img)  # ky vong tra ve tensor [1,H,W]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W]

    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        row = self.df.iloc[idx]
        row_scaled = self.df_scaled.iloc[idx]

        # --- Tabular ---
        tabular = torch.tensor(
            [float(row_scaled[c]) for c in self.feat_cols], dtype=torch.float32
        )

        # --- Anh ---
        names = P.parse_associated_images(row[cfg["columns"]["images"]])
        imt_name, cca_names = P.split_imt_cca(names)
        imt_img = self._load_image(imt_name, self.transform)
        if cca_names:
            cca_imgs = torch.stack([self._load_image(n, self.cca_transform) for n in cca_names])
        else:
            cca_imgs = torch.zeros((0, 1, self.img_size, self.img_size), dtype=torch.float32)

        # --- Labels ---
        plaque = torch.tensor([float(row[cfg["columns"]["target_plaque"]])], dtype=torch.float32)
        echo = torch.tensor([P.encode_echo_label(row[cfg["columns"]["target_echo"]], cfg)],
                            dtype=torch.long)
        risk = torch.tensor([float(row[cfg["columns"]["target_risk"]])], dtype=torch.float32)

        return {
            "patient_id": str(row[cfg["columns"]["id"]]),
            "tabular": tabular,
            "imt_img": imt_img,
            "cca_imgs": cca_imgs,
            "labels": {"plaque": plaque, "echo": echo, "risk": risk},
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Gop batch: pad cca_imgs ve K=4 + tao cca_mask.
    - imt_img: stack thang (luon co dung 1 anh/ca) -> [B,1,256,256]
    - cca_imgs: pad ve [B,4,1,256,256], cca_mask [B,4] (True=anh that)
    """
    B = len(batch)
    K_max = 4  # toi da 4 anh CCA (theo cau truc dataset)
    img_size = batch[0]["imt_img"].shape[-1]

    tabular = torch.stack([b["tabular"] for b in batch])
    imt_img = torch.stack([b["imt_img"] for b in batch])

    cca_imgs = torch.zeros((B, K_max, 1, img_size, img_size), dtype=torch.float32)
    cca_mask = torch.zeros((B, K_max), dtype=torch.bool)
    for i, b in enumerate(batch):
        k = b["cca_imgs"].shape[0]
        if k > 0:
            cca_imgs[i, :k] = b["cca_imgs"]
            cca_mask[i, :k] = True

    labels = {
        "plaque": torch.stack([b["labels"]["plaque"] for b in batch]),
        "echo": torch.stack([b["labels"]["echo"] for b in batch]),
        "risk": torch.stack([b["labels"]["risk"] for b in batch]),
    }
    return {
        "patient_id": [b["patient_id"] for b in batch],
        "tabular": tabular,
        "imt_img": imt_img,
        "cca_imgs": cca_imgs,
        "cca_mask": cca_mask,
        "labels": labels,
    }
