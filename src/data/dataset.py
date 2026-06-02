# [M1] CarotidDataset + collate_fn — DATA CONTRACT cho ca nhom (M2/M3/M4 phu thuoc vao day).
# Module FUNCTIONAL: dung cho training that lan smoke test.
from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from . import preprocess as P

logger = logging.getLogger("carotid.data")


class CarotidDataset(Dataset):
    """
    Tra ve moi sample 1 dict (xem data-contract trong PROJECT_PLAN Phan 3.3 + src/data/README.md):
      {
        "patient_id": str,
        "tabular":   FloatTensor[9],            # 8 numeric da scale + Sex
        "imt_img":   FloatTensor[1,256,256],    # luon co
        "cca_imgs":  FloatTensor[K,1,256,256],  # K=0 (Control) hoac 4 (Target)
        "labels": {"plaque": Float[1], "echo": Long[1], "risk": Float[1]}
      }

    LUU Y CHONG LEAKAGE: imt_img dung cho task plaque; cca_imgs CHI cho task echo.

    Tham so:
      scaler              : StandardScaler ĐÃ FIT TREN TRAIN FOLD (Dataset chi APPLY, khong fit).
      transform/cca_transform : augmentation (M3 cung cap) nhan PIL Image -> tra ve tensor [1,H,W].
      image_error_policy  : "raise" (mac dinh, an toan du lieu) | "zero" (log canh bao + tra tensor 0
                            de 1 file hong khong giet ca qua trinh train dai tren Colab).
      cache_images        : True -> cache mang uint8 da decode (dataset 680 anh ~ vai MB, tang toc multi-epoch).
    """

    def __init__(self, df, cfg: dict, scaler, project_root: str = ".",
                 transform=None, cca_transform=None,
                 image_error_policy: str = "raise", cache_images: bool = False):
        assert image_error_policy in ("raise", "zero"), \
            f"image_error_policy phai la 'raise' hoac 'zero', nhan '{image_error_policy}'"
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.scaler = scaler
        self.project_root = project_root
        self.transform = transform            # augmentation cho IMT (M3 cung cap)
        self.cca_transform = cca_transform    # augmentation cho CCA (M3 cung cap)
        self.image_error_policy = image_error_policy
        self.cache_images = cache_images
        self._img_cache: dict[str, np.ndarray] = {}

        self.images_dir = os.path.join(project_root, cfg["data"]["images_dir"])
        self.feat_cols = P.feature_columns(cfg)
        self.img_size = int(cfg["data"]["image_size"])
        self.img_mode = cfg["data"]["image_mode"]

        # CHONG LEAKAGE: scaler da fit san tren train fold (do train.py truyen vao).
        # Dataset CHI apply (khong bao gio fit) -> khong ro ri thong tin val sang train.
        self.df_scaled = P.apply_scaler(P.encode_categorical(self.df, cfg), scaler, cfg)

    def __len__(self) -> int:
        return len(self.df)

    # ------------------------------------------------------------------ images
    def _read_image_array(self, name: str) -> np.ndarray:
        """Doc 1 anh -> numpy uint8 [H,W] grayscale (co guard kich thuoc + cache + xu ly loi)."""
        if self.cache_images and name in self._img_cache:
            return self._img_cache[name]

        path = os.path.join(self.images_dir, name)
        try:
            with Image.open(path) as im:
                img = im.convert(self.img_mode)  # 'L' -> 1 kenh
                if img.size != (self.img_size, self.img_size):
                    logger.warning(
                        "Anh '%s' co kich thuoc %s != ky vong (%d,%d) -> tu dong resize.",
                        name, img.size, self.img_size, self.img_size,
                    )
                    img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
                arr = np.asarray(img, dtype=np.uint8)
        except (FileNotFoundError, UnidentifiedImageError, OSError, ValueError) as err:
            arr = self._on_image_error(name, path, err)

        if self.cache_images:
            self._img_cache[name] = arr
        return arr

    def _on_image_error(self, name: str, path: str, err: Exception) -> np.ndarray:
        """Bao loi anh ro rang. 'raise' -> dung han; 'zero' -> log + tra anh 0."""
        msg = f"[CarotidDataset] Khong doc duoc anh '{name}' tai '{path}': {type(err).__name__}: {err}"
        if self.image_error_policy == "zero":
            logger.error(msg + " -> tra ve tensor 0 (che do lenient).")
            return np.zeros((self.img_size, self.img_size), dtype=np.uint8)
        raise RuntimeError(msg) from err

    def _load_image(self, name: Optional[str], transform) -> torch.Tensor:
        """name -> FloatTensor[1,H,W] trong [0,1] (hoac qua augmentation neu co)."""
        if not name:
            raise ValueError(
                "[CarotidDataset] Thieu ten anh IMT — moi ca BAT BUOC co dung 1 anh '*_IMT.png'. "
                "Kiem tra cot Associated_Images trong CSV."
            )
        arr = self._read_image_array(name)
        if transform is not None:
            # torchvision transform ky vong PIL Image -> tra ve tensor [1,H,W].
            return transform(Image.fromarray(arr, mode=self.img_mode))
        return torch.from_numpy(arr.astype(np.float32) / 255.0).unsqueeze(0)  # [1,H,W]

    # ------------------------------------------------------------------ sample
    def __getitem__(self, idx: int) -> dict:
        cfg = self.cfg
        row = self.df.iloc[idx]
        row_scaled = self.df_scaled.iloc[idx]

        # --- Tabular ---
        tabular = torch.tensor(
            [float(row_scaled[c]) for c in self.feat_cols], dtype=torch.float32
        )

        # --- Anh (IMT cho plaque, CCA cho echo) ---
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
    - cca_imgs: pad ve [B,4,1,256,256], cca_mask [B,4] (True=anh that, False=pad/Control)
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


# ---------------------------------------------------------------------- helpers
def make_weighted_sampler(df_train, cfg, replacement: bool = True):
    """
    Tao WeightedRandomSampler can bang lop `Plaque_present` cho fold train.

    Trong so moi sample = 1 / (so luong cua lop do) -> 2 lop duoc lay mau xap xi
    bang nhau moi epoch, giup cai thien PR-AUC/Sensitivity tren du lieu lech 205/95.

    LUU Y: chi tao tu NHAN CUA FOLD TRAIN (khong dung val) — tranh ro ri.
    Dung kem DataLoader: DataLoader(ds, sampler=sampler, ...) — KHONG dat shuffle=True.
    """
    from torch.utils.data import WeightedRandomSampler

    y = df_train[cfg["columns"]["target_plaque"]].astype(int).to_numpy()
    class_count = np.bincount(y, minlength=2).astype(np.float64)
    class_weight = 1.0 / np.clip(class_count, 1.0, None)   # tranh chia 0
    sample_weight = class_weight[y]
    return WeightedRandomSampler(
        weights=torch.as_tensor(sample_weight, dtype=torch.double),
        num_samples=len(sample_weight),
        replacement=replacement,
    )


def scan_image_integrity(df, cfg, project_root: str = ".") -> dict:
    """
    Quet toan bo anh duoc tham chieu trong `Associated_Images` -> phat hien som:
      - missing   : file khong ton tai
      - wrong_size: kich thuoc != image_size trong config
      - corrupt   : file hong / khong decode duoc
    Tra ve dict: {"total", "ok", "missing":[...], "wrong_size":[...], "corrupt":[...]}.
    Goi truoc khi train de tranh loi giua chung.
    """
    images_dir = os.path.join(project_root, cfg["data"]["images_dir"])
    size = int(cfg["data"]["image_size"])
    report = {"total": 0, "ok": 0, "missing": [], "wrong_size": [], "corrupt": []}

    for _, row in df.iterrows():
        for name in P.parse_associated_images(row[cfg["columns"]["images"]]):
            report["total"] += 1
            path = os.path.join(images_dir, name)
            if not os.path.exists(path):
                report["missing"].append(name)
                continue
            try:
                with Image.open(path) as im:
                    im.verify()                 # phat hien file hong ma khong decode het
                with Image.open(path) as im2:   # mo lai de doc kich thuoc (verify lam hong handle)
                    w, h = im2.size
                if (w, h) != (size, size):
                    report["wrong_size"].append((name, (w, h)))
                else:
                    report["ok"] += 1
            except Exception as e:               # noqa: BLE001 - bao cao moi loi decode
                report["corrupt"].append((name, str(e)))
    return report
