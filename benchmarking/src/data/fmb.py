import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .img_augmentation import (
    GeometricAugmenterSegmentation,
    RGBAugmenter,
    ThermalAugmenter,
)


class FMBDataset(Dataset):
    """FMB RGB-LWIR semantic segmentation (val split from utils/prepare_fmb_splits.py)."""

    def __init__(
        self,
        root,
        split="train",
        geom_aug=None,
        rgb_aug=None,
        lwir_aug=None,
    ):
        self.root = root
        self.split = split

        if "fmb" not in root.lower():
            root = os.path.join(root, "fmb")

        self.geom_aug = geom_aug
        self.rgb_aug = rgb_aug
        self.lwir_aug = lwir_aug

        base_dir = os.path.join(root, split)
        self.rgb_dir = os.path.join(base_dir, "Visible")
        self.ir_dir = os.path.join(base_dir, "Infrared")
        self.target_dir = os.path.join(base_dir, "Label")

        if not os.path.exists(self.rgb_dir):
            raise RuntimeError(f"Path not found: {self.rgb_dir}")

        self.files = sorted(
            [
                f
                for f in os.listdir(self.rgb_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]

        img_rgb = np.array(
            Image.open(os.path.join(self.rgb_dir, file_name)).convert("RGB")
        )
        img_ir = np.array(Image.open(os.path.join(self.ir_dir, file_name)).convert("L"))
        target = Image.open(os.path.join(self.target_dir, file_name)).convert("L")
        target = np.array(target)

        images_dict = {"rgb": img_rgb, "lwir": img_ir}

        if self.geom_aug:
            images_dict, target = self.geom_aug(images_dict, target)

        if self.rgb_aug:
            images_dict["rgb"] = self.rgb_aug(image=images_dict["rgb"])

        if self.lwir_aug:
            images_dict["lwir"] = self.lwir_aug(image=images_dict["lwir"])

        target = torch.as_tensor(target, dtype=torch.long)

        if target.ndim == 3:
            target = target.squeeze(0)

        return images_dict, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)

    rgb = torch.stack([img["rgb"] for img in images])
    lwir = torch.stack([img["lwir"] for img in images])

    images_dict = {"rgb": rgb, "lwir": lwir}
    targets = torch.stack(targets)

    return images_dict, targets


def get_fmb_loaders(root, config):
    crop_h, crop_w = config["image_size"]

    jitter_val = config.get("color_jitter", 0.4)
    color_jitter_dict = {
        "brightness": jitter_val,
        "contrast": jitter_val,
        "saturation": jitter_val,
        "p": 1.0 if jitter_val > 0 else 0.0,
    }

    train_geom_aug = GeometricAugmenterSegmentation(
        split="train",
        modalities=["rgb", "lwir"],
        height=crop_h,
        width=crop_w,
        lsj_scale=config.get("lsj_scale", (0.5, 2.0)),
        ignore_index=255,
    )

    train_rgb_aug = RGBAugmenter(split="train", color_jitter_values=color_jitter_dict)

    train_lwir_aug = ThermalAugmenter(split="train")

    val_geom_aug = GeometricAugmenterSegmentation(
        split="val",
        modalities=["rgb", "lwir"],
        height=crop_h,
        width=crop_w,
        ignore_index=255,
    )

    val_rgb_aug = RGBAugmenter(split="val")
    val_lwir_aug = ThermalAugmenter(split="val")

    train_set = FMBDataset(root, "train", train_geom_aug, train_rgb_aug, train_lwir_aug)
    val_set = FMBDataset(root, "val", val_geom_aug, val_rgb_aug, val_lwir_aug)
    test_set = FMBDataset(root, "test", val_geom_aug, val_rgb_aug, val_lwir_aug)

    return (
        DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            drop_last=True,
        ),
        DataLoader(
            val_set,
            batch_size=max(1, config["batch_size"] // 2),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        ),
        DataLoader(
            test_set,
            batch_size=max(1, config["batch_size"] // 2),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        ),
    )
