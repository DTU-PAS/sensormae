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


class MFNetDataset(Dataset):
    """MFNet RGB-thermal semantic segmentation (thermal stored in the alpha channel)."""

    def __init__(
        self,
        root,
        split="train",
        geom_aug=None,
        rgb_aug=None,
        lwir_aug=None,
    ):
        self.root = root
        if "mfnet" not in root.lower():
            root = os.path.join(root, "mfnet")
        self.split = split

        self.geom_aug = geom_aug
        self.rgb_aug = rgb_aug
        self.lwir_aug = lwir_aug

        self.images_dir = os.path.join(root, "images")
        self.labels_dir = os.path.join(root, "labels")

        self.blacklist = set()
        blacklist_path = os.path.join(root, "black_list.txt")
        if os.path.exists(blacklist_path):
            with open(blacklist_path, "r") as f:
                for line in f:
                    clean_id = line.strip()
                    if clean_id:
                        self.blacklist.add(clean_id)

        split_file = f"{split}.txt"
        split_path = os.path.join(root, split_file)

        if not os.path.exists(split_path):
            if split == "test_day":
                split_path = os.path.join(root, "test_day.txt")
            elif split == "test_night":
                split_path = os.path.join(root, "test_night.txt")
            else:
                raise RuntimeError(f"Split file not found: {split_path}")

        self.files = []
        with open(split_path, "r") as f:
            for line in f:
                file_id = line.strip()
                if file_id and file_id not in self.blacklist:
                    self.files.append(file_id)


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]

        img_path = os.path.join(self.images_dir, f"{file_id}.png")
        try:
            img_rgba = np.array(Image.open(img_path).convert("RGBA"))
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        img_rgb = img_rgba[:, :, :3]
        img_ir = img_rgba[:, :, 3]

        label_path = os.path.join(self.labels_dir, f"{file_id}.png")
        target = np.array(Image.open(label_path))

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
        target[target == 0] = 255  # unlabeled -> ignore index

        return images_dict, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)

    rgb = torch.stack([img["rgb"] for img in images])
    lwir = torch.stack([img["lwir"] for img in images])
    images_dict = {"rgb": rgb, "lwir": lwir}
    targets = torch.stack(targets)

    return images_dict, targets


def get_mfnet_loaders(root, config):
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
        lsj_scale=config.get("lsj_scale", (0.5, 1.5)),
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

    train_set = MFNetDataset(root, "train", train_geom_aug, train_rgb_aug, train_lwir_aug)
    val_set = MFNetDataset(root, "val", val_geom_aug, val_rgb_aug, val_lwir_aug)
    test_set = MFNetDataset(root, "test", val_geom_aug, val_rgb_aug, val_lwir_aug)

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