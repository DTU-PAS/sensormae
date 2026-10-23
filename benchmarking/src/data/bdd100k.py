import os

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from .img_augmentation import GeometricAugmenterSegmentation, RGBAugmenter


class BDD100KDataset(Dataset):
    """BDD100K 10k semantic segmentation (19 train ids). Testing uses the val split."""

    def __init__(self, root, split="train", geom_aug=None, color_aug=None):
        self.root = root
        self.split = split
        self.geom_aug = geom_aug
        self.color_aug = color_aug

        if "bdd100k" not in root.lower():
            root = os.path.join(root, "bdd100k")

        self.images_dir = os.path.join(root, "images", "10k", split)
        self.targets_dir = os.path.join(root, "labels", "sem_seg", split)

        self.images = []
        self.targets = []

        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images path not found: {self.images_dir}")
        if not os.path.exists(self.targets_dir):
            raise RuntimeError(f"Targets path not found: {self.targets_dir}")

        for file_name in sorted(os.listdir(self.images_dir)):
            if file_name.endswith(".jpg"):
                self.images.append(os.path.join(self.images_dir, file_name))

                target_name = file_name.replace(".jpg", "_train_id.png")
                target_path = os.path.join(self.targets_dir, target_name)

                if not os.path.exists(target_path):
                    fallback_name = file_name.replace(".jpg", ".png")
                    if os.path.exists(os.path.join(self.targets_dir, fallback_name)):
                        target_name = fallback_name

                self.targets.append(os.path.join(self.targets_dir, target_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_pil = Image.open(self.images[index])
        target_pil = Image.open(self.targets[index])

        image_pil = ImageOps.exif_transpose(image_pil)
        target_pil = ImageOps.exif_transpose(target_pil)

        if image_pil.size != target_pil.size:
            if image_pil.size == (target_pil.size[1], target_pil.size[0]):
                image_pil = image_pil.transpose(Image.ROTATE_90)
                if image_pil.size != target_pil.size:
                    image_pil = image_pil.transpose(Image.ROTATE_180)

            if image_pil.size != target_pil.size:
                image_pil = image_pil.resize(target_pil.size, Image.BILINEAR)

        image = np.array(image_pil.convert("RGB"))
        target = np.array(target_pil)

        images_dict = {"rgb": image}

        if self.geom_aug:
            images_dict, target = self.geom_aug(images_dict, target)

        if self.color_aug:
            images_dict["rgb"] = self.color_aug(image=images_dict["rgb"])

        target = torch.as_tensor(target, dtype=torch.long)

        if target.ndim == 3:
            target = target.squeeze(0)

        return images_dict, target


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)

    rgb = torch.stack([img["rgb"] for img in images])
    images = {"rgb": rgb}
    targets = torch.stack(targets)

    return images, targets


def get_bdd100k_loaders(root, config):
    crop_h, crop_w = config["image_size"]

    jitter_val = config.get("color_jitter", 0.3)
    color_jitter_dict = {
        "brightness": jitter_val,
        "contrast": jitter_val,
        "saturation": jitter_val,
        "p": 1.0 if jitter_val > 0 else 0.0,
    }

    train_geom_aug = GeometricAugmenterSegmentation(
        split="train",
        modalities=["rgb"],
        height=crop_h,
        width=crop_w,
        lsj_scale=config.get("lsj_scale", (0.5, 2.0)),
        ignore_index=255,
    )

    train_color_aug = RGBAugmenter(split="train", color_jitter_values=color_jitter_dict)

    val_geom_aug = GeometricAugmenterSegmentation(
        split="val", modalities=["rgb"], height=720, width=1280, ignore_index=255
    )

    val_color_aug = RGBAugmenter(split="val")

    train_set = BDD100KDataset(
        root, split="train", geom_aug=train_geom_aug, color_aug=train_color_aug
    )
    val_set = BDD100KDataset(
        root, split="val", geom_aug=val_geom_aug, color_aug=val_color_aug
    )

    return (
        DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        ),
        DataLoader(
            val_set,
            batch_size=max(1, config["batch_size"] // 2),
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        ),
    )
