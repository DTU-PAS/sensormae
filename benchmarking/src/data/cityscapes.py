import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .img_augmentation import GeometricAugmenterSegmentation, RGBAugmenter


class CityscapesDataset(Dataset):
    """Cityscapes fine annotations (19 train ids). Testing uses the val split."""

    def __init__(self, root, split="train", geom_aug=None, color_aug=None):
        self.root = root
        self.split = split
        self.geom_aug = geom_aug
        self.color_aug = color_aug

        if "cityscapes" not in root.lower():
            root = os.path.join(root, "cityscapes")
        self.images_dir = os.path.join(root, "leftImg8bit", split)
        self.targets_dir = os.path.join(root, "gtFine", split)

        self.images = []
        self.targets = []

        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Path not found: {self.images_dir}")

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                if file_name.endswith(".png"):
                    self.images.append(os.path.join(img_dir, file_name))
                    target_name = file_name.replace(
                        "leftImg8bit", "gtFine_labelTrainIds"
                    )
                    self.targets.append(os.path.join(target_dir, target_name))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.images[idx]).convert("RGB"))
        target = np.array(Image.open(self.targets[idx]))

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


def get_cityscapes_loaders(root, config):
    crop_h, crop_w = config["image_size"]

    jitter_val = config.get("color_jitter", 0.0)
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
        split="val", modalities=["rgb"], height=1024, width=2048, ignore_index=255
    )

    val_color_aug = RGBAugmenter(split="val")

    train_set = CityscapesDataset(
        root, split="train", geom_aug=train_geom_aug, color_aug=train_color_aug
    )
    val_set = CityscapesDataset(
        root, split="val", geom_aug=val_geom_aug, color_aug=val_color_aug
    )

    return (
        DataLoader(
            train_set,
            collate_fn=collate_fn,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            drop_last=True,
        ),
        DataLoader(
            val_set,
            batch_size=config["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
        ),
    )
