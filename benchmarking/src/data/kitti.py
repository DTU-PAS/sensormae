import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.img_augmentation import DepthAugmenter, GeometricAugmenterDetection, RGBAugmenter
from src.data.utils import _worker_init_fn, detection_x_collate_fn


class KITTIDataset(Dataset):
    """KITTI 2D detection with RGB and projected LiDAR depth."""

    def __init__(
        self,
        root,
        split="train",
        modalities=["rgb"],
        geometric_transform=None,
        mod_transforms=None,
    ):
        self.root = root
        self.split = split
        self.modalities = modalities
        self.geometric_transform = geometric_transform
        self.mod_transforms = mod_transforms

        if "kitti" not in root.lower():
            root = os.path.join(root, "kitti")

        self.base_dir = os.path.join(root, split)
        self.rgb_dir = os.path.join(self.base_dir, "rgb")
        self.depth_dir = os.path.join(self.base_dir, "depth")
        self.labels_dir = os.path.join(self.base_dir, "labels")

        if not os.path.exists(self.rgb_dir):
            raise RuntimeError(f"Path not found: {self.rgb_dir}")

        self.files = sorted(
            [
                f
                for f in os.listdir(self.rgb_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        self.class_name_to_id = {
            "Car": 1,
            "Pedestrian": 2,
            "Cyclist": 3,
        }
        if self.split == "test":
            self.class_name_to_id["DontCare"] = -1

        self.class_map = {v: k for k, v in self.class_name_to_id.items()}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_id = os.path.splitext(file_name)[0]

        mod_paths = {
            "rgb": f"{self.rgb_dir}/{file_name}",
            "depth": f"{self.depth_dir}/{file_name}",
        }

        images_dict = {}
        for mod in self.modalities:
            try:
                mod_path = mod_paths[mod]
                match mod:
                    case "rgb":
                        image = np.array(Image.open(mod_path).convert("RGB"))
                    case "depth":
                        image = np.array(Image.open(mod_path).convert("L"))
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                return self.__getitem__((idx + 1) % len(self))

            images_dict[mod] = image

        label_path = os.path.join(self.labels_dir, f"{file_id}.txt")
        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 15:
                        continue

                    cls_name = parts[0]
                    if cls_name == "DontCare" or cls_name not in self.class_name_to_id:
                        if self.split != "test":
                            continue
                        else:
                            cls_name = "DontCare"

                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])

                    if x2 > x1 and y2 > y1:
                        boxes.append([x1, y1, x2, y2])
                        labels.append(self.class_name_to_id[cls_name])

        if self.geometric_transform:
            images_dict, boxes, labels = self.geometric_transform(
                images_dict, boxes, labels
            )

        for mod in self.modalities:
            if mod in self.mod_transforms:
                images_dict[mod] = self.mod_transforms[mod](images_dict[mod])

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return images_dict, target


def get_kitti_loaders(root, config):
    target_size = config["image_size"]
    modalities = config["modalities"]

    geometric_train = GeometricAugmenterDetection("train", modalities, target_size[0], target_size[1], config["lsj_scale"], pre_transform=None)
    geometric_val = GeometricAugmenterDetection("val", modalities, target_size[0], target_size[1], config["lsj_scale"], pre_transform=None)

    mod_transforms_train = {}
    mod_transforms_val = {}
    for mod in modalities:
        match mod:
            case "rgb":
                cj = config.get("color_jitter", 0.4)
                mod_transforms_train["rgb"] = RGBAugmenter("train", color_jitter_values={"brightness": cj, "contrast": cj, "saturation": cj, "p": 1.0})
                mod_transforms_val["rgb"] = RGBAugmenter("val", color_jitter_values=None)

            case "depth":
                mod_transforms_train["depth"] = DepthAugmenter("train")
                mod_transforms_val["depth"] = DepthAugmenter("val")

    train_set = KITTIDataset(root, "train", modalities, geometric_train, mod_transforms_train)
    val_set = KITTIDataset(root, "val", modalities, geometric_val, mod_transforms_val)

    # no labelled test split: val is used for testing
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, config["batch_size"]),
        shuffle=False,
        num_workers=4,
        collate_fn=detection_x_collate_fn,
        worker_init_fn=_worker_init_fn,
    )

    return (
        DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
        val_loader,
        val_loader,
    )
