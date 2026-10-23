import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.data.img_augmentation import GeometricAugmenterDetection, RGBAugmenter, ThermalAugmenter
from src.data.utils import _worker_init_fn, detection_x_collate_fn


class LLVIPDataset(Dataset):
    """LLVIP RGB-LWIR pedestrians (COCO json). val is the last 5% of the sorted train set."""

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

        if "llvip" not in root.lower():
            root = os.path.join(root, "llvip")

        folder_split = "test" if split == "test" else "train"

        self.rgb_dir = os.path.join(root, folder_split, "rgb")
        self.lmwir_dir = os.path.join(root, folder_split, "lwir")
        json_path = os.path.join(root, folder_split, "data_coco.json")

        if not os.path.exists(json_path):
            raise RuntimeError(f"JSON annotation not found: {json_path}")

        with open(json_path, "r") as f:
            coco_data = json.load(f)

        self.images = sorted(coco_data["images"], key=lambda x: x["file_name"])

        self.img_to_anns = {img["id"]: [] for img in self.images}
        for ann in coco_data["annotations"]:
            self.img_to_anns[ann["image_id"]].append(ann)

        if folder_split == "train":
            split_idx = int(0.95 * len(self.images))
            if split == "train":
                self.images = self.images[:split_idx]
            elif split == "val":
                self.images = self.images[split_idx:]

        self.class_map = {cat["id"]: cat["name"] for cat in coco_data["categories"]}

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_info = self.images[idx]
        file_name = img_info["file_name"]

        mod_paths = {
            "rgb": os.path.join(self.rgb_dir, file_name),
            "lmwir": os.path.join(self.lmwir_dir, file_name),
        }

        images_dict = {}
        for mod in self.modalities:
            try:
                mod_path = mod_paths[mod]
                match mod:
                    case "rgb":
                        image = np.array(Image.open(mod_path).convert("RGB"))
                    case "lmwir":
                        image = np.array(Image.open(mod_path).convert("L")).astype(np.float32)
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                return self.__getitem__((idx + 1) % len(self))

            images_dict[mod] = image

        anns = self.img_to_anns[img_info["id"]]
        boxes, labels = [], []

        for ann in anns:
            x, y, w, h = ann["bbox"]
            x1, y1 = x, y
            x2, y2 = x + w, y + h

            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                labels.append(ann["category_id"])

        if self.geometric_transform:
            images_dict, boxes, labels = self.geometric_transform(
                images_dict, boxes, labels
            )

        for mod in self.modalities:
            if mod in self.mod_transforms:
                images_dict[mod] = self.mod_transforms[mod](images_dict[mod])

        for mod in self.modalities:
            img = images_dict[mod]
            if isinstance(img, np.ndarray):
                if img.ndim == 2:
                    img = img[np.newaxis, :, :]
                elif img.ndim == 3:
                    img = img.transpose(2, 0, 1)
                images_dict[mod] = torch.from_numpy(np.ascontiguousarray(img)).float()

        if boxes:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return images_dict, target


def get_llvip_loaders(root, config):
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

            case "lmwir":
                mod_transforms_train["lmwir"] = ThermalAugmenter("train")
                mod_transforms_val["lmwir"] = ThermalAugmenter("val")

    train_set = LLVIPDataset(root, "train", modalities, geometric_train, mod_transforms_train)
    val_set = LLVIPDataset(root, "val", modalities, geometric_val, mod_transforms_val)
    test_set = LLVIPDataset(root, "test", modalities, geometric_val, mod_transforms_val)

    return (
        DataLoader(
            train_set,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
        DataLoader(
            val_set,
            batch_size=max(1, config["batch_size"] // 2),
            shuffle=False,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
        DataLoader(
            test_set,
            batch_size=max(1, config["batch_size"] // 2),
            shuffle=False,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
    )
