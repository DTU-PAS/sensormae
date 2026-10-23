"""Adapted from https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/dataset.py"""

import os
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the HDF5 compression filters)
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.img_augmentation import (
    DepthAugmenter,
    EventAugmenter,
    GeometricAugmenterDetection,
    RGBAugmenter,
)
from src.data.utils import _worker_init_fn, detection_x_collate_fn

DS_SPLIT = {
    "train": ['thun_00_a', 'interlaken_00_c', 'interlaken_00_d', 'interlaken_00_e', 'interlaken_00_f', 'interlaken_00_g', 'zurich_city_00_a', 'zurich_city_00_b', 'zurich_city_01_a', 'zurich_city_01_b', 'zurich_city_01_c', 'zurich_city_01_d', 'zurich_city_01_e', 'zurich_city_01_f', 'zurich_city_02_a', 'zurich_city_02_b', 'zurich_city_02_c', 'zurich_city_02_d', 'zurich_city_02_e', 'zurich_city_03_a', 'zurich_city_04_a', 'zurich_city_04_b', 'zurich_city_04_c', 'zurich_city_04_d', 'zurich_city_04_e', 'zurich_city_04_f', 'zurich_city_05_a', 'zurich_city_05_b', 'zurich_city_06_a', 'zurich_city_07_a', 'zurich_city_08_a', 'zurich_city_09_a', 'zurich_city_09_b', 'zurich_city_09_c', 'zurich_city_09_d', 'zurich_city_09_e', 'zurich_city_10_a', 'zurich_city_10_b', 'zurich_city_11_a', 'zurich_city_11_b', 'zurich_city_11_c'],
    "val": ['zurich_city_16_a', 'zurich_city_17_a', 'zurich_city_18_a', 'zurich_city_19_a', 'zurich_city_20_a', 'zurich_city_21_a'],
    "test": ['thun_01_a', 'thun_01_b', 'thun_02_a', 'interlaken_00_a', 'interlaken_00_b', 'interlaken_01_a', 'zurich_city_12_a', 'zurich_city_13_a', 'zurich_city_13_b', 'zurich_city_14_a', 'zurich_city_14_b', 'zurich_city_14_c', 'zurich_city_15_a']
}

class DSECDet(torch.utils.data.Dataset):
    """Preprocessed DSEC detection sequences (one h5 file per sequence)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        modalities=["rgb"],
        geometric_transform=None,
        mod_transforms=None,
        min_box_side: int = 10,
        min_box_diagonal: int = 15,
    ):
        if "dsec" not in root.lower():
            root = os.path.join(root, "dsec")
        root = Path(root)
        assert root.exists()
        assert split in ["train", "test", "val"]

        self.classes = ["pedestrian", "car"]
        self.class_map = {i: c for i, c in enumerate(self.classes, start=1)}

        self.height = 480
        self.width = 640

        self.min_box_side = min_box_side
        self.min_box_diagonal = min_box_diagonal

        root_split = "train" if split in ["train", "val"] else "test"
        self.root = root / root_split
        self.training = split == "train"
        self.modalities = modalities

        self.geometric_transform = geometric_transform
        self.mod_transforms = mod_transforms

        split_config = DS_SPLIT

        self.sequence_intervals = []
        start = 0
        self.dataset_size = 0
        for sequence in split_config[split]:
            h5_path = self.root / f"{sequence}.h5"
            with h5py.File(h5_path, "r") as f:
                length = len(f["rgb"])
            end = start + length
            self.sequence_intervals.append((sequence, start, end))
            start = end
            self.dataset_size += length

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, item):
        sequence, index = self.index_to_sequence(item)
        data_path = self.root / f"{sequence}.h5"
        images_dict = {}
        with h5py.File(data_path, "r") as f:
            for mod in self.modalities:
                try:
                    image = f[mod][index]
                    if mod == "rgb" or mod == "event":
                        image = np.transpose(image, (1, 2, 0))
                except Exception as e:
                    print(f"Error loading {data_path}: {e}")
                    return self.__getitem__((index + 1) % len(self))
                images_dict[mod] = image

            boxes = f["boxes"]
            x1 = boxes["x"][index]
            y1 = boxes["y"][index]
            x2 = x1 + boxes["w"][index]
            y2 = y1 + boxes["h"][index]
            class_id = boxes["class_id"][index]

            boxes = np.stack([x1, y1, x2, y2], axis=1)
            boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, self.width - 1)
            boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, self.height - 1)
            valid_mask = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            boxes = boxes[valid_mask]
            class_id = class_id[valid_mask]
            class_id = class_id + 1  # 0 is the background

        if self.geometric_transform is not None:
            images_dict, boxes, class_id = self.geometric_transform(
                images_dict, boxes, class_id
            )

        for mod in self.modalities:
            if mod in self.mod_transforms:
                images_dict[mod] = self.mod_transforms[mod](images_dict[mod])

        labels = self.map_classes(class_id)
        boxes, labels = self.filter_boxes(boxes, labels)
        if len(boxes) > 0:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        else:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)

        target = {
            "boxes": boxes_tensor,
            "labels": torch.tensor(labels, dtype=torch.int64),
        }

        return images_dict, target

    def index_to_sequence(self, index):
        for sequence, start, end in self.sequence_intervals:
            if start <= index < end:
                return sequence, index - start
        raise IndexError(f"Index {index} out of range for dataset of size {self.size}")

    def map_classes(self, labels: torch.Tensor):
        """pedestrian -> 1; car, bus, truck -> 2 (car); rider, bicycle, motorcycle, train -> 0 (dropped)."""
        mapping = np.array([0, 1, 0, 2, 2, 2, 0, 0, 0], dtype=np.int64)
        return mapping[labels]

    def filter_boxes(self, boxes: torch.Tensor, labels: torch.Tensor):
        """Keep pedestrian / car boxes with large enough sides and diagonal."""
        if len(boxes) == 0:
            return boxes, labels
        class_mask = labels > 0
        boxes = np.array(boxes)
        width = boxes[:, 2] - boxes[:, 0]
        height = boxes[:, 3] - boxes[:, 1]
        diagonal = np.sqrt(width**2 + height**2)
        side_mask = (width > self.min_box_side) & (height > self.min_box_side)
        diagonal_mask = diagonal > self.min_box_diagonal
        mask = class_mask & side_mask & diagonal_mask
        boxes = boxes[mask].tolist()
        labels = labels[mask].tolist()
        return [list(b) for b in boxes], list(labels)


def get_dsec_det_loaders(root, config):
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

            case "event":
                mod_transforms_train["event"] = EventAugmenter(split="train")
                mod_transforms_val["event"] = EventAugmenter(split="test")

            case "depth":
                mod_transforms_train["depth"] = DepthAugmenter("train")
                mod_transforms_val["depth"] = DepthAugmenter("val")

    min_box_side = config.get("box_min_side", 10)
    min_box_diagonal = config.get("box_min_diagonal", 15)

    train_dataset = DSECDet(
        root=root,
        split="train",
        modalities=modalities,
        geometric_transform=geometric_train,
        mod_transforms=mod_transforms_train,
        min_box_side=min_box_side,
        min_box_diagonal=min_box_diagonal,
    )

    val_dataset = DSECDet(
        root=root,
        split="val",
        modalities=modalities,
        geometric_transform=geometric_val,
        mod_transforms=mod_transforms_val,
        min_box_side=min_box_side,
        min_box_diagonal=min_box_diagonal,
    )

    test_dataset = DSECDet(
        root=root,
        split="test",
        modalities=modalities,
        geometric_transform=geometric_val,
        mod_transforms=mod_transforms_val,
        min_box_side=min_box_side,
        min_box_diagonal=min_box_diagonal,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=detection_x_collate_fn,
            shuffle=True,
            num_workers=4,
            worker_init_fn=_worker_init_fn,
        ),
        DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            collate_fn=detection_x_collate_fn,
            shuffle=False,
            num_workers=4,
            worker_init_fn=_worker_init_fn,
        ),
        DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            collate_fn=detection_x_collate_fn,
            shuffle=False,
            num_workers=4,
            worker_init_fn=_worker_init_fn,
        ),
    )
