"""Adapted from https://github.com/uzh-rpg/dsec-det/blob/master/src/dsec_det/dataset.py"""

import os
import random
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  (registers the HDF5 compression filters)
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

from src.data.img_augmentation import EventAugmenter

DS_SPLIT = {
    "train": [
        "zurich_city_00_a",
        "zurich_city_01_a",
        "zurich_city_02_a",
        "zurich_city_05_a",
        "zurich_city_06_a",
        "zurich_city_07_a",
    ],
    "val": ["zurich_city_04_a", "zurich_city_08_a"],
    "test": ["zurich_city_13_a", "zurich_city_14_c", "zurich_city_15_a"],
}

# DSEC-Semantic 11 classes (train id -> name)
CLASS_NAMES = {
    0: "sky", 1: "building", 2: "fence", 3: "person/rider", 4: "pole", 5: "road",
    6: "sidewalk", 7: "vegetation/terrain", 8: "vehicle", 9: "wall", 10: "traffic light/sign",
    255: "unlabeled",
}

class DSECSeg(torch.utils.data.Dataset):
    """Preprocessed DSEC-Semantic sequences (RGB, depth, events, 11-class labels)."""

    def __init__(
        self,
        root: str,
        split: str = "train",
        geometric_transform: v2.Transform | None = None,
        depth_transform: v2.Transform | None = None,
        rgb_transform: v2.Transform | None = None,
        event_transform: v2.Transform | None = None,
        lsj_scale: tuple[float, float] = (0.5, 2.0),
    ):
        if "dsec" not in root.lower():
            root = os.path.join(root, "dsec")
        root = Path(root)
        assert root.exists()
        assert split in ["train", "test", "val"]

        self.classes = list(CLASS_NAMES.values())
        self.class_map = dict(CLASS_NAMES)

        root_split = "train" if split in ["train", "val"] else "test"
        self.root = root / root_split
        self.training = split == "train"

        self.height = 480
        self.width = 640

        self.size = max(self.height, self.width)
        self.min_size = int(self.size * lsj_scale[0])
        self.max_size = int(self.size * lsj_scale[1])
        self.random_crop = v2.RandomCrop(
            size=(self.size, self.size), pad_if_needed=True, fill=0
        )
        self.random_crop_target = v2.RandomCrop(
            size=(self.size, self.size), pad_if_needed=True, fill=255
        )

        self.geometric_transform = geometric_transform
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.event_transform = event_transform

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
        with h5py.File(data_path, "r") as f:
            img_rgb = f["rgb"][index]
            img_depth = f["depth"][index]
            img_event = f["event"][index]
            target = f["segmentation"][index]

        img_rgb = tv_tensors.Image(img_rgb)
        img_depth = tv_tensors.Image(img_depth)
        img_event = tv_tensors.Image(img_event)
        target = tv_tensors.Mask(target, dtype=torch.long)

        # labels are 440 rows high, the images 480: pad with the ignore index
        target = F.pad(target, [0, 0, 0, 40], fill=255)

        if self.geometric_transform is not None:
            (img_rgb, img_depth, img_event), target = self.geometric_transform(
                (img_rgb, img_depth, img_event), target
            )

            if self.training:
                # one random draw shared by all modalities
                size = int(torch.randint(self.min_size, self.max_size, ()))
                img_rgb = F.resize(img_rgb, size=size)
                img_depth = F.resize(img_depth, size=size)
                img_event = F.resize(img_event, size=size)
                target = F.resize(
                    target, size=size, interpolation=F.InterpolationMode.NEAREST
                )

                crop_params = self.random_crop.make_params([img_rgb])
                img_rgb = self.random_crop.transform(img_rgb, crop_params)
                img_depth = self.random_crop.transform(img_depth, crop_params)
                img_event = self.random_crop.transform(img_event, crop_params)
                target = self.random_crop_target.transform(target, crop_params)

                if random.random() < 0.5:
                    img_rgb = F.hflip(img_rgb)
                    img_depth = F.hflip(img_depth)
                    img_event = F.hflip(img_event)
                    target = F.hflip(target)

        if self.rgb_transform is not None:
            img_rgb = self.rgb_transform(img_rgb)

        if self.depth_transform is not None:
            img_depth = self.depth_transform(img_depth)

        if self.event_transform is not None:
            img_event = img_event.permute(1, 2, 0).numpy()
            img_event = self.event_transform(img_event)
            img_event = tv_tensors.Image(img_event)

        if not self.training:
            img_rgb = F.resize(img_rgb, size=(self.size, self.size))
            img_depth = F.resize(img_depth, size=(self.size, self.size))
            img_event = F.resize(img_event, size=(self.size, self.size))
            target = F.resize(
                target,
                size=(self.size, self.size),
                interpolation=F.InterpolationMode.NEAREST,
            )

        output = {"rgb": img_rgb, "depth": img_depth, "event": img_event}

        return output, target

    def index_to_sequence(self, index):
        for sequence, start, end in self.sequence_intervals:
            if start <= index < end:
                return sequence, index - start
        raise IndexError(f"Index {index} out of range for dataset of size {self.size}")


def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    images, targets = zip(*batch)

    rgb = torch.stack([img["rgb"] for img in images])
    depth = torch.stack([img["depth"] for img in images])
    event = torch.stack([img["event"] for img in images])
    images = {"rgb": rgb, "depth": depth, "event": event}
    targets = torch.stack(targets)

    return images, targets


def get_dsec_seg_loaders(root, config):
    geometric_train = v2.Identity()
    geometric_eval = v2.Identity()

    rgb_train = v2.Compose(
        [
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    rgb_eval = v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    event_train = EventAugmenter(split="train")
    event_eval = EventAugmenter(split="test")

    lsj_scale = config.get("lsj_scale", [0.5, 2.0])

    train_dataset = DSECSeg(
        root=root,
        split="train",
        geometric_transform=geometric_train,
        rgb_transform=rgb_train,
        event_transform=event_train,
        lsj_scale=lsj_scale,
    )

    val_dataset = DSECSeg(
        root=root,
        split="val",
        geometric_transform=geometric_eval,
        rgb_transform=rgb_eval,
        event_transform=event_eval,
        lsj_scale=lsj_scale,
    )

    test_dataset = DSECSeg(
        root=root,
        split="test",
        geometric_transform=geometric_eval,
        rgb_transform=rgb_eval,
        event_transform=event_eval,
        lsj_scale=lsj_scale,
    )

    return (
        DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=4,
        ),
        DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
        ),
        DataLoader(
            test_dataset,
            batch_size=config["batch_size"],
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=4,
        ),
    )
