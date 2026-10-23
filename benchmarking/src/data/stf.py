import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import albumentations as A

from src.data.img_augmentation import (
    DepthAugmenter,
    GeometricAugmenterDetection,
    RGBAugmenter,
    ThermalAugmenter,
)
from src.data.utils import _worker_init_fn, detection_x_collate_fn


class SeeingThroughFogDataset(Dataset):
    """SeeingThroughFog: RGB (cropped to the gated camera FoV), gated NIR, stereo depth and
    radar / lidar projected to the image plane; KITTI-style 2D labels."""

    def __init__(
        self,
        root,
        split="train",
        modalities=["rgb"],
        geometric_transform=None,
        mod_transforms=None,
        test_splits=None,
    ):
        if "stf" not in root.lower():
            root = os.path.join(root, "stf")
        self.root = root
        self.split = split
        self.modalities = modalities
        self.geometric_transform = geometric_transform
        self.mod_transforms = mod_transforms

        if split == "train":
            split_files = [
                f"{self.root}/splits/train_clear_night.txt",
                f"{self.root}/splits/train_clear_day.txt",
            ]
        elif split == "val":
            split_files = [
                f"{self.root}/splits/val_clear_night.txt",
                f"{self.root}/splits/val_clear_day.txt",
            ]
        elif split == "test":
            split_files = []
            for test_split in test_splits:
                split_files.append(f"{self.root}/splits/{test_split}")

        self.rgb_dir = f"{self.root}/cam_stereo_left_lut"
        self.snir_dir = f"{self.root}/gated_full_acc_rect"
        self.depth_dir = f"{self.root}/depth_maps"
        self.radar_dir = f"{self.root}/radar_samples/yzv"
        self.lidar_dir = f"{self.root}/lidar_samples/yzi/lidar_hdl64_strongest"

        self.class_to_id_mapping = {
            "Pedestrian": 1,
            "PassengerCar": 2,
            "RidableVehicle": 3,
        }
        if split == "test":
            self.class_to_id_mapping["DontCare"] = -1
        self.class_map = {v: k for k, v in self.class_to_id_mapping.items()}

        self.datapoints = self.gather_datapoints(split_files)
        self.RGB2GatedCrop = [230, 950, 290, 1570]

    def read_split_files(self, sample_files):
        samples = []
        for file in sample_files:
            with open(file, "r") as f:
                for line in f:
                    line = line.strip()
                    line_parts = line.split(",")
                    samples.append(f"{line_parts[0]}_{line_parts[1]}")
        return samples

    def gather_datapoints(self, split_files):
        sample_ids = self.read_split_files(split_files)
        H, W = np.array(Image.open(f"{self.snir_dir}/{sample_ids[0]}.png")).shape[:2]
        ann_path = f"{self.root}/gt_labels/gated_labels_TMP"
        datapoints = []
        for sample_id in sample_ids:
            object_list = self.read_kitti_labels(f"{ann_path}/{sample_id}.txt", (H, W))
            if len(object_list) > 0:
                datapoints.append({"file_name": sample_id, "objects": object_list})
        return datapoints

    def __len__(self):
        return len(self.datapoints)

    def read_kitti_labels(self, label_path, img_shape):
        object_list = []
        with open(label_path, "r") as file:
            for line in file:
                line = line.replace("\n", "")
                kitti_properties = line.split(" ")
                if kitti_properties[0] not in self.class_to_id_mapping:
                    kitti_properties[0] = "DontCare"
                    if self.split != "test":
                        continue
                object_dict = {
                    "type": kitti_properties[0],
                    "id": self.class_to_id_mapping.get(kitti_properties[0], 0),
                    "xleft": min(int(round(float(kitti_properties[4]))), img_shape[1] - 1),
                    "ytop": min(int(round(float(kitti_properties[5]))), img_shape[0] - 1),
                    "xright": min(int(round(float(kitti_properties[6]))), img_shape[1] - 1),
                    "ybottom": min(int(round(float(kitti_properties[7]))), img_shape[0] - 1),
                    "truncated": float(kitti_properties[1]),
                    "occluded": float(kitti_properties[2]),
                    "alpha": float(kitti_properties[3]),
                }
                area_bbox = (object_dict["xright"] - object_dict["xleft"]) * (object_dict["ybottom"] - object_dict["ytop"])
                if area_bbox <= 0:
                    continue
                if object_dict["xright"] <= object_dict["xleft"] or object_dict["ybottom"] <= object_dict["ytop"]:
                    continue
                object_list.append(object_dict)
        return object_list

    def __getitem__(self, idx):
        img_info = self.datapoints[idx]
        file_name = img_info["file_name"]

        mod_paths = {
            "rgb": f"{self.rgb_dir}/{file_name}.png",
            "snir": f"{self.snir_dir}/{file_name}.png",
            "depth": f"{self.depth_dir}/{file_name}.npy",
            "radar": f"{self.radar_dir}/{file_name}.png",
            "lidar": f"{self.lidar_dir}/{file_name}.png",
        }

        images_dict = {}
        for mod in self.modalities:
            try:
                mod_path = mod_paths[mod]
                match mod:
                    case "rgb":
                        image = np.array(Image.open(mod_path).convert("RGB"))
                        image = image[
                            self.RGB2GatedCrop[0] : self.RGB2GatedCrop[1],
                            self.RGB2GatedCrop[2] : self.RGB2GatedCrop[3],
                        ]
                    case "snir":
                        image = np.array(Image.open(mod_path)).astype(np.float32)
                    case "depth":
                        image = np.load(mod_path).astype(np.float32)
                    case "radar":
                        image = (np.array(Image.open(mod_path))[:, :, 1]).astype(np.float32)
                        image[image == 78] = np.max(image) + 1.0
                        image = (255 - image)
                        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                        image = image[:720, :]
                    case "lidar":
                        image = (np.array(Image.open(mod_path))[:, :, 1]).astype(np.float32)
                        image[image == 78] = np.max(image) + 1.0
                        image = (255 - image)
                        image = (image - image.min()) / (image.max() - image.min() + 1e-5)
                        image = image[20:740, :]

            except Exception as e:
                print(f"Error loading {file_name}: {e}")
                return self.__getitem__((idx + 1) % len(self))

            images_dict[mod] = image

        object_list = img_info["objects"]
        boxes, labels = [], []
        occluded, truncated, alpha = [], [], []

        for obj in object_list:
            boxes.append([obj["xleft"], obj["ytop"], obj["xright"], obj["ybottom"]])
            labels.append(obj["id"])
            occluded.append(obj["occluded"])
            truncated.append(obj["truncated"])
            alpha.append(obj["alpha"])

        # indices in place of labels, to keep the attributes of the boxes that survive
        if self.geometric_transform:
            indexed = list(range(len(labels)))
            images_dict, boxes, surviving = self.geometric_transform(
                images_dict, boxes, indexed
            )
            surviving = [int(i) for i in surviving]
            labels = [labels[i] for i in surviving]
            occluded = [occluded[i] for i in surviving]
            truncated = [truncated[i] for i in surviving]
            alpha = [alpha[i] for i in surviving]

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
            "occluded": torch.tensor(occluded, dtype=torch.int64),
            "truncated": torch.tensor(truncated, dtype=torch.int64),
            "alpha": torch.tensor(alpha, dtype=torch.float32),
        }

        return images_dict, target


def get_stf_loaders(root, config):
    target_size = config["image_size"]
    crop_size = config.get("crop_size", target_size)
    modalities = config["modalities"]

    crop_h, crop_w = crop_size if isinstance(crop_size, (list, tuple)) else (crop_size, crop_size)

    geo_train = GeometricAugmenterDetection("train", modalities, target_size[0], target_size[1], config["lsj_scale"], pre_transform=[A.CenterCrop(height=crop_h, width=crop_w)])
    geo_val = GeometricAugmenterDetection("val", modalities, target_size[0], target_size[1], config["lsj_scale"], pre_transform=[A.CenterCrop(height=crop_h, width=crop_w)])

    mod_transforms_train = {}
    mod_transforms_val = {}
    for mod in modalities:
        match mod:
            case "rgb":
                cj = config.get("color_jitter", 0.4)
                mod_transforms_train["rgb"] = RGBAugmenter("train", color_jitter_values={"brightness": cj, "contrast": cj, "saturation": cj, "p": 1.0})
                mod_transforms_val["rgb"] = RGBAugmenter("val", color_jitter_values=None)

            case "snir":
                mod_transforms_train["snir"] = ThermalAugmenter("train")
                mod_transforms_val["snir"] = ThermalAugmenter("val")

            case "depth":
                mod_transforms_train["depth"] = DepthAugmenter("train")
                mod_transforms_val["depth"] = DepthAugmenter("val")

            case "radar":
                mod_transforms_train["radar"] = DepthAugmenter("train")
                mod_transforms_val["radar"] = DepthAugmenter("val")

            case "lidar":
                mod_transforms_train["lidar"] = DepthAugmenter("train")
                mod_transforms_val["lidar"] = DepthAugmenter("val")

    train_set = SeeingThroughFogDataset(
        root, "train", modalities, geo_train, mod_transforms_train,
    )
    val_set = SeeingThroughFogDataset(
        root, "val", modalities, geo_val, mod_transforms_val,
    )
    test_set = SeeingThroughFogDataset(
        root, "test", modalities, geo_val, mod_transforms_val,
        test_splits=config["test_splits"],
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
        DataLoader(
            val_set,
            batch_size=max(1, config["batch_size"]),
            shuffle=False,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
        DataLoader(
            test_set,
            batch_size=max(1, config["batch_size"]),
            shuffle=False,
            num_workers=4,
            collate_fn=detection_x_collate_fn,
            worker_init_fn=_worker_init_fn,
        ),
    )

