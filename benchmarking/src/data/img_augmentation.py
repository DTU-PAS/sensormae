import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2

CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)
THERM_MEAN = 0.5
THERM_STD = 0.28
DEPTH_MEAN = 0.5
DEPTH_STD = 0.28
RGB_MEAN = (0.485, 0.456, 0.406)
RGB_STD = (0.229, 0.224, 0.225)


class EventAugmenter:
    """log(1 + counts) of the event frames."""

    def __init__(
        self, split: str, mul_noise=(0.9, 1.1), p=0.0, seed=None
    ):
        self.seed = seed
        transform = []
        if split == "train":
            transform.append(
                A.MultiplicativeNoise(
                    multiplier=mul_noise, per_channel=True, elementwise=True, p=p
                )
            )

        transform.append(A.Lambda(name="log_transform", image=self._log_transform))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform, seed=self.seed)

    def _log_transform(self, image: np.ndarray, **kwargs) -> np.ndarray:
        img_float = image.astype(np.float32)
        return np.log1p(img_float)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)["image"]


class GeometricAugmenterDetection:
    """Resize, and for training flip / affine / large scale jitter, applied identically to
    all modalities and boxes. Works on numpy arrays."""

    def __init__(
        self,
        split: str,
        modalities,
        height: int = 224,
        width: int = 224,
        lsj_scale: tuple = (0.5, 2.0),
        pre_transform: list = None,
        seed: int = None,
    ):
        self.seed = seed
        self.modalities = modalities
        additional_targets = {f"image_{mod}": "mask" if mod in ["radar", "lidar"] else "image" for mod in modalities[1:]}
        H, W = self._resize_div16(height, width)
        transforms = []
        if pre_transform is not None:
            transforms.extend(pre_transform)
        transforms.append(
            A.Resize(height=H, width=W),
        )
        if split == "train":
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        rotate=(-5, 5),
                        translate_percent={"x": (-0.14, 0.14), "y": (-0.14, 0.14)},
                        mode=cv2.BORDER_CONSTANT,
                        cval=0,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.Resize(height=int(H * scale), width=int(W * scale))
                            for scale in np.linspace(lsj_scale[0], lsj_scale[1], num=20)
                        ],
                        p=0.5,
                    ),
                    A.PadIfNeeded(
                        min_height=H,
                        min_width=W,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0,
                    ),
                    A.CenterCrop(height=H, width=W),
                ]
            )

        self.transform = A.Compose(
            transforms,
            bbox_params=A.BboxParams(
                format="pascal_voc",
                label_fields=["labels"],
                min_area=1.0,
                min_visibility=0.1,
            ),
            additional_targets=additional_targets,
            seed=self.seed,
        )

    @staticmethod
    def _resize_div16(h, w):
        return max(16, (h // 16) * 16), max(16, (w // 16) * 16)

    @staticmethod
    def _to_hwc(img):
        return img[:, :, np.newaxis] if img.ndim == 2 else img

    @staticmethod
    def _restore_ndim(img, orig_ndim):
        if orig_ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
            return img[:, :, 0]
        return img

    def __call__(self, images_dict, boxes, labels):
        """boxes: [[x1, y1, x2, y2], ...]; returns the transformed images, boxes, labels."""
        mods = self.modalities
        orig_ndims = {mod: images_dict[mod].ndim for mod in mods}

        data = {
            "image": self._to_hwc(images_dict[mods[0]]),
            "bboxes": boxes if len(boxes) > 0 else [],
            "labels": labels if len(labels) > 0 else [],
        }
        for mod in mods[1:]:
            data[f"image_{mod}"] = self._to_hwc(images_dict[mod])

        result = self.transform(**data)

        images_dict[mods[0]] = self._restore_ndim(result["image"], orig_ndims[mods[0]])
        for mod in mods[1:]:
            images_dict[mod] = self._restore_ndim(
                result[f"image_{mod}"], orig_ndims[mod]
            )

        return images_dict, [list(b) for b in result["bboxes"]], list(result["labels"])


class GeometricAugmenterSegmentation:
    """As GeometricAugmenterDetection, for images and a segmentation mask."""

    def __init__(
        self,
        split: str,
        modalities: list,
        height: int = 1024,
        width: int = 1024,
        lsj_scale: tuple = (0.5, 2.0),
        ignore_index: int = 255,
        pre_transform: list = None,
        seed: int = None,
    ):
        self.seed = seed
        self.modalities = modalities
        self.ignore_index = ignore_index

        additional_targets = {f"image_{mod}": "image" for mod in modalities[1:]}
        H, W = self._resize_div16(height, width)
        transforms = []

        if pre_transform is not None:
            transforms.extend(pre_transform)

        transforms.append(
            A.Resize(height=H, width=W),
        )

        if split == "train":
            transforms.extend(
                [
                    A.HorizontalFlip(p=0.5),
                    A.Affine(
                        rotate=(-5, 5),
                        translate_percent={"x": (-0.14, 0.14), "y": (-0.14, 0.14)},
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=self.ignore_index,
                        p=0.5,
                    ),
                    A.OneOf(
                        [
                            A.Resize(height=int(H * scale), width=int(W * scale))
                            for scale in np.linspace(lsj_scale[0], lsj_scale[1], num=20)
                        ],
                        p=0.5,
                    ),
                    A.PadIfNeeded(
                        min_height=H,
                        min_width=W,
                        border_mode=cv2.BORDER_CONSTANT,
                        fill=0,
                        fill_mask=self.ignore_index,
                    ),
                    A.RandomCrop(height=H, width=W),
                ]
            )

        self.transform = A.Compose(
            transforms,
            additional_targets=additional_targets,
            seed=self.seed,
        )

    @staticmethod
    def _resize_div16(h, w):
        return max(16, (h // 16) * 16), max(16, (w // 16) * 16)

    @staticmethod
    def _to_hwc(img):
        return img[:, :, np.newaxis] if img.ndim == 2 else img

    @staticmethod
    def _restore_ndim(img, orig_ndim):
        if orig_ndim == 2 and img.ndim == 3 and img.shape[2] == 1:
            return img[:, :, 0]
        return img

    def __call__(self, images_dict, mask):
        mods = self.modalities
        orig_ndims = {mod: images_dict[mod].ndim for mod in mods}

        data = {
            "image": self._to_hwc(images_dict[mods[0]]),
            "mask": mask,
        }
        for mod in mods[1:]:
            data[f"image_{mod}"] = self._to_hwc(images_dict[mod])

        result = self.transform(**data)

        images_dict[mods[0]] = self._restore_ndim(result["image"], orig_ndims[mods[0]])
        for mod in mods[1:]:
            images_dict[mod] = self._restore_ndim(
                result[f"image_{mod}"], orig_ndims[mod]
            )

        return images_dict, result["mask"]


class RGBAugmenter:
    def __init__(self, split: str, color_jitter_values: dict = None, seed=None):
        self.seed = seed

        if color_jitter_values is None:
            color_jitter_values = {
                "brightness": 0,
                "contrast": 0,
                "saturation": 0,
                "p": 0,
            }

        transform = []
        if split == "train":
            transform.append(
                A.ColorJitter(
                    brightness=color_jitter_values["brightness"],
                    contrast=color_jitter_values["contrast"],
                    saturation=color_jitter_values["saturation"],
                    p=color_jitter_values["p"],
                )
            )
        transform.append(A.Normalize(mean=RGB_MEAN, std=RGB_STD, max_pixel_value=255.0))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform, seed=self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)["image"]


class ThermalAugmenter:
    """Min-max + CLAHE (8 bit), color jitter for training, then normalization."""

    def __init__(self, split: str, seed: int = None):
        self.seed = seed
        transform = [A.Lambda(image=self.min_max_clahe, p=1.0)]
        if split == "train":
            transform.append(A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0, p=1.0))
        transform.append(A.Normalize(mean=THERM_MEAN, std=THERM_STD, max_pixel_value=1.0))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform, seed=self.seed)

    @staticmethod
    def min_max_clahe(img, **kwargs):
        img_8bit = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        return clahe.apply(img_8bit).astype(np.float32) / 255.0

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)["image"]


class DepthAugmenter:
    def __init__(
        self, split: str, gauss_noise=(0.0, 0.0), gauss_blur=(0, 0), p=0.0, seed=None
    ):
        self.seed = seed
        transform = []
        if split == "train":
            transform.append(
                A.GaussNoise(var_limit=(gauss_noise[0] ** 2, gauss_noise[1] ** 2), p=p)
            )
            transform.append(A.GaussianBlur(blur_limit=gauss_blur, p=p))

        transform.append(A.Normalize(mean=DEPTH_MEAN, std=DEPTH_STD, max_pixel_value=1))
        transform.append(ToTensorV2())
        self.transform = A.Compose(transform, seed=self.seed)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        return self.transform(image=image)["image"]
