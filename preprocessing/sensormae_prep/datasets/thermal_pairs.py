"""RGB-thermal datasets that are already aligned by their authors: only file reading,
renaming and (in the manifest) the split differ."""

from ..common import imread_raw, imread_rgb, imread_thermal
from .base import Dataset


class FMB(Dataset):
    """800x600 RGB-LWIR pairs. val is 18 pairs held out from the official train split."""

    name = "fmb"
    manual = ("https://github.com/JinyuanLiu-CV/SegMiF (Google Drive link 'FMB dataset'), "
              "unzip into fmb/ so that fmb/train/Visible and fmb/test/Infrared exist")
    required = ("train/Visible", "train/Infrared", "test/Visible", "test/Infrared")

    def load(self, split, sample_id):
        folder = self.root / ("test" if split == "test" else "train")
        return {"rgb": imread_rgb(folder / "Visible" / f"{sample_id}.png"),
                "lwir": imread_thermal(folder / "Infrared" / f"{sample_id}.png")}


class KAIST(Dataset):
    """KAIST multispectral pedestrian, the subsampled release on Hugging Face.
    Sample ids are {set}_{video}_{frame}; val is 2% of every video."""

    name = "kaist-ms-pedestrian"
    urls = {"git": "https://huggingface.co/datasets/richidubey/KAIST-Multispectral-Pedestrian-Detection-Dataset"}
    manual = "git lfs clone the Hugging Face repository into kaist-ms-pedestrian/"
    required = ("set00", "set11")

    def load(self, split, sample_id):
        set_name, video, frame = sample_id.split("_")
        folder = self.root / set_name / video
        return {"rgb": imread_rgb(folder / "visible" / f"{frame}.jpg"),
                "lwir": imread_thermal(folder / "lwir" / f"{frame}.jpg")}


class LLVIP(Dataset):
    """1280x1024 RGB-LWIR pairs, official train/test, val is 1.5% of train. Two 640px slices."""

    name = "llvip"
    sliced = True
    manual = ("https://bupt-ai-cz.github.io/LLVIP/ (LLVIP.zip), unzip into llvip/ so that "
              "llvip/visible/train and llvip/infrared/test exist")
    required = ("visible/train", "visible/test", "infrared/train", "infrared/test")

    def load(self, split, sample_id):
        folder = "test" if split == "test" else "train"
        return {"rgb": imread_rgb(self.root / "visible" / folder / f"{sample_id}.jpg"),
                "lwir": imread_thermal(self.root / "infrared" / folder / f"{sample_id}.jpg")}


class SMOD(Dataset):
    """RGB-LWIR pairs from the day (01) and night (02) folders, 2% val."""

    name = "smod"
    manual = ("https://github.com/zizhaochen6/SMOD-Dataset (download links in the README), "
              "unzip into smod/ so that smod/day and smod/night exist")
    required = ("day", "night")

    def load(self, split, sample_id):
        prefix, frame = sample_id.split("_")
        folder = self.root / {"01": "day", "02": "night"}[prefix]
        return {"rgb": imread_rgb(folder / f"{frame}_rgb.jpg"),
                "lwir": imread_thermal(folder / f"{frame}_tir.jpg")}


class MFNet(Dataset):
    """RGB-LWIR stored as 4-channel PNGs (R, G, B, T). Splits follow train.txt / val.txt /
    test.txt of the release, so train contains the horizontally flipped copies (`*_flip`)."""

    name = "mfnet"
    manual = ("https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/ "
              "(ir_seg_dataset), unzip into mfnet/ so that mfnet/images exists")
    required = ("images", "train.txt", "val.txt", "test.txt")

    def load(self, split, sample_id):
        img = imread_raw(self.root / "images" / f"{sample_id}.png")  # B, G, R, T
        return {"rgb": img[..., 2::-1].copy(), "lwir": img[..., 3]}
