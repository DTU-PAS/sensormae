import subprocess
import zipfile

from ..common import imread_rgb
from .base import Dataset

CROP_HEIGHT = 775  # drops the ego-vehicle hood at the bottom of every frame
PACKAGES = ("leftImg8bit_trainvaltest.zip", "rightImg8bit_trainvaltest.zip",
            "leftImg8bit_trainextra.zip", "rightImg8bit_trainextra.zip")


class Cityscapes(Dataset):
    """Left RGB + stereo depth. train includes train_extra. Depth is predicted on the full
    2048x1024 pair (half-resolution model input), then everything is cropped to the top
    775 rows and cut into four 640px slices."""

    name = "cityscapes"
    sliced = True
    depth_model = "foundation_stereo_s_w1024.onnx"  # input 512x1024
    urls = {p: "csDownload" for p in PACKAGES}
    manual = ("https://www.cityscapes-dataset.com/downloads/ (account required): "
              + ", ".join(PACKAGES) + "; unzip into cityscapes/. `download` fetches them with "
              "csDownload (pip install cityscapesscripts) and asks for the account credentials.")
    required = ("leftImg8bit/train", "leftImg8bit/train_extra", "leftImg8bit/val",
                "leftImg8bit/test", "rightImg8bit/train", "rightImg8bit/train_extra",
                "rightImg8bit/val", "rightImg8bit/test")

    def fetch(self):
        """Download the four left/right 8-bit packages (~110 GB) with the official tool."""
        self.root.mkdir(parents=True, exist_ok=True)
        for package in PACKAGES:
            target = self.root / package
            done = self.root / f"{package}.done"
            if done.exists():
                continue
            if not target.exists():
                subprocess.run(["csDownload", "-d", str(self.root), package], check=True)
            print(f"[{self.name}] extracting {package}")
            with zipfile.ZipFile(target) as z:
                z.extractall(self.root)
            target.unlink()
            done.touch()

    def _path(self, side, split, sample_id):
        city = sample_id.split("_")[0]
        folders = ("train", "train_extra") if split == "train" else (split,)
        for folder in folders:
            p = self.root / f"{side}Img8bit" / folder / city / f"{sample_id}_{side}Img8bit.png"
            if p.exists():
                return p
        raise FileNotFoundError(f"{side} image for {sample_id}")

    def load(self, split, sample_id):
        left = imread_rgb(self._path("left", split, sample_id))
        right = imread_rgb(self._path("right", split, sample_id))
        depth = self.stereo(left, right)
        return {"rgb": left[:CROP_HEIGHT], "depth": depth[:CROP_HEIGHT]}
