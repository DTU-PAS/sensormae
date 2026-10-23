import shutil
import subprocess
import tarfile
import urllib.request
import zipfile
from pathlib import Path


class Dataset:
    """One source dataset.

    Subclasses describe where the raw data comes from and implement `load`, which returns
    the spatially aligned modalities of one sample at full resolution:
    rgb (RGB uint8), lwir / nir / swir (raw sensor values), event (H, W, 10) uint8 and
    depth (H, W) float16. Thermal normalization, slicing and segmentation happen later.
    """

    name = ""
    sliced = False          # images wider than 800px are cut into 640px slices
    depth_model = None      # FoundationStereo ONNX used for depth, if any
    urls = {}               # {file name: url} for automatic download
    manual = ""             # manual download instructions
    required = ()           # paths (relative to the dataset root) that must exist
    extra_modalities = {}   # {split: modalities} kept although absent from the manifest

    def __init__(self, root, work_dir):
        self.root = Path(root)
        self.work = Path(work_dir)
        self.stereo = None  # StereoDepth, set by the builder when depth_model is not None

    @property
    def downloadable(self):
        return bool(self.urls)

    def missing(self):
        """Required raw paths that are not present."""
        if not self.root.exists():
            return [str(self.root)]
        return [p for p in self.required if not (self.root / p).exists()]

    def fetch(self):
        """Download and extract `urls` into the dataset root (git URLs are cloned)."""
        self.root.mkdir(parents=True, exist_ok=True)
        for fname, url in self.urls.items():
            if "huggingface.co" in url or url.endswith(".git"):
                subprocess.run(["git", "clone", url, str(self.root)], check=True)
                continue
            target = self.root / fname
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if target.with_suffix(target.suffix + ".done").exists():
                    continue  # already downloaded and extracted
                print(f"[{self.name}] downloading {url}")
                part = Path(f"{target}.part")
                with urllib.request.urlopen(url) as r, open(part, "wb") as f:
                    shutil.copyfileobj(r, f)
                part.rename(target)
            print(f"[{self.name}] extracting {fname}")
            if zipfile.is_zipfile(target):
                with zipfile.ZipFile(target) as z:
                    z.extractall(target.parent)
            elif tarfile.is_tarfile(target):
                with tarfile.open(target) as t:
                    t.extractall(target.parent)
            target.unlink()  # free the space of the archive once extracted
            target.with_suffix(target.suffix + ".done").touch()

    def prepare(self, ids):
        """Optional one-off step for the given sample ids before loading (e.g. ROS bag extraction)."""

    def load(self, split, sample_id):
        raise NotImplementedError
