import os
import random
import zipfile
from functools import lru_cache

from ..common import imread_rgb, imread_thermal
from .base import Dataset

SUBSAMPLE = 5
VAL_RATIO = 0.02
SEED = 42


class RASMD(Dataset):
    """RGB-SWIR pairs from the `croped_data` folders of the release (SWIR is used as the
    NIR modality of the paper). Every 5th pair is used, the pairs are shuffled with seed 42
    and numbered 000000, 000001, ... (train first, then the 2% val). Images wider than
    800px are cut into 640px slices. Folders are walked in sorted order."""

    name = "rasmd"
    sliced = True
    urls = {"RASMD_aligned.zip": "huggingface:STL-Yonsei/RASMD"}
    manual = ("https://huggingface.co/datasets/STL-Yonsei/RASMD : accept the terms on the page "
              "(granted automatically), log in with `huggingface-cli login`, then `download` "
              "fetches RASMD_aligned.zip (46 GB, registered RGB-SWIR pairs) and extracts it into "
              "rasmd/ (the pipeline looks for the */croped_data/{rgb,swir} folders inside)")
    required = ()

    def fetch(self):
        from huggingface_hub import hf_hub_download

        self.root.mkdir(parents=True, exist_ok=True)
        done = self.root / "RASMD_aligned.zip.done"
        if done.exists():
            return
        archive = hf_hub_download("STL-Yonsei/RASMD", "RASMD_aligned.zip", repo_type="dataset",
                                  local_dir=self.root)
        print(f"[{self.name}] extracting RASMD_aligned.zip")
        with zipfile.ZipFile(archive) as z:
            z.extractall(self.root)
        os.remove(archive)
        done.touch()

    def missing(self):
        return [] if self._pairs() else [f"{self.root}/**/croped_data"]

    @lru_cache(maxsize=None)
    def _pairs(self):
        pairs = []
        for root, dirs, _ in os.walk(self.root):
            dirs.sort()
            if os.path.basename(root) != "croped_data":
                continue
            rgb = sorted(f for f in os.listdir(os.path.join(root, "rgb")) if f.endswith(".jpg"))
            swir = sorted(f for f in os.listdir(os.path.join(root, "swir")) if f.endswith(".jpg"))
            rgb, swir = rgb[::SUBSAMPLE], swir[::SUBSAMPLE]
            n = min(len(rgb), len(swir))
            pairs += [(os.path.join(root, "rgb", r), os.path.join(root, "swir", s))
                      for r, s in zip(rgb[:n], swir[:n])]
        random.Random(SEED).shuffle(pairs)
        n_val = int(len(pairs) * VAL_RATIO)
        ordered = pairs[n_val:] + pairs[:n_val]  # train ids first, then val
        return {f"{i:06d}": p for i, p in enumerate(ordered)}

    def load(self, split, sample_id):
        rgb, swir = self._pairs()[sample_id]
        return {"rgb": imread_rgb(rgb), "swir": imread_thermal(swir)}
