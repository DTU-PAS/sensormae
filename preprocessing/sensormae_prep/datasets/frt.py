from functools import lru_cache

from ..common import imread_rgb, imread_thermal
from .base import Dataset

CROP = (300, 0, 1700, 650)  # x1, y1, x2, y2: removes the border artifacts -> 1400x650


class FreiburgThermal(Dataset):
    """Freiburg Thermal (HeatNet) RGB + aligned LWIR. Sample ids are
    {sequence index}_{timestamp}; frames are cropped to 1400x650 and cut into three
    640px slices. val is held out from train.

    In the test split the RGB and IR file names disagree in the nanosecond field
    (10 vs 9 digits); the IR is matched on the truncated field."""

    name = "frt"
    sliced = True
    urls = {f"{s}.zip": f"http://aisdatasets.informatik.uni-freiburg.de/freiburg-thermal-segmentation/{s}.zip"
            for s in ("train", "test")}
    manual = ("http://thermal.cs.uni-freiburg.de/ ; extract into frt/ so that "
              "frt/train/seq_00_day and frt/test/day exist")
    required = ("train", "test")
    extra_modalities = {"test": ("lwir",)}

    @lru_cache(maxsize=None)
    def _index(self, folder):
        """{sample id: (rgb path, ir path)}; ids follow the sorted sequence folders."""
        index = {}
        base = self.root / folder
        names = sorted(p.name for p in base.iterdir() if not p.name.startswith("."))
        for seq_index, seq in enumerate(names):
            seq_dir = base / seq
            if not seq_dir.is_dir():
                continue
            parts = sorted(p for p in seq_dir.iterdir() if p.is_dir() and p.name.isdigit())
            for part in parts or [seq_dir]:
                irs = {p.name: p for p in (part / "fl_ir_aligned").glob("*.png")}
                for rgb in sorted((part / "fl_rgb").glob("*.png")):
                    if folder == "test":  # fl_ir_aligned_<s>_<ns>_rgb.png / ..._<ns[:9]>_ir.png
                        sec, nsec = rgb.stem.split("_")[-3:-1]
                        ir = irs.get(f"fl_ir_aligned_{sec}_{nsec[:9]}_ir.png")
                    else:                 # fl_rgb_<s>_<ns>.png / fl_ir_aligned_<s>_<ns>.png
                        sec, nsec = rgb.stem.split("_")[-2:]
                        ir = irs.get(f"fl_ir_aligned_{sec}_{nsec}.png")
                    index[f"{seq_index:02d}_{sec}_{nsec}"] = (rgb, ir)
        return index

    def load(self, split, sample_id):
        rgb_path, ir_path = self._index("test" if split == "test" else "train")[sample_id]
        x1, y1, x2, y2 = CROP
        sample = {"rgb": imread_rgb(rgb_path)[y1:y2, x1:x2]}
        if ir_path is not None:
            sample["lwir"] = imread_thermal(ir_path)[y1:y2, x1:x2]
        return sample
