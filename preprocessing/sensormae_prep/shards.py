"""Final stage shared by all datasets: thermal normalization, 640px slicing,
semantic labels and packing into LZ4-compressed HDF5 shards.

Each sample is stored as one uint8 dataset holding a msgpack dict {modality: raw bytes},
with the per-modality `shapes` and `dtypes` as JSON attributes:

    rgb   (H, W, 3)  uint8
    lwir / nir / swir (H, W, 1) uint16   min-max + CLAHE
    depth (H, W)     float16  FoundationStereo disparity, min-max normalized per frame
    event (H, W, 10) uint8    event counts (2 polarities x 5 time bins)
    segm  (H, W)     uint8    Mask2Former Cityscapes train ids
"""

import json
from pathlib import Path

import cv2
import h5py
import hdf5plugin
import msgpack
import numpy as np

THERMAL = ("lwir", "mwir", "nir", "swir")
CLAHE_CLIP, CLAHE_TILE = 2.0, (8, 8)
SLICE_WIDTH, SLICE_THRESHOLD = 640, 800
SHARD_BYTES = 1800 * 1024**2


def normalize_thermal(img):
    """Min-max scale to uint16, keep the first channel and apply CLAHE -> (H, W, 1)."""
    img = img.astype(np.float32)
    lo, hi = img.min(), img.max()
    img = ((img - lo) / (hi - lo if hi > lo else 1.0) * 65535.0).astype(np.uint16)
    if img.ndim == 3:
        img = img[..., 0]
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
    return clahe.apply(np.ascontiguousarray(img))[..., None]


def slices(width):
    """Overlapping 640px column ranges for images wider than 800px, else the full width."""
    if width <= SLICE_THRESHOLD:
        return [(0, width)]
    n = max(2, int(np.ceil(width / SLICE_WIDTH)))
    starts = np.linspace(0, width - SLICE_WIDTH, n, dtype=int)
    return [(int(s), int(s) + SLICE_WIDTH) for s in starts]


def split_sample(key, sample, sliced):
    """Normalize thermal channels and cut into slices. Returns [(key, sample), ...]."""
    sample = {m: normalize_thermal(a) if m in THERMAL else a for m, a in sorted(sample.items())}
    if not sliced:
        return [(key, sample)]
    ranges = slices(sample["rgb"].shape[1])
    if len(ranges) == 1:
        return [(key, sample)]
    return [(f"{key}_{i}", {m: a[:, x0:x1] for m, a in sample.items()})
            for i, (x0, x1) in enumerate(ranges)]


class Segmenter:
    """Mask2Former (Swin-L, Cityscapes semantic) run on the RGB of every final sample."""

    MODEL_ID = "facebook/mask2former-swin-large-cityscapes-semantic"

    def __init__(self, device="cuda", batch_size=32):
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        self.torch = torch
        self.device = device
        self.batch_size = batch_size
        self.processor = AutoImageProcessor.from_pretrained(self.MODEL_ID)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(
            self.MODEL_ID, torch_dtype=torch.bfloat16).to(device).eval()

    def __call__(self, rgbs):
        masks = []
        for i in range(0, len(rgbs), self.batch_size):
            batch = rgbs[i:i + self.batch_size]
            inputs = self.processor(images=[np.ascontiguousarray(r) for r in batch],
                                    return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device, dtype=self.torch.bfloat16)
            with self.torch.no_grad():
                out = self.model(pixel_values=pixels)
            pred = self.processor.post_process_semantic_segmentation(
                out, target_sizes=[r.shape[:2] for r in batch])
            masks += [m.cpu().numpy().astype(np.uint8) for m in pred]
        return masks


class ShardWriter:
    """Writes `{split}_{n}.h5` shards (numbered after the existing ones) and returns the
    {key: {file, path}} index. Each dataset/split gets its own shards, so it can be rebuilt
    by deleting the files listed in its previous index (`replace`)."""

    def __init__(self, out_dir, split, replace=(), max_bytes=SHARD_BYTES):
        self.out_dir, self.split, self.max_bytes = Path(out_dir), split, max_bytes
        self.file, self.index = None, {}
        for name in replace:
            (self.out_dir / name).unlink(missing_ok=True)

    def _open(self):
        if self.file is not None:
            self.file.close()
        taken = {p.name for p in self.out_dir.glob(f"{self.split}_*.h5")}
        n = next(i for i in range(len(taken) + 1) if f"{self.split}_{i}.h5" not in taken)
        self.name = f"{self.split}_{n}.h5"
        self.file = h5py.File(self.out_dir / self.name, "w")

    def write(self, key, sample):
        if self.file is None or Path(self.file.filename).stat().st_size >= self.max_bytes:
            self._open()
        blob = msgpack.packb({m: a.tobytes() for m, a in sample.items()}, use_bin_type=True)
        ds = self.file.create_dataset(key, data=np.frombuffer(blob, np.uint8), **hdf5plugin.LZ4())
        ds.attrs["shapes"] = json.dumps({m: list(a.shape) for m, a in sample.items()})
        ds.attrs["dtypes"] = json.dumps({m: str(a.dtype) for m, a in sample.items()})
        self.index[key] = {"file": self.name, "path": key}

    def close(self):
        if self.file is not None:
            self.file.close()
        return self.index


def read_sample(h5_file, key):
    """Decode one packed sample back into {modality: array}."""
    ds = h5_file[key]
    shapes, dtypes = json.loads(ds.attrs["shapes"]), json.loads(ds.attrs["dtypes"])
    data = msgpack.unpackb(ds[()].tobytes(), raw=False)
    return {m: np.frombuffer(data[m], dtypes[m]).reshape(shapes[m]) for m in shapes}
