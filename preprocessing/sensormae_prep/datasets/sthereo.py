from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from ..common import crop_resize, imread_raw, load_yaml, nearest
from .base import Dataset

CALIB = Path(__file__).resolve().parents[2] / "calib" / "sthereo"  # hand-adjusted calibration
SEQUENCES = ("kaist_afternoon", "kaist_evening", "kaist_morning", "snu_afternoon",
             "snu_evening", "snu_morning", "valley_afternoon", "valley_evening", "valley_morning")
# Common FoV after rectification, found by hand (STheReO test_data notebook).
BOX_RGB = (0, 0, 1280, 560)
BOX_LWIR = (142, 172, 533, 352)
OUT_SIZE = (1280, 560)
MAX_DT_NS = 50e6


def rectify_maps(name):
    c = load_yaml(CALIB / f"{name}.yaml")
    m = lambda k, s: np.array(c[k]["data"], np.float64).reshape(s)  # noqa: E731
    return cv2.initUndistortRectifyMap(
        m("camera_matrix", (3, 3)), m("distortion_coefficients", (-1, 1)),
        m("rectification_matrix", (3, 3)), m("projection_matrix", (3, 4)),
        (c["image_width"], c["image_height"]), cv2.CV_16SC2)


class STheReO(Dataset):
    """Debayered + rectified stereo RGB and 14-bit left thermal. Every 3rd left frame is
    used (10 -> 3.3 Hz); right and LWIR frames are the nearest within 50 ms. All images
    are cropped to the common FoV, resized to 1280x560 and cut into two 640px slices.
    Ids are the left image timestamps; val is every ~50th frame of each sequence."""

    name = "sthereo"
    sliced = True
    depth_model = "foundation_stereo_s_sthereo.onnx"  # input 288x640
    manual = ("https://sites.google.com/view/rpmsthereo/ (request access); download the image "
              "data of the 9 sequences into sthereo/<sequence>/image/{stereo_left,stereo_right,"
              "stereo_thermal_14_left}")
    required = tuple(f"{s}/image/stereo_left" for s in SEQUENCES)

    def __init__(self, root, work_dir):
        super().__init__(root, work_dir)
        self.maps = {c: rectify_maps(c) for c in ("left", "right", "thermal_14bit_left")}

    @lru_cache(maxsize=None)
    def _sequence_of(self):
        index = {}
        for seq in SEQUENCES:
            for p in (self.root / seq / "image/stereo_left").glob("*.png"):
                index[p.stem] = seq
        return index

    @lru_cache(maxsize=16)
    def _stamps(self, seq, folder):
        files = sorted((self.root / seq / "image" / folder).glob("*.png"))
        return files, np.array([int(p.stem) for p in files], np.int64)

    def _rectified(self, path, calib, bayer):
        img = imread_raw(path)
        if bayer:
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_BGGR2RGB)
        return cv2.remap(img, *self.maps[calib], interpolation=cv2.INTER_LINEAR)

    def _nearest(self, seq, folder, ts):
        files, stamps = self._stamps(seq, folder)
        i = nearest(stamps, ts, MAX_DT_NS)
        if i is None:
            raise LookupError(f"no {folder} frame within 50 ms of {ts}")
        return files[i]

    def load(self, split, sample_id):
        seq = self._sequence_of()[sample_id]
        image = self.root / seq / "image"
        ts = int(sample_id)
        rgb = self._rectified(image / f"stereo_left/{sample_id}.png", "left", True)
        right = self._rectified(self._nearest(seq, "stereo_right", ts), "right", True)
        lwir = self._rectified(self._nearest(seq, "stereo_thermal_14_left", ts),
                               "thermal_14bit_left", False)
        rgb = crop_resize(rgb, BOX_RGB, OUT_SIZE)
        right = crop_resize(right, BOX_RGB, OUT_SIZE)
        return {"rgb": rgb, "depth": self.stereo(rgb, right),
                "lwir": crop_resize(lwir, BOX_LWIR, OUT_SIZE)}
