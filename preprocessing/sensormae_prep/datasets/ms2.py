from functools import lru_cache

import numpy as np

from ..common import crop_resize, fov_box, imread_raw, imread_rgb
from .base import Dataset

# Sequence index used in the sample ids -> MS2 sync_data sequence (index 13 is unused).
SEQUENCES = {
    "01": "_2021-08-06-17-44-55", "02": "_2021-08-13-16-50-57", "03": "_2021-08-06-11-23-45",
    "04": "_2021-08-06-17-21-04", "05": "_2021-08-06-11-37-46", "06": "_2021-08-13-21-58-13",
    "07": "_2021-08-13-21-18-04", "08": "_2021-08-13-21-36-10", "09": "_2021-08-13-16-08-46",
    "10": "_2021-08-13-16-14-48", "11": "_2021-08-13-17-06-04", "12": "_2021-08-06-10-59-33",
    "14": "_2021-08-13-22-03-03", "15": "_2021-08-13-15-46-56", "16": "_2021-08-06-16-59-13",
    "17": "_2021-08-13-22-36-41", "18": "_2021-08-06-16-45-28", "19": "_2021-08-13-22-16-02",
    "20": "_2021-08-06-16-19-00", "21": "_2021-08-13-16-31-10",
}

# Dropbox keys of the sync_data archives (official MS2 download script), in sequence order.
DROPBOX_KEYS = (
    "65hkrz768n6wnqt",
    "gith8ciig2eccla",
    "h03hx66ov057l7k",
    "yhpyyydi08vwp1p",
    "sht99tbkwrhxssg",
    "r75o84a6qayj2p6",
    "54hcoy2hzs2ausg",
    "o06b2th99fpgxs7",
    "n6pieerypk5f77i",
    "umafrxglzktetgm",
    "psn3gyks9mwuvyh",
    "zk2pc0lisaeevtw",
    "f6fnp41xaj5qw3l",
    "pqi0bk29z3cisq8",
    "33jfb59tux847tk",
    "urc65q9luelfflc",
    "fhqhja5e3x4g698",
    "m38zrl2r8nvarot",
    "g4j375qx7yqfv7a",
    "so29g3cgj7e0yed",
)


def to_T(R, t_mm):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, np.asarray(t_mm).reshape(-1)[:3] / 1000.0
    return T


class MS2(Dataset):
    """Rectified, synchronized RGB / NIR / LWIR (left cameras) + stereo depth from the RGB
    pair. Every 3rd frame is used; ids are {sequence index}_{frame}, val is the first 2%
    of every sequence. NIR and LWIR are cropped to the RGB FoV and resized to 1224x384,
    then everything is cut into two 640px slices."""

    name = "ms2"
    sliced = True
    depth_model = "foundation_stereo_s_ms2.onnx"  # input 192x608
    urls = {f"sync_data/{seq}.tar.bz2": f"https://dl.dropboxusercontent.com/s/{key}/{seq}.tar.bz2"
            for key, seq in zip(DROPBOX_KEYS, sorted(SEQUENCES.values()))}
    manual = ("https://sites.google.com/view/multi-spectral-stereo-dataset (sync_data, all 20 "
              "sequences); extract into ms2/sync_data/")
    required = tuple(f"sync_data/{s}" for s in SEQUENCES.values())

    @lru_cache(maxsize=None)
    def _boxes(self, seq):
        calib = np.load(self.root / "sync_data" / seq / "calib.npy", allow_pickle=True).item()
        T_nir2rgb = to_T(calib["R_nir2rgb"], calib["T_nir2rgb"])
        T_nir2lwir = to_T(calib["R_nir2thr"], calib["T_nir2thr"])
        T_rgb2nir = np.linalg.inv(T_nir2rgb)
        return calib, T_rgb2nir, T_nir2lwir @ T_rgb2nir

    def load(self, split, sample_id):
        seq_index, frame = sample_id.split("_")
        seq = self.root / "sync_data" / SEQUENCES[seq_index]
        rgb = imread_rgb(seq / f"rgb/img_left/{frame}.png")
        right = imread_rgb(seq / f"rgb/img_right/{frame}.png")
        lwir = imread_raw(seq / f"thr/img_left/{frame}.png")
        nir = imread_raw(seq / f"nir/img_left/{frame}.png")

        calib, T_rgb2nir, T_rgb2lwir = self._boxes(SEQUENCES[seq_index])
        size = (rgb.shape[1], rgb.shape[0])
        box_lwir = fov_box(calib["K_rgbL"], size, calib["K_thrL"],
                           (lwir.shape[1], lwir.shape[0]), T_rgb2lwir)
        box_nir = fov_box(calib["K_rgbL"], size, calib["K_nirL"],
                          (nir.shape[1], nir.shape[0]), T_rgb2nir)
        return {"rgb": rgb, "depth": self.stereo(rgb, right),
                "lwir": crop_resize(lwir, box_lwir, size),
                "nir": crop_resize(nir, box_nir, size)}
