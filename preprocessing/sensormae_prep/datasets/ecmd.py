from functools import lru_cache
from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .. import rosbag
from ..common import crop_resize, event_frame, fov_box, load_yaml, nearest, undistort
from .base import Dataset

CALIB = Path(__file__).resolve().parents[2] / "calib" / "ecmd.yaml"
TOPICS = {"rgb": "/camera_left/image_color/compressed", "right": "/camera_right/image_color/compressed",
          "lwir": "/hikrobot_infrared_camera/rgb", "event": "/DVXplorer_right/events"}
LWIR_DT_NS = -60_000_000   # extra LWIR delay applied when matching to RGB
MAX_DT_NS = 50_000_000
EVENT_MSG_WINDOW_S = 0.2   # event messages within +-0.2 s of the image are considered
MIN_EVENT_RATIO = 0.01     # frames with fewer than 1% of W*H events are dropped


def name_to_ns(name):
    return int(name.replace("_", ""))


class ECMD(Dataset):
    """ECMD rosbags: left/right RGB, LWIR (Hikrobot) and the right DVXplorer event camera.

    Images are undistorted, LWIR and right RGB are matched to the left RGB within 50 ms,
    events of the 50 ms before every left frame are accumulated and undistorted. Left/right
    RGB and LWIR are cropped to the event camera FoV and resized to its 640x480 grid; depth
    is predicted on the cropped stereo pair. Ids are {bag index}_{left RGB timestamp}, the
    bag index being the 1-based position of the bag in the sorted list of bag names."""

    name = "ecmd"
    depth_model = "foundation_stereo_s_640.onnx"  # input 480x640
    manual = ("https://arclab-hku.github.io/ecmd/ : download all rosbags (Tunnel_difficult_c and "
              "Tunnel_easy_b are broken and not used) into ecmd/ (any sub-folder structure)")
    required = ()

    def __init__(self, root, work_dir):
        super().__init__(root, work_dir)
        c = self.c = load_yaml(CALIB)
        K = lambda k: np.array(c[k], np.float64).reshape(3, 3)  # noqa: E731
        T_ev = np.array(c["T_event_imu"])
        self.size = (c["w_event"], c["h_event"])
        # Crop box of each camera covering the event camera FoV.
        self.boxes = {cam: fov_box(K("K_event"), self.size, K(f"K_{cam}"), (c[f"w_{cam}"], c[f"h_{cam}"]),
                                   np.linalg.inv(np.array(c[f"T_{cam}_imu"])) @ T_ev)
                      for cam in ("rgbl", "rgbr", "lwir")}

    @lru_cache(maxsize=None)
    def bags(self):
        return sorted(self.root.rglob("*.bag"), key=lambda p: p.name)

    def missing(self):
        return [] if self.root.exists() and self.bags() else [f"{self.root}/*.bag"]

    def prepare(self, ids):
        """Extract the aligned frames of the given ids into work/<bag>.h5 (once per bag)."""
        wanted = {}
        for sample_id in ids:
            idx, name = sample_id.split("_", 1)
            wanted.setdefault(int(idx), set()).add(name)
        self.work.mkdir(parents=True, exist_ok=True)
        for idx, names in sorted(wanted.items()):
            bag = self.bags()[idx - 1]
            out = self.work / f"{bag.stem}.h5"
            if out.exists():
                with h5py.File(out, "r") as f:
                    if f.attrs.get("complete") and names <= set(f):
                        continue
            self._extract(bag, names, out)

    def _undistort(self, img, cam):
        c = self.c
        return undistort(img, c[f"K_{cam}"], c[f"D_{cam}"], (c[f"w_{cam}"], c[f"h_{cam}"]))

    def _event_frame(self, evs, ts_ns):
        w, h = self.size
        t = evs["secs"] + evs["nsecs"] * 1e-9
        t_end = ts_ns * 1e-9
        keep = (t >= t_end - 0.05) & (t <= t_end)
        if keep.sum() < MIN_EVENT_RATIO * w * h:
            return None
        e = evs[keep]
        frame, _ = event_frame(e["x"], e["y"], e["p"], t[keep], t_end, w, h, select=False)
        c = self.c
        K = np.array(c["K_event"], np.float64).reshape(3, 3)
        D = np.array(c["D_event"], np.float64).reshape(-1, 1)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, self.size, 0, self.size)
        return cv2.undistort(frame, K, D, None, new_K)

    def _extract(self, bag, names, out):
        c = self.c
        shifts = {"rgb": c["timeshift_s_rgbl"], "right": c["timeshift_s_rgbr"],
                  "lwir": c["timeshift_s_lwir"]}
        with rosbag.open_bag(bag) as reader:
            conns = {k: [x for x in reader.connections if x.topic == t] for k, t in TOPICS.items()}
            # Pass 1: timestamps of every frame, named by their shifted header stamp.
            stamps = {k: [] for k in shifts}
            for k in shifts:
                for _, _, raw in reader.messages(conns[k]):
                    stamps[k].append(name_to_ns(rosbag.time_name(*rosbag.stamp(raw), shifts[k])))
            stamps = {k: np.array(v, np.int64) for k, v in stamps.items()}
            lwir_matched = np.sort(stamps["lwir"]) + LWIR_DT_NS

            # Which right / lwir message every wanted left frame uses.
            targets = {"rgb": {}, "right": {}, "lwir": {}}
            for name in names:
                ts = name_to_ns(name)
                targets["rgb"][ts] = name
                i = nearest(np.sort(stamps["right"]), ts, MAX_DT_NS)
                j = nearest(lwir_matched, ts, MAX_DT_NS)
                if i is not None:
                    targets["right"][int(np.sort(stamps["right"])[i])] = name
                if j is not None:
                    targets["lwir"][int(lwir_matched[j] - LWIR_DT_NS)] = name

            # Pass 2: decode the selected images and the events around every wanted frame.
            frames = {name: {} for name in names}
            for k in ("rgb", "right", "lwir"):
                for _, _, raw in reader.messages(conns[k]):
                    ts = name_to_ns(rosbag.time_name(*rosbag.stamp(raw), shifts[k]))
                    if ts in targets[k]:
                        img = rosbag.image(raw) if k == "lwir" else rosbag.compressed_image(raw)
                        cam = {"rgb": "rgbl", "right": "rgbr", "lwir": "lwir"}[k]
                        frames[targets[k][ts]][k] = self._undistort(img, cam)
            centers = np.array(sorted(name_to_ns(n) for n in names), np.int64)
            events = {int(t): [] for t in centers}
            win = int(EVENT_MSG_WINDOW_S * 1e9)
            for _, _, raw in tqdm(reader.messages(conns["event"]), desc=f"events {bag.stem}",
                                  leave=False):
                s, ns = rosbag.stamp(raw)
                t = s * 10**9 + ns
                lo, hi = np.searchsorted(centers, [t - win, t + win])
                if lo < hi:
                    ev = rosbag.events(raw)
                    for center in centers[lo:hi]:
                        events[int(center)].append(ev)

        with h5py.File(out, "w") as f:
            for name, mods in frames.items():
                evs = events[name_to_ns(name)]
                ev = self._event_frame(np.concatenate(evs), name_to_ns(name)) if evs else None
                if ev is not None:
                    mods["event"] = ev
                g = f.create_group(name)
                for k, a in mods.items():
                    g.create_dataset(k, data=a, compression="lzf")
            f.attrs["complete"] = True

    def load(self, split, sample_id):
        idx, name = sample_id.split("_", 1)
        with h5py.File(self.work / f"{self.bags()[int(idx) - 1].stem}.h5", "r") as f:
            raw = {k: v[()] for k, v in f[name].items()}
        sample = {}
        if "rgb" in raw and "right" in raw:
            rgb = cv2.cvtColor(crop_resize(raw["rgb"], self.boxes["rgbl"], self.size), cv2.COLOR_BGR2RGB)
            right = cv2.cvtColor(crop_resize(raw["right"], self.boxes["rgbr"], self.size), cv2.COLOR_BGR2RGB)
            sample.update(rgb=rgb, depth=self.stereo(rgb, right))
        if "lwir" in raw:
            sample["lwir"] = crop_resize(raw["lwir"], self.boxes["lwir"], self.size)
        if "event" in raw:
            sample["event"] = raw["event"]
        return sample
