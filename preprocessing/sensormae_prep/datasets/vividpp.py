from pathlib import Path

import cv2
import h5py
import numpy as np
from tqdm import tqdm

from .. import rosbag
from ..common import crop_resize, event_frame, fov_box, load_yaml, nearest
from .base import Dataset

CALIB = Path(__file__).resolve().parents[2] / "calib" / "vividpp.yaml"
# Sequence index used in the sample ids -> Vivid++ driving sequence.
SEQUENCES = {
    "01": "driving_vision/campus_day1", "02": "driving_vision/campus_day2",
    "03": "driving_vision/campus_day2_2", "04": "driving_vision/campus_evening",
    "05": "driving_vision/campus_morning", "06": "driving_vision/campus_morning_2",
    "07": "driving_vision/campus_morning_manual", "08": "driving_vision/campus_morning_manual_small",
    "09": "driving_vision/city_day1", "10": "driving_vision/city_day2",
    "11": "driving_vision/city_morning", "12": "driving_vision/city_morning_manual",
    "13": "driving_vision/city_night", "14": "driving_full/campus_day1",
    "15": "driving_full/campus_day2", "16": "driving_full/campus_night",
    "17": "driving_full/city_day1", "18": "driving_full/city_night",
    "19": "driving_vision/urban_evening", "20": "driving_vision/urban_evening_road",
    "21": "driving_vision/urban_morning", "22": "driving_vision/urban_morning_manual",
    "23": "driving_vision/urban_night",
}
TOPICS = {"rgb": "/camera/image_color", "lwir": "/thermal/image_raw", "event": "/dvs/events"}
MAX_DT_NS = 50_000_000
EVENT_MSG_WINDOW_S = 0.2   # event messages within +-0.2 s of the image are considered
MIN_EVENT_RATIO = 0.01     # frames with fewer than 1% of W*H events are dropped


def name_to_ns(name):
    return int(name.replace("_", ""))


class VividPP(Dataset):
    """Vivid++ driving sequences (driving setup calibration): RGB, LWIR and DVS events.

    RGB and LWIR are undistorted (fisheye model), LWIR is matched to every selected RGB
    frame within 50 ms, events of the 50 ms before the frame are accumulated and
    undistorted. RGB and LWIR are cropped to the event camera FoV and resized to its
    640x480 grid. There is no stereo pair, hence no depth. Ids are
    {sequence index}_{RGB timestamp}, the index following SEQUENCES."""

    name = "vividpp"
    manual = ("https://visibilitydataset.github.io/ (download links on the site): the rosbags of "
              "the driving sequences listed in sensormae_prep/datasets/vividpp.py, into "
              "vividpp/driving_full/ and vividpp/driving_vision/ as <sequence>.bag "
              "(or <sequence>/*.bag)")
    required = ("driving_full", "driving_vision")

    def __init__(self, root, work_dir):
        super().__init__(root, work_dir)
        c = self.c = load_yaml(CALIB)
        K = lambda k: np.array(c[k], np.float64).reshape(3, 3)  # noqa: E731

        def T(rot, trans):
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = np.array(c[rot]), np.array(c[trans])
            return m

        T_ev2rgb = T("R_event2rgb", "T_event2rgb")
        T_ev2lwir = np.linalg.inv(T("R_lwir2rgb", "T_lwir2rgb")) @ T_ev2rgb
        self.size = (c["w_event"], c["h_event"])
        self.box_rgb = fov_box(K("K_event"), self.size, K("K_rgb"), (c["w_rgb"], c["h_rgb"]), T_ev2rgb)
        self.box_lwir = fov_box(K("K_event"), self.size, K("K_lwir"), (c["w_lwir"], c["h_lwir"]), T_ev2lwir)

    def _bags(self, seq):
        single = self.root / f"{seq}.bag"
        return [single] if single.exists() else sorted((self.root / seq).glob("*.bag"))

    def _undistort(self, img, cam):
        c = self.c
        K = np.array(c[f"K_{cam}"], np.float64).reshape(3, 3)
        D = np.array(c[f"D_{cam}"], np.float64).reshape(-1, 1)
        size = (c[f"w_{cam}"], c[f"h_{cam}"])
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, size, cv2.CV_16SC2)
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def _event_frame(self, evs, ts_ns):
        w, h = self.size
        t = evs["secs"] + evs["nsecs"] * 1e-9
        t_end = ts_ns * 1e-9
        keep = (t >= t_end - 0.05) & (t <= t_end)
        if keep.sum() < MIN_EVENT_RATIO * w * h:
            return None
        e = evs[keep]
        frame, _ = event_frame(e["x"], e["y"], e["p"], t[keep], t_end, w, h, select=False)
        K = np.array(self.c["K_event"], np.float64).reshape(3, 3)
        D = np.array(self.c["D_event"], np.float64).reshape(-1, 1)
        new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, self.size, 0, self.size)
        return cv2.undistort(frame, K, D, None, new_K)

    def prepare(self, ids):
        """Extract the aligned frames of the given ids into work/<sequence index>.h5."""
        wanted = {}
        for sample_id in ids:
            idx, name = sample_id.split("_", 1)
            wanted.setdefault(idx, set()).add(name)
        self.work.mkdir(parents=True, exist_ok=True)
        for idx, names in sorted(wanted.items()):
            out = self.work / f"{idx}.h5"
            if out.exists():
                with h5py.File(out, "r") as f:
                    if f.attrs.get("complete") and names <= set(f):
                        continue
            self._extract(SEQUENCES[idx], names, out)

    def _messages(self, seq, kind):
        for bag in self._bags(seq):
            with rosbag.open_bag(bag) as r:
                conns = [x for x in r.connections if x.topic == TOPICS[kind]]
                for _, _, raw in r.messages(conns):
                    s, ns = rosbag.stamp(raw)
                    yield f"{s:09d}_{ns:09d}", raw

    def _extract(self, seq, names, out):
        # LWIR frame used by every wanted RGB frame (nearest within 50 ms).
        lwir_names = sorted(n for n, _ in self._messages(seq, "lwir"))
        lwir_ts = np.array([name_to_ns(n) for n in lwir_names], np.int64)
        lwir_for = {}
        for name in names:
            i = nearest(lwir_ts, name_to_ns(name), MAX_DT_NS)
            if i is not None:
                lwir_for.setdefault(lwir_names[i], []).append(name)

        frames = {name: {} for name in names}
        for msg_name, raw in self._messages(seq, "rgb"):
            if msg_name in frames:
                img = rosbag.image(raw)
                if rosbag.image_encoding(raw) in ("bgr8", "8UC3"):
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames[msg_name]["rgb"] = self._undistort(img, "rgb")
        for msg_name, raw in self._messages(seq, "lwir"):
            for name in lwir_for.get(msg_name, []):
                frames[name]["lwir"] = self._undistort(rosbag.image(raw), "lwir")

        centers = np.array(sorted(name_to_ns(n) for n in names), np.int64)
        events = {int(t): [] for t in centers}
        win = int(EVENT_MSG_WINDOW_S * 1e9)
        for msg_name, raw in tqdm(self._messages(seq, "event"), desc=f"events {seq}", leave=False):
            t = name_to_ns(msg_name)
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
        with h5py.File(self.work / f"{idx}.h5", "r") as f:
            raw = {k: v[()] for k, v in f[name].items()}
        sample = {}
        if "rgb" in raw:
            sample["rgb"] = crop_resize(raw["rgb"], self.box_rgb, self.size)
        if "lwir" in raw:
            sample["lwir"] = crop_resize(raw["lwir"], self.box_lwir, self.size)
        if "event" in raw:
            sample["event"] = raw["event"]  # already on the 640x480 event grid
        return sample
