from functools import lru_cache

import h5py
import hdf5plugin  # noqa: F401  (DSEC events are blosc compressed)
import numpy as np

from ..common import (common_fov, crop_resize, event_frame, image_corners, imread_rgb,
                      load_yaml, project_corners, remap_channels)
from .base import Dataset

E_W, E_H = 640, 480        # event camera resolution
RGB_W, RGB_H = 1440, 1080  # rectified RGB resolution
SUBSAMPLE = 5              # 20 Hz images -> 4 Hz
MIN_EVENTS = 200
MARGIN = (slice(5, -45), slice(20, -20))  # trims the rectification borders -> 430x600


def K_from(entry):
    fx, fy, cx, cy = entry["camera_matrix"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


class DSEC(Dataset):
    """Rectified left RGB + left event camera + stereo depth.

    Every 5th exposure is used. Events of the 50 ms before the exposure are accumulated,
    rectified with the provided map, and the RGB pair is cropped to the event FoV and
    resized to 640x480. Sample ids are a running 8-digit index over the sampled frames of
    the (sorted) scenes of each official split; val is 2% held out from train."""

    name = "dsec"
    depth_model = "foundation_stereo_s_640.onnx"  # input 480x640
    urls = {f"{s}/{s}_{m}.zip": f"https://download.ifi.uzh.ch/rpg/DSEC/{s}_coarse/{s}_{m}.zip"
            for s in ("train", "test") for m in ("images", "events", "calibration")}
    manual = ("https://dsec.ifi.uzh.ch/dsec-datasets/download/ ; unzip {train,test}_{images,"
              "events,calibration}.zip into dsec/train/ and dsec/test/ (as the official "
              "download script does), giving dsec/<split>/<scene>/{images,events,calibration}")
    required = ("train", "test")

    @lru_cache(maxsize=None)
    def _index(self, folder):
        """{sample id: (scene, sampled frame number, exposure timestamp in s)}."""
        index, global_idx = {}, 0
        for scene in sorted(p.name for p in (self.root / folder).iterdir()):
            lines = (self.root / folder / scene /
                     "images/left/exposure_timestamps.txt").read_text().splitlines()
            stamps = np.array([l.strip().split()[1] for l in lines[1::SUBSAMPLE]], np.int64) / 1e6
            for i, ts in enumerate(stamps):
                global_idx += 1
                index[f"{global_idx:08d}"] = (scene, i * SUBSAMPLE, ts)
        return index

    @lru_cache(maxsize=4)
    def _scene(self, folder, scene):
        calib = load_yaml(self.root / folder / scene / "calibration/cam_to_cam.yaml")
        K_event = K_from(calib["intrinsics"]["camRect0"])
        K_rgb = K_from(calib["intrinsics"]["camRect1"])
        T = np.array(calib["extrinsics"]["T_10"])
        box = common_fov([project_corners(image_corners(E_W, E_H), K_event, K_rgb, T)], RGB_W, RGB_H)
        events_dir = self.root / folder / scene / "events/left"
        with h5py.File(events_dir / "rectify_map.h5", "r") as f:
            rect = f["rectify_map"][:]
        events = h5py.File(events_dir / "events.h5", "r")
        return box, rect[..., 0].astype(np.float32), rect[..., 1].astype(np.float32), events

    def _events(self, events, ts):
        """Event frame of the 50 ms window ending at `ts` (seconds), or None if too few."""
        t_offset = int(events["t_offset"][()]) if "t_offset" in events else 0
        # ms_to_idx[k] is the first event with t >= k ms: read a slightly wider window.
        ms_to_idx = events["ms_to_idx"]
        t0_ms = int((ts - 0.05) * 1e6 - t_offset) // 1000 - 1
        t1_ms = int(ts * 1e6 - t_offset) // 1000 + 2
        lo = int(ms_to_idx[max(t0_ms, 0)])
        hi = int(ms_to_idx[t1_ms]) if t1_ms < len(ms_to_idx) else len(events["events/t"])
        t = (events["events/t"][lo:hi].astype(np.int64) + t_offset) * 1e-6
        x, y, p = (events[f"events/{k}"][lo:hi].astype(np.int32) for k in "xyp")
        frame, n = event_frame(x, y, p, t, ts, E_W, E_H)
        return frame if n >= MIN_EVENTS else None

    def load(self, split, sample_id):
        folder = "test" if split == "test" else "train"
        scene, frame, ts = self._index(folder)[sample_id]
        box, map_x, map_y, events = self._scene(folder, scene)
        images = self.root / folder / scene / "images"
        left = crop_resize(imread_rgb(images / f"left/rectified/{frame:06d}.png"), box, (E_W, E_H))
        right = crop_resize(imread_rgb(images / f"right/rectified/{frame:06d}.png"), box, (E_W, E_H))
        left, right = left[MARGIN], right[MARGIN]
        sample = {"rgb": left, "depth": self.stereo(left, right)}
        event = self._events(events, ts)
        if event is not None:
            sample["event"] = remap_channels(event, map_x, map_y)[MARGIN]
        return sample
