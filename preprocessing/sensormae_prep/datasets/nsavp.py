from functools import lru_cache

import cv2
import h5py
import numpy as np

from ..common import common_fov, crop_resize, event_frame, image_corners, load_yaml, project_corners
from .base import Dataset

SEQUENCES = ("R0_FA0", "R0_FN0", "R0_FS0", "R0_RA0", "R0_RN0", "R0_RS0", "R1_DA0", "R1_FA0", "R1_RA0")
E_W, E_H = 640, 480
MIN_EVENTS = 200
FOV_DEPTH_M = 10.0
# cam0: thermal (adk) left, cam4 / cam5: RGB left / right, cam6: DVXplorer left
FRAME_IDS = {"lwir": "adk_left_optical", "rgb": "rgb_left_optical",
             "right": "rgb_right_optical", "event": "dvxplorer_left_optical"}


def K_of(cam):
    fx, fy, cx, cy = cam["intrinsics"]
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], np.float64)


def dist_of(cam):
    d = np.zeros(5)
    c = np.asarray(cam["distortion_coeffs"], np.float64)
    d[:min(5, c.size)] = c[:5]
    return d


class NSAVP(Dataset):
    """NSAVP (left RGB, right RGB, left thermal, left DVXplorer) from the released HDF5 files.

    1. Images are undistorted (alpha 0.5) and cropped to the valid ROI; frames present in
       all three image streams are used (ids are {sequence}_{timestamp ns}).
    2. Depth is predicted on the undistorted stereo pair, given in BGR channel order.
    3. Events of the 50 ms before the frame are accumulated and undistorted.
    4. Everything is cropped to the common FoV of all cameras (projected into the thermal
       image at 10 m) and resized to the thermal crop size (639x511)."""

    name = "nsavp"
    depth_model = "foundation_stereo_nsavp.onnx"  # input 512x608
    manual = ("https://umautobots.github.io/nsavp (download via the project page / Globus, "
              "format: https://github.com/umautobots/nsavp_tools); for the 9 sequences put "
              "<SEQ>_rgb_left.h5, <SEQ>_rgb_right.h5, <SEQ>_adk_left.h5, <SEQ>_dvxplorer_left.h5 "
              "and <SEQ>_C1_calibration_results.yaml into nsavp/<SEQ>/")
    required = tuple(f"{s}/{s}_rgb_left.h5" for s in SEQUENCES)

    @lru_cache(maxsize=None)
    def _calib(self, seq):
        calib = load_yaml(self.root / seq / f"{seq}_C1_calibration_results.yaml")
        cams = {"lwir": calib["cam0"], "rgb": calib["cam4"], "right": calib["cam5"], "event": calib["cam6"]}
        for kind, cam in cams.items():  # the chain below relies on this camera order
            if cam.get("frame_id", FRAME_IDS[kind]) != FRAME_IDS[kind]:
                raise ValueError(f"{seq}: unexpected camera order in the calibration file "
                                 f"({cam['frame_id']} where {FRAME_IDS[kind]} was expected)")
        T = {i: np.array(calib[f"cam{i}"]["T_cn_cnm1"]) for i in range(1, 7)}
        T_rgb_lwir = T[4] @ T[3] @ T[2] @ T[1]
        T_to_lwir = {"rgb": T_rgb_lwir, "right": T[5] @ T_rgb_lwir, "event": T[6] @ T[5] @ T_rgb_lwir}
        res = {k: tuple(c["resolution"]) for k, c in cams.items()}
        K = {k: K_of(c) for k, c in cams.items()}

        # Common FoV in the thermal image, then the thermal crop projected into every camera.
        corners_lwir = image_corners(*res["lwir"])
        box_lwir = common_fov([project_corners(image_corners(*res[k]), K[k], K["lwir"], T_to_lwir[k],
                                               FOV_DEPTH_M) for k in ("rgb", "right", "event")],
                              *res["lwir"])
        boxes = {"lwir": box_lwir}
        for k in ("rgb", "right", "event"):
            proj = project_corners(corners_lwir, K["lwir"], K[k], np.linalg.inv(T_to_lwir[k]), FOV_DEPTH_M)
            boxes[k] = common_fov([proj], *res[k])
        size = (box_lwir[2] - box_lwir[0], box_lwir[3] - box_lwir[1])

        undist = {}
        for k in ("rgb", "right", "lwir"):
            new_K, roi = cv2.getOptimalNewCameraMatrix(K[k], dist_of(cams[k]), res[k], 0.5, res[k])
            maps = cv2.initUndistortRectifyMap(K[k], dist_of(cams[k]), np.eye(3), new_K, res[k],
                                               cv2.CV_32FC1)
            undist[k] = (maps, roi)
        event_maps = cv2.initUndistortRectifyMap(
            K["event"], np.asarray(cams["event"]["distortion_coeffs"], np.float64), None, K["event"],
            res["event"], cv2.CV_32FC1)
        return boxes, size, undist, event_maps

    @lru_cache(maxsize=32)
    def _stream(self, seq, kind):
        f = h5py.File(self.root / seq / f"{seq}_{kind}.h5", "r")
        if kind == "dvxplorer_left":
            return f, f["events/timestamps"][:]
        return f, {int(t): i for i, t in enumerate(f["image_raw/timestamps"][:])}

    def _image(self, seq, kind, cam, ts, undist):
        f, index = self._stream(seq, kind)
        img = f["image_raw/images"][index[ts]]
        if img.ndim == 2 and cam != "lwir":
            img = cv2.cvtColor(img, cv2.COLOR_BAYER_BG2RGB)
        (map1, map2), (x, y, w, h) = undist[cam]
        return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)[y:y + h, x:x + w]

    def _events(self, seq, ts, maps):
        f, t = self._stream(seq, "dvxplorer_left")
        idx = np.flatnonzero((t >= ts - 0.05e9) & (t <= ts))
        if idx.size < MIN_EVENTS:
            return None
        sl = slice(idx[0], idx[-1] + 1)
        x, y, p = (f[f"events/{k}"][sl].astype(np.int32)
                   for k in ("x_coordinates", "y_coordinates", "polarities"))
        frame, n = event_frame(x, y, p, t[sl], ts, E_W, E_H, window=0.05e9)
        if n < MIN_EVENTS or frame.sum() < 0.01 * E_W * E_H:
            return None
        out = np.zeros_like(frame)
        for c in range(frame.shape[2]):
            out[..., c] = cv2.remap(frame[..., c], *maps, cv2.INTER_LINEAR)
        return out

    def load(self, split, sample_id):
        seq, ts = sample_id[:6], int(sample_id[7:])
        boxes, size, undist, event_maps = self._calib(seq)
        rgb = self._image(seq, "rgb_left", "rgb", ts, undist)
        right = self._image(seq, "rgb_right", "right", ts, undist)
        lwir = self._image(seq, "adk_left", "lwir", ts, undist)
        depth = self.stereo(rgb[..., ::-1], right[..., ::-1])
        sample = {"rgb": crop_resize(rgb, boxes["rgb"], size),
                  "depth": crop_resize(depth, boxes["rgb"], size),
                  "lwir": crop_resize(lwir, boxes["lwir"], size)}
        event = self._events(seq, ts, event_maps)
        if event is not None:
            event = crop_resize(event, boxes["event"], size)
            if event.sum() > MIN_EVENTS:
                sample["event"] = event
        return sample
