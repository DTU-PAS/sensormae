"""DSEC: one h5 file per sequence with RGB, events, stereo depth, DSEC-Det boxes and
DSEC-Semantic labels, all in the 640x480 view of the left event camera.

Sample i of a sequence follows DSEC-Det (sync="front", index i + 1):
    rgb           (3, 480, 640) uint8   rectified left image i + 1 remapped to the distorted
                                        event view (bicubic), BGR channel order
    event         (10, 480, 640) uint8  events between images i and i + 1, counts per
                                        polarity (channels 0-4: p=0, 5-9: p=1) and 10 ms bin,
                                        bin 0 the most recent
    depth         (480, 640) float32    FoundationStereo disparity of stereo pair i + 1,
                                        min-max normalized, in the rectified event view
    segmentation  (440, 640) uint8      DSEC-Semantic 11-class label i (labelled sequences)
    boxes/{x,y,w,h,class_id}            DSEC-Det tracks at image i (distorted event view)
The last sample of a sequence is left empty (zeros, no boxes).

Inputs (official DSEC releases, unzipped into one folder per split):

    <dsec>/<split>/<sequence>/images/timestamps.txt
    <dsec>/<split>/<sequence>/images/{left,right}/rectified/*.png       {split}_images.zip
    <dsec>/<split>/<sequence>/events/left/{events,rectify_map}.h5       {split}_events.zip
    <dsec>/<split>/<sequence>/calibration/cam_to_cam.yaml               {split}_calibration.zip
    <dsec>/<split>/<sequence>/object_detections/left/tracks.npy         {split}_object_detections.zip
    <semantic>/<split>/<sequence>/11classes/*.png                       {split}_semantic_segmentation.zip

DSEC-Det: https://github.com/uzh-rpg/dsec-det (scripts/download_dsec_extra.sh);
DSEC-Semantic: https://dsec.ifi.uzh.ch/dsec-semantic/. Depth uses the FoundationStereo
ViT-S ONNX export with a 480x640 input (foundation_stereo_s_640.onnx, see
preprocessing/README.md).

    python data_preparation/dsec.py --dsec /path/to/dsec --semantic /path/to/dsec_semantic \
        --stereo_model foundation_stereo_s_640.onnx --out data/dsec
"""

import argparse
import math
import os
from pathlib import Path

import cv2
import h5py
import hdf5plugin  # noqa: F401  (DSEC events are blosc compressed)
import numpy as np
import yaml
from tqdm import tqdm

WIDTH, HEIGHT = 640, 480
SEG_HEIGHT = 440
NUM_BINS = 5
BIN_WINDOW_US = 50_000
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def camera_matrix(entry):
    K = np.eye(3)
    K[[0, 1, 0, 1], [0, 1, 2, 2]] = entry["camera_matrix"]
    return K


def image_to_event_homography(calib):
    """Rectified image camera (camRect1) -> rectified event camera (camRect0)."""
    return (
        camera_matrix(calib["intrinsics"]["camRect0"])
        @ np.array(calib["extrinsics"]["R_rect0"])
        @ np.array(calib["extrinsics"]["T_10"])[:3, :3].T
        @ np.array(calib["extrinsics"]["R_rect1"]).T
        @ np.linalg.inv(camera_matrix(calib["intrinsics"]["camRect1"]))
    )


def remap_from(event_coords, homography):
    """cv2.remap maps sampling the rectified image at the given event-view coordinates."""
    h, w = event_coords.shape[:2]
    coords = np.concatenate((event_coords, np.ones((h, w, 1))), axis=-1)
    mapped = (np.linalg.inv(homography) @ coords[..., None]).squeeze()
    return (mapped[..., :2] / mapped[..., -1:]).astype("float32")


class StereoDisparity:
    def __init__(self, onnx_path, device):
        import torch  # noqa: F401  (load torch's CUDA libraries before onnxruntime's)
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.h, self.w = self.session.get_inputs()[0].shape[2:]

    def _prep(self, img):
        img = cv2.resize(np.ascontiguousarray(img), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        return ((img.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]

    def __call__(self, left_rgb, right_rgb):
        """Disparity at the input resolution, min-max normalized to [0, 1]."""
        h, w = left_rgb.shape[:2]
        disp = self.session.run(None, dict(zip(self.inputs, (self._prep(left_rgb), self._prep(right_rgb)))))[0]
        disp = disp[0, 0] if disp.ndim == 4 else disp[0]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        return (disp - disp.min()) / (disp.max() - disp.min())


def event_frame(events, ms_to_idx, t_offset, t_start, t_end):
    """Event counts of [t_start, t_end) (DSEC-Det millisecond index lookup)."""
    start = ms_to_idx[(t_start - t_offset) // 1000]
    end = ms_to_idx[np.clip(math.floor((t_end - t_offset) / 1000), 0, len(ms_to_idx) - 1)]
    x = events["x"][start:end].astype(np.int64)
    y = events["y"][start:end].astype(np.int64)
    p = events["p"][start:end].astype(np.int64)
    t = events["t"][start:end].astype(np.int64) + t_offset

    bins = np.clip((t_end - t) * NUM_BINS // BIN_WINDOW_US, 0, NUM_BINS - 1)
    frame = np.zeros((2 * NUM_BINS, HEIGHT, WIDTH), np.int64)
    np.add.at(frame, (p * NUM_BINS + bins, y, x), 1)
    return np.clip(frame, 0, 255).astype(np.uint8)


def track_ranges(track_t, image_t):
    """[start, end) rows of the tracks of every image (exact timestamp match, as DSEC-Det)."""
    stamps, counts = np.unique(track_t, return_counts=True)
    i, j = (stamps.reshape(-1, 1) == image_t.reshape(1, -1)).nonzero()
    deltas = np.zeros_like(image_t)
    deltas[j] = counts[i]
    idx = np.concatenate([np.array([0]), deltas]).cumsum()
    return np.stack([idx[:-1], idx[1:]], axis=-1).astype("uint64")


def process_sequence(seq_dir, semantic_dir, out_path, stereo):
    timestamps = np.genfromtxt(seq_dir / "images/timestamps.txt", dtype="int64")
    n = len(timestamps)
    with open(seq_dir / "calibration/cam_to_cam.yaml") as f:
        calib = yaml.safe_load(f)
    homography = image_to_event_homography(calib)
    with h5py.File(seq_dir / "events/left/rectify_map.h5", "r") as f:
        rectify_map = f["rectify_map"][()]
    gy, gx = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    to_distorted = remap_from(rectify_map, homography)
    to_rectified = remap_from(np.stack([gx, gy], axis=-1), homography)

    tracks = np.load(seq_dir / "object_detections/left/tracks.npy")
    ranges = track_ranges(tracks["t"], timestamps)
    left = sorted((seq_dir / "images/left/rectified").glob("*.png"))
    right = sorted((seq_dir / "images/right/rectified").glob("*.png"))
    labels = sorted((semantic_dir / "11classes").glob("*.png")) if semantic_dir and semantic_dir.is_dir() else []

    tmp_path = out_path.with_suffix(".h5.tmp")
    with h5py.File(seq_dir / "events/left/events.h5", "r") as ev_file, h5py.File(tmp_path, "w") as out:
        ms_to_idx = np.asarray(ev_file["ms_to_idx"], dtype="int64")
        t_offset = int(ev_file["t_offset"][()])
        events = ev_file["events"]

        rgb = out.create_dataset("rgb", (n, 3, HEIGHT, WIDTH), np.uint8)
        event = out.create_dataset("event", (n, 2 * NUM_BINS, HEIGHT, WIDTH), np.uint8)
        depth = out.create_dataset("depth", (n, HEIGHT, WIDTH), np.float32)
        segmentation = out.create_dataset("segmentation", (n, SEG_HEIGHT, WIDTH), np.uint8) if labels else None
        boxes = {k: [] for k in ("x", "y", "w", "h", "class_id")}

        for i in tqdm(range(n), desc=seq_dir.name, leave=False):
            if i == n - 1:
                for k in boxes:
                    boxes[k].append(np.zeros(0, np.int64 if k == "class_id" else np.float32))
                break
            j = i + 1
            image = cv2.imread(str(left[j]))
            rgb[i] = cv2.remap(image, to_distorted, None, interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)
            event[i] = event_frame(events, ms_to_idx, t_offset, timestamps[j - 1], timestamps[j])

            disparity = stereo(image[..., ::-1], cv2.imread(str(right[j]))[..., ::-1])
            depth[i] = cv2.remap(disparity, to_rectified, None, interpolation=cv2.INTER_LINEAR)

            if segmentation is not None:
                segmentation[i] = cv2.imread(str(labels[i]), cv2.IMREAD_UNCHANGED)

            t0, t1 = ranges[i]
            frame_tracks = tracks[int(t0):int(t1)]
            for k in ("x", "y", "w", "h"):
                boxes[k].append(frame_tracks[k].astype(np.float32))
            boxes["class_id"].append(frame_tracks["class_id"].astype(np.int64))

        group = out.create_group("boxes")
        for k, values in boxes.items():
            dtype = h5py.vlen_dtype(np.int64 if k == "class_id" else np.float32)
            ds = group.create_dataset(k, (n,), dtype=dtype)
            for i, v in enumerate(values):
                ds[i] = v
    os.replace(tmp_path, out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsec", required=True, type=Path, help="DSEC root with train/ and test/")
    parser.add_argument("--semantic", type=Path, default=None, help="DSEC-Semantic root with train/ and test/")
    parser.add_argument("--stereo_model", required=True, type=Path, help="foundation_stereo_s_640.onnx")
    parser.add_argument("--out", default="data/dsec", type=Path)
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--sequences", nargs="*", default=None, help="default: all")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    stereo = StereoDisparity(args.stereo_model, args.device)
    for split in args.splits:
        (args.out / split).mkdir(parents=True, exist_ok=True)
        sequences = sorted(p for p in (args.dsec / split).iterdir() if (p / "object_detections").is_dir())
        for seq_dir in sequences:
            if args.sequences and seq_dir.name not in args.sequences:
                continue
            out_path = args.out / split / f"{seq_dir.name}.h5"
            if out_path.exists():
                continue
            semantic_dir = args.semantic / split / seq_dir.name if args.semantic else None
            process_sequence(seq_dir, semantic_dir, out_path, stereo)


if __name__ == "__main__":
    main()
