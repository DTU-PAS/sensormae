"""KITTI 2D object detection: train / val split with RGB, LiDAR depth, labels and calibration.

Expects the training set of the official object detection release
(data_object_{image_2,label_2,calib,velodyne}.zip unzipped into one folder):

    <kitti>/training/{image_2,label_2,calib,velodyne}/<id>.{png,txt,txt,bin}

and writes, for the split lists in data_preparation/kitti_splits/ (3712 / 3769 frames):

    kitti/{train,val}/rgb/<id>.png       image_2
    kitti/{train,val}/labels/<id>.txt    label_2
    kitti/{train,val}/calibs/<id>.txt    calib
    kitti/{train,val}/lidar/<id>.bin     velodyne
    kitti/{train,val}/depth/<id>.png     LiDAR depth, uint8 (near = 255, 0 = no depth)

Depth: the LiDAR points are projected into image_2 and linearly interpolated over their
Delaunay triangulation, then inverted and scaled between the 1st and 99th depth
percentile of the frame.

    python data_preparation/kitti.py --kitti /path/to/kitti --out data/kitti
"""

import argparse
import os
import shutil

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

SPLITS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kitti_splits")


def read_kitti_calib(calib_path):
    """P2 (3, 4), R0_rect (4, 4) and Tr_velo_to_cam (4, 4) of a KITTI object calib file."""
    with open(calib_path) as f:
        raw = {k.strip(): np.array(v.split(), dtype=np.float64)
               for k, v in (line.split(":", 1) for line in f if ":" in line)}
    R0, Tr = np.eye(4), np.eye(4)
    R0[:3, :3] = raw["R0_rect"].reshape(3, 3)
    Tr[:3, :] = raw["Tr_velo_to_cam"].reshape(3, 4)
    return {"P2": raw["P2"].reshape(3, 4), "R0_rect": R0, "Tr_velo_to_cam": Tr}


def kitti_delaunay_depth(points, P2, R0_rect, Tr_velo_to_cam, image_size):
    """Dense metric depth (H, W) from velodyne points (N, 3), 0 outside the triangulation."""
    w, h = image_size
    cam = np.c_[points, np.ones(len(points))] @ (R0_rect @ Tr_velo_to_cam).T
    cam = cam[cam[:, 2] > 0.1]
    uvz = cam @ P2.T
    uv = uvz[:, :2] / uvz[:, 2:3]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    interp = LinearNDInterpolator(uv[keep], cam[keep, 2], fill_value=0.0)
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    return interp(gx, gy).astype(np.float32)


def encode_depth(depth, low_percentile=1.0, high_percentile=99.0):
    """uint8, near = 255, linear in depth between the frame's percentiles; 0 = no depth."""
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [low_percentile, high_percentile])
    scaled = 255 * (1 - np.clip((depth - lo) / (hi - lo), 0, 1))
    return np.where(valid, scaled, 0).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kitti", required=True, help="folder containing training/")
    parser.add_argument("--out", default="data/kitti")
    args = parser.parse_args()

    src = os.path.join(args.kitti, "training")
    for split in ("train", "val"):
        with open(os.path.join(SPLITS_DIR, f"{split}.txt")) as f:
            ids = [line.strip() for line in f if line.strip()]
        out = os.path.join(args.out, split)
        for sub in ("rgb", "labels", "calibs", "lidar", "depth"):
            os.makedirs(os.path.join(out, sub), exist_ok=True)

        for frame in tqdm(ids, desc=split):
            depth_path = os.path.join(out, "depth", f"{frame}.png")
            if os.path.exists(depth_path):
                continue
            for sub, src_sub, ext in (("rgb", "image_2", "png"), ("labels", "label_2", "txt"),
                                      ("calibs", "calib", "txt"), ("lidar", "velodyne", "bin")):
                shutil.copyfile(os.path.join(src, src_sub, f"{frame}.{ext}"), os.path.join(out, sub, f"{frame}.{ext}"))

            h, w = cv2.imread(os.path.join(src, "image_2", f"{frame}.png")).shape[:2]
            points = np.fromfile(os.path.join(src, "velodyne", f"{frame}.bin"), dtype=np.float32).reshape(-1, 4)[:, :3]
            depth = kitti_delaunay_depth(points, **read_kitti_calib(os.path.join(src, "calib", f"{frame}.txt")), image_size=(w, h))
            cv2.imwrite(depth_path, encode_depth(depth))

        shutil.copyfile(os.path.join(SPLITS_DIR, f"{split}.txt"), os.path.join(args.out, f"{split}.txt"))


if __name__ == "__main__":
    main()
