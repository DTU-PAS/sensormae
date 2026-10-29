"""SeeingThroughFog: LiDAR and radar projected onto the RGB image cropped to the gated
camera field of view ("RGB2Gatedv2": rows 202-970, columns 280-1560, 1280x768).

Every pixel hit by a point stores (y, z, intensity) for the LiDAR and (y, z, velocity) for
the radar, in camera coordinates, encoded as uint16 (value + 200) * 100; empty pixels hold
20000. Radar targets fill their whole image column (nearest target wins).

Depth: the LiDAR points (beyond 1.5 m) are projected into the RGB crop used by the loader (rows 230-950,
columns 290-1570, 1280x720) and linearly interpolated over their Delaunay triangulation
(as for KITTI); the metric depth is inverted and scaled to [0, 1] between the 1st and 99th
depth percentile of the frame (float32, 0 = no depth).

Uses the dataset tools of the official repository:

    git clone https://github.com/princeton-computational-imaging/SeeingThroughFog
    pip install pyquaternion

    python data_preparation/stf.py --root data/stf --stf_repo SeeingThroughFog

Writes stf/lidar_samples/yzi/lidar_hdl64_strongest/<sample>.png,
stf/radar_samples/yzv/<sample>.png and stf/depth_maps/<sample>.npy for every sample of
splits/all.txt.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm

LIDAR_SCALE = 100
SHIFT = 200
MIN_DISTANCE = 1.5  # LiDAR points closer than this (m) are dropped
MIN_X = 2.5  # points behind this plane (m, sensor x) are dropped
DEPTH_CROP = (230, 950, 290, 1570)  # RGB crop of the loader (y0, y1, x0, x1)


def project(points, vtc, sensor_to_camera, crop):
    """Image coordinates (in the crop) and camera-frame (y, z, value) of the points that
    fall inside the crop. `points` columns: x, y, z, value, ..."""
    vtc = crop.get_image_scaling() @ vtc
    width, height = crop.dsize

    points = points[:, 0:4]
    points = points[points[:, 0] > MIN_X]
    points_h = np.ones((points.shape[0], 5))
    points_h[:, 0:3] = points[:, 0:3]
    points_h[:, 4] = points[:, 3]
    sensor_to_camera_5 = np.zeros((5, 5))
    sensor_to_camera_5[0:4, 0:4] = sensor_to_camera
    sensor_to_camera_5[4, 4] = 1

    uv = vtc @ np.vstack((points[:, 0:3].T, np.ones((1, points.shape[0]))))
    uv[0] = uv[0] / uv[2]
    uv[1] = uv[1] / uv[2]
    uv = uv[0:2].T

    pts_3d = np.delete(sensor_to_camera_5 @ points_h.T, 3, axis=0)[1:, :]

    inside = (width > uv[:, 0]) & (uv[:, 0] >= 0) & (height > uv[:, 1]) & (uv[:, 1] >= 0)
    idx = np.where(inside)[0]
    return uv[idx, :], pts_3d[:, idx].T


def create_img(img_coordinates, pts_3d, crop, radar=False):
    width, height = crop.dsize
    img_coordinates = img_coordinates.astype(np.int32)
    image = LIDAR_SCALE * SHIFT * np.ones((width, height, 3)).astype(np.uint16)

    values = (pts_3d + SHIFT) * LIDAR_SCALE
    if not radar:
        image[img_coordinates[:, 0], img_coordinates[:, 1], :] = values
    else:
        for point, value in zip(img_coordinates, values):
            if image[point[0], point[1], 0] == LIDAR_SCALE * SHIFT or image[point[0], point[1], 0] > value[1]:
                image[point[0], :, 0] = value[0]
                image[point[0], :, 1] = value[1]
                image[point[0], :, 2] = value[2]

    return image.transpose([1, 0, 2]).squeeze()


def delaunay_depth(points, vtc, velodyne_to_camera):
    """Dense metric depth (720, 1280) in the depth crop, 0 outside the triangulation."""
    y0, y1, x0, x1 = DEPTH_CROP
    width, height = x1 - x0, y1 - y0
    xyz1 = np.c_[points[:, 0:3], np.ones(len(points))]
    z = (xyz1 @ velodyne_to_camera.T)[:, 2]
    uvw = xyz1 @ (np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]]) @ vtc).T
    front = z > 0.1
    uv, z = uvw[front, :2] / uvw[front, 2:3], z[front]
    keep = (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
    interp = LinearNDInterpolator(uv[keep], z[keep], fill_value=0.0)
    gx, gy = np.meshgrid(np.arange(width), np.arange(height))
    return interp(gx, gy).astype(np.float32)


def encode_depth(depth, low_percentile=1.0, high_percentile=99.0):
    """[0, 1], near = 1, linear in depth between the frame's percentiles; 0 = no depth."""
    valid = depth > 0
    lo, hi = np.percentile(depth[valid], [low_percentile, high_percentile])
    return np.where(valid, 1 - np.clip((depth - lo) / (hi - lo), 0, 1), 0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/stf")
    parser.add_argument("--stf_repo", default="SeeingThroughFog", help="clone of the official repository")
    parser.add_argument("--split", default="all.txt", help="sample list in <root>/splits")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.stf_repo))
    from tools.CreateTFRecords.generic_tf_tools.resize import resize
    from tools.DatasetViewer.lib.read import load_calib_data, load_radar_points, load_velodyne_scan

    velodyne_to_camera, _, P, R, vtc, radar_to_camera, _ = load_calib_data(
        os.path.join(args.stf_repo, "tools/DatasetViewer/calibs"),
        name_camera_calib="calib_cam_stereo_left.json",
        tf_tree="calib_tf_tree_full.json",
        velodyne_name="lidar_hdl64_s3_roof",
    )
    rtc = P @ R @ radar_to_camera
    crop = resize("RGB2Gatedv2")

    with open(os.path.join(args.root, "splits", args.split)) as f:
        samples = [line.replace(",", "_").split("\n")[0] for line in f.readlines()]

    lidar_dir = os.path.join(args.root, "lidar_samples/yzi/lidar_hdl64_strongest")
    radar_dir = os.path.join(args.root, "radar_samples/yzv")
    depth_dir = os.path.join(args.root, "depth_maps")
    for d in (lidar_dir, radar_dir, depth_dir):
        os.makedirs(d, exist_ok=True)

    for sample in tqdm(samples):
        lidar_file = os.path.join(lidar_dir, sample + ".png")
        radar_file = os.path.join(radar_dir, sample + ".png")
        depth_file = os.path.join(depth_dir, sample + ".npy")
        if all(os.path.exists(f) for f in (lidar_file, radar_file, depth_file)):
            continue

        lidar = load_velodyne_scan(os.path.join(args.root, "lidar_hdl64_strongest", sample + ".bin"))
        distance = np.sqrt(lidar[:, 0] ** 2 + lidar[:, 1] ** 2 + lidar[:, 2] ** 2)
        lidar = lidar[distance > MIN_DISTANCE]
        np.save(depth_file, encode_depth(delaunay_depth(lidar, vtc, velodyne_to_camera)))
        cv2.imwrite(lidar_file, create_img(*project(lidar, vtc, velodyne_to_camera, crop), crop))

        radar = load_radar_points(os.path.join(args.root, "radar_targets", sample + ".json"))
        if len(radar) == 0:
            width, height = crop.dsize
            image = LIDAR_SCALE * SHIFT * np.ones((height, width, 3)).astype(np.uint16)
        else:
            image = create_img(*project(radar, rtc, radar_to_camera, crop), crop, radar=True)
        cv2.imwrite(radar_file, image)


if __name__ == "__main__":
    main()
