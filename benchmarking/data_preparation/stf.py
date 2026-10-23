"""SeeingThroughFog: LiDAR and radar projected onto the RGB image cropped to the gated
camera field of view ("RGB2Gatedv2": rows 202-970, columns 280-1560, 1280x768).

Every pixel hit by a point stores (y, z, intensity) for the LiDAR and (y, z, velocity) for
the radar, in camera coordinates, encoded as uint16 (value + 200) * 100; empty pixels hold
20000. Radar targets fill their whole image column (nearest target wins).

Uses the dataset tools of the official repository:

    git clone https://github.com/princeton-computational-imaging/SeeingThroughFog
    pip install pyquaternion

    python data_preparation/stf.py --root data/stf --stf_repo SeeingThroughFog

Writes stf/lidar_samples/yzi/lidar_hdl64_strongest/<sample>.png and
stf/radar_samples/yzv/<sample>.png for every sample of splits/all.txt.
"""

import argparse
import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

LIDAR_SCALE = 100
SHIFT = 200
MIN_DISTANCE = 1.5  # LiDAR points closer than this (m) are dropped
MIN_X = 2.5  # points behind this plane (m, sensor x) are dropped


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
    os.makedirs(lidar_dir, exist_ok=True)
    os.makedirs(radar_dir, exist_ok=True)

    for sample in tqdm(samples):
        lidar_file = os.path.join(lidar_dir, sample + ".png")
        radar_file = os.path.join(radar_dir, sample + ".png")
        if os.path.exists(lidar_file) and os.path.exists(radar_file):
            continue

        lidar = load_velodyne_scan(os.path.join(args.root, "lidar_hdl64_strongest", sample + ".bin"))
        distance = np.sqrt(lidar[:, 0] ** 2 + lidar[:, 1] ** 2 + lidar[:, 2] ** 2)
        lidar = lidar[distance > MIN_DISTANCE]
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
