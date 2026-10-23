"""Image, geometry and event helpers shared by the dataset modules."""

import cv2
import numpy as np
import yaml

# Event frames: all events in the 50 ms before the image timestamp, binned into
# 5 slices of 10 ms, separately for negative and positive polarity -> 10 channels.
EVENT_WINDOW_S = 0.05
EVENT_BINS = 5

# Fixed scene depth (m) used to project camera frustums when computing the common FoV.
FOV_DEPTH_M = 25.0


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def imread_rgb(path):
    """Read an image as RGB uint8."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def imread_raw(path):
    """Read an image unchanged (keeps 16-bit thermal and BGR channel order)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    return img


def imread_thermal(path):
    """Read a thermal image unchanged; 3-channel files are returned in RGB order.

    The thermal normalization keeps the first channel, so for 3-channel files the R
    channel is used.
    """
    img = imread_raw(path)
    if img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# ---------------------------------------------------------------------------
# Field-of-view alignment
# ---------------------------------------------------------------------------

def image_corners(width, height):
    """Homogeneous pixel corners (3, 4): TL, TR, BR, BL."""
    return np.array(
        [[0, 0, 1], [width - 1, 0, 1], [width - 1, height - 1, 1], [0, height - 1, 1]]
    ).T


def project_corners(corners, K_src, K_dst, T_src2dst, depth=FOV_DEPTH_M):
    """Project source image corners, back-projected at a fixed depth, into the target image."""
    pts = np.linalg.inv(K_src) @ corners * depth
    pts = T_src2dst[:3, :3] @ pts + T_src2dst[:3, 3:4]
    proj = K_dst @ pts
    return (proj[:2] / proj[2]).T


def common_fov(projections, width, height):
    """Intersection of the projected quads with the target image, as (x1, y1, x2, y2)."""
    x1, y1, x2, y2 = 0, 0, width - 1, height - 1
    for p in projections:
        x1, y1 = max(x1, p[:, 0].min()), max(y1, p[:, 1].min())
        x2, y2 = min(x2, p[:, 0].max()), min(y2, p[:, 1].max())
    return int(np.ceil(x1)), int(np.ceil(y1)), int(np.floor(x2)), int(np.floor(y2))


def fov_box(K_src, size_src, K_dst, size_dst, T_src2dst, depth=FOV_DEPTH_M):
    """Crop box in the target image covering the source camera FoV."""
    proj = project_corners(image_corners(*size_src), K_src, K_dst, T_src2dst, depth)
    return common_fov([proj], *size_dst)


def crop_resize(img, box, size):
    """Crop (x1, y1, x2, y2) and resize to size=(w, h) with area interpolation.

    float16 is resized in float32; images with more than 4 channels channel by channel.
    """
    x1, y1, x2, y2 = box
    crop = np.ascontiguousarray(img[y1:y2, x1:x2])
    if crop.dtype == np.float16:
        return crop_resize(crop.astype(np.float32), (0, 0, x2 - x1, y2 - y1), size).astype(np.float16)
    if crop.ndim == 3 and crop.shape[2] > 4:
        return np.stack(
            [cv2.resize(np.ascontiguousarray(crop[..., c]), size, interpolation=cv2.INTER_AREA)
             for c in range(crop.shape[2])], axis=-1)
    return cv2.resize(crop, size, interpolation=cv2.INTER_AREA)


def undistort(img, K, D, size, alpha=0.0):
    """Undistort with the optimal new camera matrix (same image size)."""
    K = np.asarray(K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).reshape(-1, 1)
    new_K, _ = cv2.getOptimalNewCameraMatrix(K, D, size, alpha, size)
    map1, map2 = cv2.initUndistortRectifyMap(K, D, None, new_K, size, cv2.CV_16SC2)
    return cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def remap_channels(img, map_x, map_y):
    """cv2.remap applied channel by channel (event frames have 10 channels)."""
    out = np.zeros_like(img)
    for c in range(img.shape[2]):
        out[..., c] = cv2.remap(img[..., c], map_x, map_y, interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------

def event_frame(x, y, p, t, t_end, width, height, window=EVENT_WINDOW_S, bins=EVENT_BINS,
                select=True):
    """Accumulate events with t in [t_end - window, t_end] into an (H, W, 2*bins) count image.

    Channels are ordered [p=0 bins..., p=1 bins...], counts are clipped to 255 (uint8).
    `t`, `t_end` and `window` must share the same unit. With select=False the events
    are assumed to be already restricted to the window. Returns (frame, n_events).
    """
    t0 = t_end - window
    if select:
        keep = (t >= t0) & (t <= t_end)
        x, y, p, t = (a[keep] for a in (x, y, p, t))
    b = np.clip(((t - t0) / (window / bins)).astype(int), 0, bins - 1)
    area = width * height
    idx = (p.astype(np.int64) * bins * area + b * area
           + y.astype(np.int64) * width + x.astype(np.int64))
    img = np.bincount(idx, minlength=2 * bins * area).reshape(2, bins, height, width)
    img = img.transpose(2, 3, 0, 1).reshape(height, width, 2 * bins)
    return np.clip(img, 0, 255).astype(np.uint8), len(t)


# ---------------------------------------------------------------------------
# Temporal matching
# ---------------------------------------------------------------------------

def nearest(sorted_ts, ts, max_diff):
    """Index of the timestamp closest to `ts`, or None if further than max_diff."""
    i = int(np.argmin(np.abs(sorted_ts - ts)))
    return i if abs(int(sorted_ts[i]) - int(ts)) <= max_diff else None
