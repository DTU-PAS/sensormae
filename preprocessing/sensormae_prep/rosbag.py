"""Minimal ROS1 bag decoding (no ROS install needed), built on the `rosbags` reader.

Only the message types used by the pipeline are parsed, directly from the serialized
bytes: sensor_msgs/Image, sensor_msgs/CompressedImage and dvs / dv_ros EventArray.
"""

import struct

import cv2
import numpy as np

EVENT_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("secs", "<u4"), ("nsecs", "<u4"), ("p", "u1")])
IMAGE_DTYPES = {"mono8": ("u1", 1), "8UC1": ("u1", 1), "mono16": ("<u2", 1), "16UC1": ("<u2", 1),
                "bgr8": ("u1", 3), "rgb8": ("u1", 3), "8UC3": ("u1", 3)}


def open_bag(path):
    from rosbags.rosbag1 import Reader
    return Reader(path)


def stamp(raw):
    """Header stamp (secs, nsecs) of a serialized message."""
    return struct.unpack_from("<II", raw, 4)


def _after_header(raw):
    (frame_len,) = struct.unpack_from("<I", raw, 12)
    return 16 + frame_len


def _string(raw, offset):
    (n,) = struct.unpack_from("<I", raw, offset)
    return raw[offset + 4:offset + 4 + n].decode(), offset + 4 + n


def compressed_image(raw):
    """sensor_msgs/CompressedImage -> decoded BGR image (as cv_bridge does)."""
    _, o = _string(raw, _after_header(raw))
    (n,) = struct.unpack_from("<I", raw, o)
    return cv2.imdecode(np.frombuffer(raw, np.uint8, n, o + 4), cv2.IMREAD_COLOR)


def image_encoding(raw):
    """Encoding string of a serialized sensor_msgs/Image (e.g. 'rgb8', 'mono16')."""
    encoding, _ = _string(raw, _after_header(raw) + 8)
    return encoding


def image(raw):
    """sensor_msgs/Image -> array in the message's own channel order ('passthrough')."""
    o = _after_header(raw)
    height, width = struct.unpack_from("<II", raw, o)
    encoding, o = _string(raw, o + 8)
    o += 1 + 4  # is_bigendian, step
    (n,) = struct.unpack_from("<I", raw, o)
    dtype, channels = IMAGE_DTYPES[encoding]
    img = np.frombuffer(raw, dtype, n // np.dtype(dtype).itemsize, o + 4)
    return img.reshape(height, width, channels).squeeze(-1) if channels == 1 else \
        img.reshape(height, width, channels)


def events(raw):
    """EventArray -> structured array with x, y, secs, nsecs, p."""
    o = _after_header(raw) + 8  # height, width
    (n,) = struct.unpack_from("<I", raw, o)
    return np.frombuffer(raw, EVENT_DTYPE, n, o + 4)


def time_name(secs, nsecs, shift_s=0.0):
    """Frame name f"{(stamp + shift).to_sec():.9f}" with '.' replaced by '_'
    (rospy Time/Duration arithmetic, float seconds formatting)."""
    shift_secs = int(shift_s)
    shift_nsecs = int((shift_s - shift_secs) * 1e9)
    total = (secs + shift_secs) * 10**9 + nsecs + shift_nsecs
    s, ns = divmod(total, 10**9)
    return f"{float(s) + float(ns) / 1e9:.9f}".replace(".", "_")
