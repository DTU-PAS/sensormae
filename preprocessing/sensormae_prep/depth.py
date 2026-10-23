"""Stereo depth with FoundationStereo (ViT-S checkpoint) exported to ONNX.

Each dataset uses an export whose input size is roughly half its stereo resolution
(see README, "FoundationStereo ONNX models"). Both images are resized to the model input,
the predicted disparity is resized back with linear interpolation and min-max normalized
to [0, 1] per frame, then stored as float16.
"""

import os
from pathlib import Path

import cv2
import numpy as np

os.environ.setdefault("ORT_LOGGING_LEVEL", "ERROR")

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class StereoDepth:
    def __init__(self, onnx_path, device="cuda"):
        import torch  # noqa: F401  (load torch's CUDA libraries before onnxruntime's)
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        onnx_path = Path(onnx_path)
        if not onnx_path.exists():
            raise FileNotFoundError(f"FoundationStereo model not found: {onnx_path}")
        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers.insert(0, "CUDAExecutionProvider")
        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.inputs = [i.name for i in self.session.get_inputs()]
        self.h, self.w = self.session.get_inputs()[0].shape[2:]

    def _prep(self, img):
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        img = cv2.resize(np.ascontiguousarray(img), (self.w, self.h), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(np.float32) / 255.0 - MEAN) / STD
        return img.transpose(2, 0, 1)[None]

    def __call__(self, left_rgb, right_rgb):
        """Normalized disparity (H, W) float16 for an RGB uint8 stereo pair."""
        h, w = left_rgb.shape[:2]
        feed = dict(zip(self.inputs, (self._prep(left_rgb), self._prep(right_rgb))))
        disp = self.session.run(None, feed)[0]
        disp = disp[0, 0] if disp.ndim == 4 else disp[0]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_LINEAR)
        lo, hi = disp.min(), disp.max()
        disp = (disp - lo) / (hi - lo) if hi > lo else np.zeros_like(disp)
        return disp.astype(np.float16)


class DepthModels:
    """Loads the StereoDepth of an ONNX file in `model_dir`, keeping one model in memory."""

    def __init__(self, model_dir, device="cuda"):
        self.model_dir, self.device, self.name, self.model = Path(model_dir), device, None, None

    def __getitem__(self, name):
        if name != self.name:
            self.model = None  # release the previous session first
            self.model, self.name = StereoDepth(self.model_dir / name, self.device), name
        return self.model
