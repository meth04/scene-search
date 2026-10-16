from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import ROOT
from .tracks import Detection, digest


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class PlateDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def require_weights(path):
    if not Path(path).is_file():
        raise RuntimeError(
            f"Missing privacy weights: {path}. Setup: .\\.venv\\Scripts\\python.exe "
            "-B scripts/prefetch_privacy_assets.py. No downloads occur during scan/search."
        )


def validate_plate_output(predictions):
    """NMS may silently discard NaNs; inference errors must never become an empty privacy cache."""
    import torch
    while isinstance(predictions, (tuple, list)):
        predictions = predictions[0]
    if isinstance(predictions, torch.Tensor) and not torch.isfinite(predictions).all().item():
        raise RuntimeError("Non-finite plate inference output; cache not committed. Use --privacy-fp32 or batch 1 and revalidate hardware.")


def checked_plate_predictor():
    from ultralytics.models.yolo.detect import DetectionPredictor
    class CheckedPlatePredictor(DetectionPredictor):
        def inference(self, image, *args, **kwargs):
            predictions = super().inference(image, *args, **kwargs)
            validate_plate_output(predictions)
            return predictions
    return CheckedPlatePredictor


class YuNetFaceDetector:
    def __init__(self, config):
        require_weights(config.face_weights)
        self.size = config.face_size
        self.input_size = (320, 320)
        self.resize_buffer = None
        self.profile = dict(resize_seconds=0., setup_seconds=0., inference_seconds=0., postprocess_seconds=0., calls=0)
        self.model = cv2.FaceDetectorYN.create(
            config.face_weights,
            "",
            (320, 320),
            config.face_score,
            config.face_nms,
            5000,
            config.face_backend,
            config.face_target,
        )
        self.descriptor = {
            "name": "OpenCV YuNet 2023mar",
            "version": cv2.__version__,
            "weights_sha256": digest(config.face_weights),
            "weights": config.face_weights,
        }

    def detect(self, frame):
        started = time.perf_counter()
        height, width = frame.shape[:2]
        scale = min(1.0, self.size / max(width, height))
        image = frame
        if scale < 1:
            target = (round(width * scale), round(height * scale))
            if self.resize_buffer is None or self.resize_buffer.shape[:2] != (target[1], target[0]):
                self.resize_buffer = np.empty((target[1], target[0], 3), dtype=frame.dtype)
            image = cv2.resize(frame, target, dst=self.resize_buffer)
        self.profile["resize_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        size = (image.shape[1], image.shape[0])
        if size != self.input_size:
            self.model.setInputSize(size)
            self.input_size = size
        self.profile["setup_seconds"] += time.perf_counter() - started
        started = time.perf_counter()
        _, faces = self.model.detect(image)
        self.profile["inference_seconds"] += time.perf_counter() - started
        self.profile["calls"] += 1
        if faces is None:
            return []
        started = time.perf_counter()
        output = [
            Detection("face", tuple(map(float, (x / scale, y / scale, (x + w) / scale, (y + h) / scale))), float(row[-1]),
                      tuple((float(row[i]) / scale, float(row[i + 1]) / scale) for i in range(4, 14, 2)),
                      origin="global_face", detector="YuNet 2023mar")
            for row in faces
            for x, y, w, h in [row[:4]]
        ]
        self.profile["postprocess_seconds"] += time.perf_counter() - started
        return output


class YoloPlateDetector:
    def __init__(self, config):
        require_weights(config.plate_weights)
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/model-cache/ultralytics-config"))
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.cfg import DEFAULT_CFG_DICT

        self.config = config
        self.profile = dict(preprocess_ms=0., inference_ms=0., postprocess_ms=0., calls=0)
        self.precision = (
            {"quantize": 16 if config.fp16 else None} if "quantize" in DEFAULT_CFG_DICT else {"half": config.fp16}
        )
        self.model = YOLO(config.plate_weights, task="detect")
        self.predictor = checked_plate_predictor()
        names = [str(n).lower().replace("-", "_").replace(" ", "_") for n in self.model.names.values()]
        if len(names) != 1 or names[0] not in {"license_plate", "licence_plate", "plate", "number_plate"}:
            raise ValueError(f"Privacy requires dedicated single-class plate weights; got {names}")
        self.descriptor = {
            "name": "YOLOv8 nano license_plate",
            "version": ultralytics.__version__,
            "weights": config.plate_weights,
            "weights_sha256": digest(config.plate_weights),
            "classes": names,
            "finite_output_guard": True,
        }

    def detect_batch(self, frames, *, imgsz=None):
        cfg = self.config
        results = self.model.predict(
            frames,
            predictor=self.predictor,
            device=cfg.plate_device,
            imgsz=imgsz or cfg.plate_imgsz,
            conf=cfg.plate_score,
            iou=cfg.plate_iou,
            **self.precision,
            max_det=1000,
            verbose=False,
            batch=cfg.batch_size,
        )
        for result in results:
            for key in ("preprocess", "inference", "postprocess"):
                self.profile[key + "_ms"] += result.speed[key]
            self.profile["calls"] += 1
        return [
            [Detection("plate", tuple(map(float, row[:4])), float(row[4]),
                       origin="global_plate", detector="YOLOv8n plate") for row in result.boxes.data.cpu().numpy()]
            for result in results
        ]

    def detect(self, frame):
        return self.detect_batch([frame])[0]

    def detect_roi(self, frame, imgsz):
        return self.detect_batch([frame], imgsz=imgsz)[0]
