from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "data/model-cache/privacy"


def hardware():
    import torch

    if not torch.cuda.is_available():
        return {"cuda": False, "memory_gb": 0, "name": "CPU", "capability": [0, 0]}
    props = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "memory_gb": props.total_memory / 2**30,
        "name": props.name,
        "capability": [props.major, props.minor],
    }


@dataclass
class PrivacyConfig:
    profile: str = "auto"
    face_weights: str = str(ASSETS / "yunet.onnx")
    plate_weights: str = str(ASSETS / "plate-nano.pt")
    detect_every: int | None = None
    face_score: float = 0.45
    face_nms: float = 0.3
    face_size: int | None = None
    face_backend: int = 0
    face_target: int = 0
    plate_score: float = 0.12
    plate_iou: float = 0.5
    plate_imgsz: int | None = None
    plate_device: str | None = None
    fp16: bool = True
    batch_size: int | None = None
    tracking_score: float = 0.1
    face_track_grace_frames: int = 8
    plate_track_grace_frames: int = 6
    max_track_age: int = 15
    face_expansion: float = 0.25
    plate_expansion: float = 0.2
    ema_alpha: float = 0.8
    scene_cut_threshold: float = 0.32
    # Keep v1 defaults until the separately generated v2 caches pass human review.
    quality_mode: str = "v1"
    face_low_conf: float = 0.45
    face_high_conf: float = 0.68
    plate_low_conf: float = 0.12
    plate_high_conf: float = 0.40
    confirmation_frames: int = 2
    confirmation_window: int = 3
    consistency_iou: float = 0.10
    consistency_motion: float = 0.75
    consistency_scale: float = 2.0
    consistency_aspect: float = 1.8
    provisional_grace: int = 1
    confirmed_grace: int = 3
    provisional_max_age: int = 2
    confirmed_max_age: int = 6
    face_confirmed_expansion: float = 0.20
    plate_confirmed_expansion: float = 0.15
    stale_expansion_cap: float = 0.10
    stable_confirmation_frames: int = 6
    stable_quality_threshold: float = 0.72
    # Refinement is opt-in; the global safety pass remains mandatory.
    face_roi: bool = False
    plate_roi: bool = False
    small_object_rescue: bool = False
    head_fallback: bool = False
    overlap: bool = False
    object_weights: str = str(ROOT / "data/model-cache/yolo11n.pt")
    object_interval: int = 3
    max_face_rois_per_frame: int = 2
    max_plate_rois_per_frame: int = 2
    head_fraction: float = 0.40
    head_padding: float = 0.20
    vehicle_padding: float = 0.15
    face_roi_size: int = 320
    plate_roi_size: int = 640
    roi_face_score: float = 0.68
    roi_plate_score: float = 0.40
    cpu_threads: int = 4

    def resolved(self, capabilities=None):
        from dataclasses import replace

        info = hardware() if capabilities is None else capabilities
        profile = self.profile
        if profile not in {"auto", "cpu", "gtx1650", "rtx3090"}:
            raise ValueError("Unknown privacy profile")
        if profile == "auto":
            profile = "rtx3090" if info["cuda"] and info["memory_gb"] >= 16 else "gtx1650" if info["cuda"] else "cpu"
        device = self.plate_device or ("cuda:0" if profile != "cpu" and info["cuda"] else "cpu")
        if device.startswith("cuda") and not info["cuda"]:
            device = "cpu"
        cfg = replace(
            self,
            profile=profile,
            plate_device=device,
            face_size=self.face_size if self.face_size is not None else (1280 if profile == "rtx3090" else 960),
            batch_size=self.batch_size
            if self.batch_size is not None
            else (4 if profile == "rtx3090" and device != "cpu" else 1),
            # Recall-first default until a representative audit approves frame skipping.
            detect_every=self.detect_every if self.detect_every is not None else 1,
            plate_imgsz=self.plate_imgsz if self.plate_imgsz is not None else (1280 if profile == "rtx3090" else 960),
            fp16=self.fp16 and device != "cpu",
            face_score=self.face_low_conf if self.quality_mode == "v2" else self.face_score,
            plate_score=self.plate_low_conf if self.quality_mode == "v2" else self.plate_score,
        )
        cfg.validate()
        return cfg

    def validate(self):
        if self.face_roi or self.plate_roi or self.head_fallback or self.overlap or self.small_object_rescue:
            if self.quality_mode != "v2" or self.detect_every != 1:
                raise ValueError("Privacy refinement/overlap requires v2 and global detect-every=1")
        if min(self.object_interval, self.max_face_rois_per_frame, self.max_plate_rois_per_frame,
               self.face_roi_size, self.plate_roi_size) < 1:
            raise ValueError("Privacy ROI budgets, sizes and interval must be positive")
        if not 1 <= self.cpu_threads <= 16:
            raise ValueError("Privacy CPU threads must be between 1 and 16")
        if not (0.35 <= self.head_fraction <= 0.45 and 0.10 <= self.head_padding <= 0.25
                and 0.10 <= self.vehicle_padding <= 0.20):
            raise ValueError("Privacy head/vehicle crop geometry is outside supported bounds")
        if not (self.face_low_conf <= self.roi_face_score <= 1 and self.plate_low_conf <= self.roi_plate_score <= 1):
            raise ValueError("ROI confidence must be at least the corresponding v2 low threshold")
        if self.quality_mode not in {"v1", "v2"}:
            raise ValueError("Privacy quality mode must be v1 or v2")
        for low, high in ((self.face_low_conf, self.face_high_conf), (self.plate_low_conf, self.plate_high_conf)):
            if not (math.isfinite(low) and math.isfinite(high) and 0 <= low < high <= 1):
                raise ValueError("Privacy hysteresis requires 0 <= low < high <= 1")
        if self.confirmation_frames < 2 or self.confirmation_window < 1 or self.stable_confirmation_frames < self.confirmation_frames:
            raise ValueError("Temporal confirmation requires at least two observations and a positive window")
        if min(self.provisional_grace, self.confirmed_grace) < 0 or min(self.provisional_max_age, self.confirmed_max_age) < 1:
            raise ValueError("Adaptive persistence must be nonnegative with positive ages")
        if not (0 <= self.consistency_iou <= 1 and 0 < self.consistency_motion <= 5
                and self.consistency_scale > 1 and self.consistency_aspect > 1):
            raise ValueError("Invalid spatial consistency settings")
        for value in (self.detect_every, self.face_size, self.plate_imgsz, self.batch_size, self.max_track_age):
            if value is None or value <= 0:
                raise ValueError("Privacy sizes, interval, batch and maximum age must be positive")
        for value in (
            self.face_score,
            self.face_nms,
            self.plate_score,
            self.plate_iou,
            self.tracking_score,
            self.face_expansion,
            self.plate_expansion,
            self.scene_cut_threshold,
            self.ema_alpha,
            self.face_confirmed_expansion,
            self.plate_confirmed_expansion,
            self.stale_expansion_cap,
            self.stable_quality_threshold,
        ):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("Privacy thresholds/ratios must be finite and between zero and one")
        if min(self.face_track_grace_frames, self.plate_track_grace_frames) < 0:
            raise ValueError("Grace frames cannot be negative")
