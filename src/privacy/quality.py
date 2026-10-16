"""Cheap localization checks. No identity, OCR, extra models or image classifier."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def dimensions(box):
    return max(1e-6, box[2] - box[0]), max(1e-6, box[3] - box[1])


def iou(a, b):
    w, h = max(0, min(a[2], b[2]) - max(a[0], b[0])), max(0, min(a[3], b[3]) - max(a[1], b[1]))
    aw, ah = dimensions(a)
    bw, bh = dimensions(b)
    return w * h / max(1e-6, aw * ah + bw * bh - w * h)


def continuity(a, b, cfg, elapsed=1):
    aw, ah = dimensions(a)
    bw, bh = dimensions(b)
    center = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    motion = center / max(8.0, math.hypot(aw, ah))
    scale = max(aw / bw, bw / aw, ah / bh, bh / ah)
    aspect = max((aw / ah) / (bw / bh), (bw / bh) / (aw / ah))
    plausible = (scale <= cfg.consistency_scale and aspect <= cfg.consistency_aspect
                 and (iou(a, b) >= cfg.consistency_iou or motion <= cfg.consistency_motion * min(3, elapsed)))
    return plausible, max(0.0, 1 - motion / max(0.01, cfg.consistency_motion)) if plausible else 0.0


def geometry(detection, shape):
    """Soft quality only: even suspicious low observations receive a one-frame mask."""
    w, h = dimensions(detection.box)
    height, width = shape[:2]
    area, aspect = w * h / (width * height), w / h
    if detection.kind == "plate":
        score = 1.0
        if aspect < 0.45 or aspect > 12 or area > 0.20 or min(w, h) < 2:
            score = 0.15
        elif aspect < 1.15 or aspect > 8 or area > 0.08 or min(w, h) < 4:
            score = 0.50
        x1, y1, x2, y2 = detection.box
        if (x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1) and area > 0.08:
            score = min(score, 0.3)
        return score
    if aspect < 0.25 or aspect > 3.5:
        return 0.15
    if detection.landmarks is None:
        return 0.75  # Custom test detectors/legacy observations have no landmarks.
    points = np.asarray(detection.landmarks, dtype=float)
    if points.shape != (5, 2) or not np.isfinite(points).all():
        return 0.15
    normalized = (points - np.array(detection.box[:2])) / np.array([w, h])
    eye_a, eye_b, nose, mouth_a, mouth_b = normalized
    eyes_y, mouth_y = (eye_a[1] + eye_b[1]) / 2, (mouth_a[1] + mouth_b[1]) / 2
    # Loose bounds accommodate profile faces and imperfect low-resolution landmarks.
    if ((normalized < -0.5).any() or (normalized > 1.5).any()
            or np.linalg.norm(eye_a - eye_b) < 0.025
            or mouth_y < eyes_y - 0.20 or nose[1] < eyes_y - 0.35 or nose[1] > mouth_y + 0.35):
        return 0.15
    return 1.0


@dataclass
class Candidate:
    kind: str
    track_id: int
    box: tuple
    last_frame: int
    start_frame: int
    confidence: float
    geometry: float
    count: int = 1
    trusted: bool = False
    confirmed: bool = False
    stability: float = 1.0
    quality: float = 0.0
    ema: tuple | None = None
    origin: str = "legacy"
    parent_roi_id: int = -1
    detector: str = ""

    def score(self, high, age=0):
        self.quality = float(np.clip(
            0.35 * min(1, self.confidence / max(high, 0.01))
            + 0.25 * min(1, self.count / 3) + 0.15 * self.stability
            + 0.15 * self.geometry + 0.10 * min(1, (self.last_frame - self.start_frame + 1) / 3)
            - 0.06 * age, 0, 1))
        return self.quality
