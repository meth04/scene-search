#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import math
import numpy as np

MOTION_CLASSES = ("person", "bicycle", "car", "motorcycle", "bus", "truck")
CLASS_TO_SLOT = {n: i for i, n in enumerate(MOTION_CLASSES)}

FEATURE_NAMES = [
    *[f"class_{x}" for x in MOTION_CLASSES],
    "mean_cx", "mean_cy", "mean_w", "mean_h",
    "start_cx", "start_cy", "end_cx", "end_cy",
    "dx", "dy",
    "mean_speed", "max_speed",
    "mean_area", "area_ratio_end_start",
    "visible_ratio",
    "sin_direction", "cos_direction",
    "min_center_distance",
    "start_center_distance", "end_center_distance",
    "approach_center_delta",
    "crosses_vertical_center", "crosses_horizontal_center",
]
FEATURE_DIM = len(FEATURE_NAMES)


@dataclass
class Obs:
    frame_idx: int
    track_id: int
    class_name: str
    conf: float
    cx: float
    cy: float
    w: float
    h: float


def _crosses(a: float, b: float, center: float = 0.5) -> float:
    return float((a - center) * (b - center) <= 0 and abs(a - b) > 1e-6)


def track_to_feature(obs: Sequence[Obs], total_samples: int) -> np.ndarray:
    if not obs:
        raise ValueError("empty track")
    obs = sorted(obs, key=lambda x: x.frame_idx)
    first, last = obs[0], obs[-1]

    counts = {}
    for o in obs:
        counts[o.class_name] = counts.get(o.class_name, 0) + 1
    cls = max(counts, key=counts.get)

    out = np.zeros(FEATURE_DIM, dtype=np.float32)
    if cls in CLASS_TO_SLOT:
        out[CLASS_TO_SLOT[cls]] = 1.0

    cxs = np.asarray([o.cx for o in obs], np.float32)
    cys = np.asarray([o.cy for o in obs], np.float32)
    ws = np.asarray([o.w for o in obs], np.float32)
    hs = np.asarray([o.h for o in obs], np.float32)
    areas = ws * hs
    dxs = np.diff(cxs)
    dys = np.diff(cys)
    speeds = np.sqrt(dxs * dxs + dys * dys)
    center_d = np.sqrt((cxs - .5) ** 2 + (cys - .5) ** 2)

    dx = float(last.cx - first.cx)
    dy = float(last.cy - first.cy)
    ang = math.atan2(dy, dx) if abs(dx) + abs(dy) > 1e-8 else 0.0
    start_d = float(center_d[0])
    end_d = float(center_d[-1])

    vals = [
        float(cxs.mean()), float(cys.mean()), float(ws.mean()), float(hs.mean()),
        float(first.cx), float(first.cy), float(last.cx), float(last.cy),
        dx, dy,
        float(speeds.mean()) if len(speeds) else 0.0,
        float(speeds.max()) if len(speeds) else 0.0,
        float(areas.mean()),
        float(areas[-1] / max(float(areas[0]), 1e-6)),
        float(len({o.frame_idx for o in obs}) / max(total_samples, 1)),
        math.sin(ang), math.cos(ang),
        float(center_d.min()), start_d, end_d,
        start_d - end_d,
        _crosses(first.cx, last.cx),
        _crosses(first.cy, last.cy),
    ]
    out[len(MOTION_CLASSES):] = np.asarray(vals, np.float32)
    return out


def aggregate_window_tracks(
    observations: Sequence[Obs],
    total_samples: int,
    max_tracks: int = 24,
    min_track_obs: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    by_id: Dict[int, List[Obs]] = {}
    for o in observations:
        if o.class_name not in CLASS_TO_SLOT:
            continue
        by_id.setdefault(int(o.track_id), []).append(o)

    candidates = []
    for tid, seq in by_id.items():
        if len(seq) < min_track_obs:
            continue
        feat = track_to_feature(seq, total_samples)
        persistence = len({o.frame_idx for o in seq}) / max(total_samples, 1)
        conf = float(np.mean([o.conf for o in seq]))
        area = float(np.mean([o.w * o.h for o in seq]))
        score = 2.0 * persistence + conf + 0.5 * min(area * 10.0, 1.0)
        candidates.append((score, tid, feat))

    candidates.sort(reverse=True, key=lambda x: x[0])
    feats = np.zeros((max_tracks, FEATURE_DIM), dtype=np.float32)
    mask = np.zeros((max_tracks,), dtype=np.bool_)
    for i, (_, _, feat) in enumerate(candidates[:max_tracks]):
        feats[i] = feat
        mask[i] = True
    return feats, mask
