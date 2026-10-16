from __future__ import annotations

import importlib.util
import inspect
import os
import time
from types import SimpleNamespace

import numpy as np

from .config import ROOT
from .tracks import TrackBox


def expand_box(box, ratio, width, height):
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio / 2, (y2 - y1) * ratio / 2
    return (max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(width), x2 + dx), min(float(height), y2 + dy))


def union_box(a, b):
    return tuple([min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])])


class PrivacyTracker:
    """ByteTrack association, plus privacy persistence and immediate unmatched-box coverage.

    Empty updates advance the Kalman prediction by one VIDEO frame. Lost tracks are
    emitted for bounded grace, including on detector misses. Never use ReID.
    """

    def __init__(self, config, fps):
        if importlib.util.find_spec("lap") is None or importlib.util.find_spec("ultralytics") is None:
            raise RuntimeError(
                "Privacy tracking requires ultralytics and lap. Install requirements_laptop_hybrid.txt; no automatic installation is allowed."
            )
        os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "data/model-cache/ultralytics-config"))
        from ultralytics.engine.results import Boxes
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.boxes_class = Boxes
        args = SimpleNamespace(
            track_high_thresh=config.tracking_score,
            track_low_thresh=0.01,
            new_track_thresh=config.tracking_score,
            track_buffer=config.max_track_age,
            match_thresh=0.85,
            fuse_score=False,
        )
        options = {"frame_rate": 30} if "frame_rate" in inspect.signature(BYTETracker).parameters else {}
        self.backends = {kind: BYTETracker(args, **options) for kind in ("face", "plate")}
        self.config, self.fps = config, fps
        self.last = {}
        self.smooth = {}
        self.next_direct = -1
        self.direct = {}
        self.candidates = {}
        self.backend_keys = {}
        self.force_refresh = False
        self.events = []
        self.validation_seconds = 0.0

    def reset(self):
        # Do not reset ByteTrack's global ID counter: IDs must remain unique across cuts.
        for backend in self.backends.values():
            backend.tracked_stracks = []
            backend.lost_stracks = []
            backend.removed_stracks = []
        self.last.clear()
        self.smooth.clear()
        self.direct.clear()
        self.candidates.clear()
        self.backend_keys.clear()
        self.force_refresh = False

    def refresh_needed(self, frame_index):
        return self.force_refresh or any(frame_index - last >= max(2, self.config.detect_every) for last in self.last.values())

    def update(self, frame_index, detections, shape):
        if self.config.quality_mode == "v2":
            return self.update_quality(frame_index, detections, shape)
        height, width = shape[:2]
        output = []
        for kind, backend in self.backends.items():
            group = [d for d in detections if d.kind == kind]
            data = np.array([[*d.box, d.confidence, 0] for d in group], dtype=np.float32).reshape(-1, 6)
            rows = backend.update(self.boxes_class(data, (height, width)))
            matched = set()
            observed = {}
            for row in rows:
                tid, idx = int(row[4]), int(row[-1])
                if 0 <= idx < len(group):
                    matched.add(idx)
                    observed[tid] = group[idx].box
                    self.last[kind, tid] = frame_index
            for track in [*backend.tracked_stracks, *backend.lost_stracks]:
                key = kind, track.track_id
                if key not in self.last:
                    continue
                age = frame_index - self.last[key]
                grace = self.config.face_track_grace_frames if kind == "face" else self.config.plate_track_grace_frames
                if age > min(grace, self.config.max_track_age):
                    continue
                raw = tuple(map(float, track.xyxy if hasattr(track, "xyxy") else track.tlbr))
                if track.track_id in observed:
                    # A filtered estimate may lag a moving detection. Every
                    # observed detector ROI must be included without relying
                    # on expansion to make up the difference.
                    raw = union_box(raw, observed[track.track_id])
                prior = self.smooth.get(key, raw)
                ema = tuple(self.config.ema_alpha * a + (1 - self.config.ema_alpha) * b for a, b in zip(raw, prior))
                self.smooth[key] = ema
                ratio = self.config.face_expansion if kind == "face" else self.config.plate_expansion
                box = expand_box(union_box(raw, ema), ratio + min(0.3, age * 0.025), width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    output.append(
                        TrackBox(
                            frame_index,
                            frame_index / self.fps,
                            kind,
                            track.track_id,
                            box,
                            float(track.score),
                            "detector" if age == 0 else "tracker",
                        )
                    )
            # ByteTrack can delay confirmation or reject low-confidence first detections.
            # Every detector box is anonymized on the very frame it is observed.
            for idx, d in enumerate(group):
                if idx not in matched:
                    ratio = self.config.face_expansion if kind == "face" else self.config.plate_expansion
                    box = expand_box(d.box, ratio, width, height)
                    if box[2] > box[0] and box[3] > box[1]:
                        output.append(
                            TrackBox(
                                frame_index,
                                frame_index / self.fps,
                                kind,
                                self.next_direct,
                                box,
                                d.confidence,
                                "detector",
                            )
                        )
                        self.direct[kind, self.next_direct] = output[-1]
                        self.next_direct -= 1
            grace = self.config.face_track_grace_frames if kind == "face" else self.config.plate_track_grace_frames
            for key, prior in list(self.direct.items()):
                if key[0] != kind:
                    continue
                age = frame_index - prior.frame_index
                if age > min(grace, self.config.max_track_age):
                    del self.direct[key]
                elif age > 0:
                    output.append(
                        TrackBox(
                            frame_index,
                            frame_index / self.fps,
                            kind,
                            prior.track_id,
                            prior.box,
                            prior.confidence,
                            "tracker",
                        )
                    )
        expired = [key for key, last in self.last.items() if frame_index - last > self.config.max_track_age]
        for key in expired:
            self.last.pop(key, None)
            self.smooth.pop(key, None)
        return output

    def update_quality(self, frame_index, detections, shape):
        from .quality import Candidate, continuity, dimensions, geometry, iou

        cfg = self.config
        height, width = shape[:2]
        output = []
        self.force_refresh, self.events = False, []
        for kind, backend in self.backends.items():
            low = cfg.face_low_conf if kind == "face" else cfg.plate_low_conf
            high = cfg.face_high_conf if kind == "face" else cfg.plate_high_conf
            started = time.perf_counter()
            group = [d for d in detections if d.kind == kind and d.confidence >= low
                     and np.isfinite(d.box).all() and d.box[2] > d.box[0] and d.box[3] > d.box[1]]
            qualities = [geometry(d, shape) for d in group]
            self.validation_seconds += time.perf_counter() - started
            data = np.array([[*d.box, d.confidence, 0] for d in group], dtype=np.float32).reshape(-1, 6)
            rows = backend.update(self.boxes_class(data, (height, width)))
            matched = {int(row[-1]): int(row[4]) for row in rows if 0 <= int(row[-1]) < len(group)}
            predictions = {t.track_id: tuple(map(float, t.xyxy if hasattr(t, "xyxy") else t.tlbr))
                           for t in [*backend.tracked_stracks, *backend.lost_stracks]}
            started = time.perf_counter()
            used = set()
            for index, d in enumerate(group):
                backend_id = matched.get(index)
                key = self.backend_keys.get((kind, backend_id))
                candidate = self.candidates.get(key) if key not in used else None
                if candidate and not continuity(candidate.box, d.box, cfg, frame_index - candidate.last_frame)[0]:
                    self.events.append("detector_track_jump")
                    candidate = None
                if candidate is None:
                    nearby = [(iou(c.box, d.box), k, c) for k, c in self.candidates.items()
                              if k[0] == kind and k not in used
                              and 0 < frame_index - c.last_frame <= cfg.confirmation_window * cfg.detect_every
                              and continuity(c.box, d.box, cfg, frame_index - c.last_frame)[0]]
                    if nearby:
                        _, key, candidate = max(nearby, key=lambda item: item[0])
                if candidate is None:
                    key = kind, self.next_direct
                    self.next_direct -= 1
                    candidate = Candidate(kind, key[1], tuple(d.box), frame_index, frame_index, d.confidence, qualities[index])
                    self.candidates[key] = candidate
                else:
                    gap = frame_index - candidate.last_frame
                    if gap > cfg.confirmation_window * cfg.detect_every and not (candidate.trusted or candidate.confirmed):
                        candidate.count = 0
                    candidate.stability = continuity(candidate.box, d.box, cfg, gap)[1]
                    candidate.count += 1
                    candidate.box = tuple(d.box)
                    candidate.last_frame, candidate.confidence, candidate.geometry = frame_index, d.confidence, qualities[index]
                used.add(key)
                candidate.origin, candidate.parent_roi_id, candidate.detector = d.origin, d.parent_roi_id, d.detector
                if backend_id is not None:
                    self.backend_keys[kind, backend_id] = key
                # A zoomed crop can be confidently wrong on one frame. Preserve
                # v2 global hysteresis; ROI-only evidence earns long trusted grace
                # after spatially consistent temporal confirmation, never one hit.
                candidate.trusted |= d.origin != "head_fallback" and d.confidence >= high and (
                    d.origin not in {"roi_face", "roi_plate"} or candidate.count >= cfg.confirmation_frames)
                quality = candidate.score(high)
                # Severe face geometry never earns persistence from repetition alone.
                # Distorted plates may confirm with an extra consistent observation.
                required = cfg.confirmation_frames if candidate.geometry >= 0.55 else cfg.confirmation_frames + 1
                promotable = candidate.geometry >= 0.55 or (kind == "plate" and candidate.geometry >= 0.3)
                candidate.confirmed |= candidate.count >= required and promotable and quality >= 0.55
                state = ("TRUSTED" if candidate.trusted else "CONFIRMED" if candidate.confirmed
                         else "SUSPICIOUS" if candidate.geometry < 0.3 else "PROVISIONAL")
                if state == "SUSPICIOUS":
                    self.events.append("suspicious_geometry")
                raw = candidate.box
                # Detector ROI is authoritative; prevent Kalman/old EMA union ballooning.
                prior = candidate.ema or raw
                ema = tuple(cfg.ema_alpha * a + (1 - cfg.ema_alpha) * b for a, b in zip(raw, prior))
                combined = union_box(raw, ema)
                rw, rh = dimensions(raw)
                cw, ch = dimensions(combined)
                candidate.ema = ema
                if cw * ch > 1.35 * rw * rh:
                    combined = raw
                ratio = (cfg.face_expansion if kind == "face" else cfg.plate_expansion) if candidate.trusted else (
                    cfg.face_confirmed_expansion if kind == "face" else cfg.plate_confirmed_expansion)
                box = expand_box(combined, ratio, width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    output.append(TrackBox(frame_index, frame_index / self.fps, kind, candidate.track_id, box,
                                           d.confidence, "detector", state, candidate.count, quality, candidate.geometry, 0,
                                           d.origin, d.parent_roi_id, d.detector))
                self.last[key] = frame_index
            reverse = {key: tid for (k, tid), key in self.backend_keys.items() if k == kind}
            for key, candidate in list(self.candidates.items()):
                if key[0] != kind or key in used:
                    continue
                age = frame_index - candidate.last_frame
                # Long, geometrically plausible lower-confidence observations
                # are stronger evidence than two hits: preserve occlusion recall.
                stable = (candidate.confirmed and candidate.count >= cfg.stable_confirmation_frames
                          and candidate.geometry >= .75 and candidate.stability >= .70
                          and candidate.score(high) >= cfg.stable_quality_threshold)
                quality = candidate.score(high, age)
                persistent = candidate.trusted or stable
                grace = (cfg.face_track_grace_frames if kind == "face" else cfg.plate_track_grace_frames) if persistent else (
                    cfg.confirmed_grace if candidate.confirmed else cfg.provisional_grace if candidate.geometry >= 0.3 else 0)
                max_age = cfg.max_track_age if persistent else cfg.confirmed_max_age if candidate.confirmed else cfg.provisional_max_age
                if age > max_age:
                    del self.candidates[key]
                    self.last.pop(key, None)
                    continue
                if age > min(grace, max_age) or quality < 0.10:
                    continue
                prediction = predictions.get(reverse.get(key), candidate.box)
                valid, _ = continuity(candidate.box, prediction, cfg, 1)
                if not valid:
                    self.events.append("tracker_drift")
                    self.force_refresh = True
                    if not candidate.trusted:
                        del self.candidates[key]
                        self.last.pop(key, None)
                        continue
                    prediction = candidate.box  # Trusted: freeze safe last ROI and request fresh detection.
                ratio = (cfg.face_expansion if kind == "face" else cfg.plate_expansion) if candidate.trusted else (
                    cfg.face_confirmed_expansion if kind == "face" else cfg.plate_confirmed_expansion)
                extra = min(cfg.stale_expansion_cap, age * 0.015) if candidate.trusted and quality >= 0.35 else 0
                box = expand_box(prediction, ratio + extra, width, height)
                if box[2] > box[0] and box[3] > box[1]:
                    state = "STALE" if age > cfg.detect_every else "TRACKER_ONLY"
                    output.append(TrackBox(frame_index, frame_index / self.fps, kind, candidate.track_id, box,
                                           candidate.confidence, "tracker", state, candidate.count, quality, candidate.geometry, age,
                                           candidate.origin, candidate.parent_roi_id, candidate.detector))
            self.validation_seconds += time.perf_counter() - started
        self.backend_keys = {key: value for key, value in self.backend_keys.items() if value in self.candidates}
        return output
