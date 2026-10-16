"""Privacy-only coarse guidance. Retrieval motion summaries are never used as boxes."""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import cv2
import numpy as np

from .quality import continuity, geometry, iou
from .tracker import expand_box, union_box
from .tracks import Detection, digest


@dataclass(frozen=True)
class ObjectROI:
    box: tuple
    confidence: float
    class_id: int
    roi_id: int
    observations: int = 1
    stable: bool = False
    age: int = 0
    motion: float = 0.0
    velocity: tuple = (0., 0., 0., 0.)


def crop_box(roi, shape, cfg, kind):
    x1, y1, x2, y2 = roi.box
    if kind == "face":
        y2 = y1 + (y2 - y1) * cfg.head_fraction
    box = expand_box((x1, y1, x2, y2), cfg.head_padding if kind == "face" else cfg.vehicle_padding,
                     shape[1], shape[0])
    return tuple(int(math.floor(v) if i < 2 else math.ceil(v)) for i, v in enumerate(box))


def remap(detection, crop, scaled_shape, original_shape, roi_id):
    sx, sy = (crop[2] - crop[0]) / scaled_shape[1], (crop[3] - crop[1]) / scaled_shape[0]
    def point(x, y):
        return (float(np.clip(x * sx + crop[0], 0, original_shape[1])),
                float(np.clip(y * sy + crop[1], 0, original_shape[0])))
    a, b = point(*detection.box[:2]), point(*detection.box[2:])
    landmarks = tuple(point(*p) for p in detection.landmarks) if detection.landmarks else None
    return replace(detection, box=(*a, *b), landmarks=landmarks, origin=f"roi_{detection.kind}", parent_roi_id=roi_id)


def fuse_detections(detections):
    """Merge same-kind duplicates without shrinking either observed region."""
    ordered = sorted(detections, key=lambda d: (-d.confidence, d.origin, d.parent_roi_id, d.box))
    merged = []
    for d in ordered:
        while True:
            index = next((i for i, prior in enumerate(merged) if duplicates(d, prior)), None)
            if index is None:
                break
            prior = merged.pop(index)
            winner = prior if prior.confidence >= d.confidence else d
            fallback = d.origin == "head_fallback" or prior.origin == "head_fallback"
            d = replace(winner, box=union_box(prior.box, d.box),
                        origin="head_fallback" if fallback else winner.origin,
                        detector="YOLO11n head + face evidence" if fallback else winner.detector)
        merged.append(d)
    return merged


def duplicates(a, b):
    if a.kind != b.kind:
        return False
    if iou(a.box, b.box) >= .45:
        return True
    # A guarded head prior intentionally encloses a more precise face detection.
    # Collapse these into one mask, retaining their entire observed union.
    if a.origin == "head_fallback" or b.origin == "head_fallback":
        head, face = (a, b) if a.origin == "head_fallback" else (b, a)
        cx, cy = (face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2
        intersection = max(0, min(head.box[2], face.box[2]) - max(head.box[0], face.box[0])) * max(
            0, min(head.box[3], face.box[3]) - max(head.box[1], face.box[1]))
        area = (face.box[2] - face.box[0]) * (face.box[3] - face.box[1])
        return head.box[0] <= cx <= head.box[2] and head.box[1] <= cy <= head.box[3] and intersection >= .8 * area
    return False


def head_fallback(roi, faces, shape):
    x1, y1, x2, y2 = roi.box
    w, h = x2 - x1, y2 - y1
    if not (roi.class_id == 0 and roi.stable and roi.observations >= 2 and roi.confidence >= .55
            and w >= 14 and h >= 60 and 1.6 <= h / w <= 6):
        return None
    head = expand_box((x1 + .10 * w, y1, x2 - .10 * w, y1 + .25 * h), .15, shape[1], shape[0])
    area = (head[2] - head[0]) * (head[3] - head[1])
    if area > .006 * shape[0] * shape[1] or head[3] - head[1] > 90:
        return None
    # A partial face box may still leave the visible head exposed.
    for d in faces:
        intersection = max(0, min(head[2], d.box[2]) - max(head[0], d.box[0])) * max(
            0, min(head[3], d.box[3]) - max(head[1], d.box[1]))
        if intersection >= .9 * area:
            return None
    return Detection("face", head, .55, origin="head_fallback", parent_roi_id=roi.roi_id,
                     detector="YOLO11n guarded head prior")


class ObjectGuide:
    """Bounded causal ROI hints, never a gate for global face/plate inference."""
    def __init__(self, cfg):
        from ultralytics import YOLO

        from .detectors import require_weights
        require_weights(cfg.object_weights)
        self.cfg = cfg
        self.model = YOLO(cfg.object_weights, task="detect")
        expected = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
        if any(self.model.names.get(i) != name for i, name in expected.items()):
            raise ValueError("Privacy ROI guidance requires COCO person/vehicle class IDs")
        from ultralytics.cfg import DEFAULT_CFG_DICT
        self.precision = {"quantize": 16 if cfg.fp16 else None} if "quantize" in DEFAULT_CFG_DICT else {"half": cfg.fp16}
        self.descriptor = {"name": "YOLO11n privacy-only coarse guide", "weights": cfg.object_weights,
                           "weights_sha256": digest(cfg.object_weights), "classes": expected, "imgsz": 640,
                           "confidence": .35, "interval": cfg.object_interval, "persisted_retrieval_rois": False}
        self.previous = []
        self.last_frame = -1
        self.next_id = 1
        self.previous_thumbnail = None
        self.calls = 0

    def reset(self):
        self.previous = []
        self.last_frame = -1
        self.previous_thumbnail = None

    def detect(self, frame, index):
        small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
        cut = self.previous_thumbnail is not None and np.mean(np.abs(
            small.astype(float) - self.previous_thumbnail.astype(float))) / 255 > self.cfg.scene_cut_threshold
        self.previous_thumbnail = small
        if cut:
            self.previous = []
            self.last_frame = -1
        if self.last_frame >= 0 and index - self.last_frame < self.cfg.object_interval:
            age = index - self.last_frame
            projected = []
            for r in self.previous:
                box = tuple(float(np.clip(v + delta * age, 0, frame.shape[1] if i % 2 == 0 else frame.shape[0]))
                            for i, (v, delta) in enumerate(zip(r.box, r.velocity)))
                valid = box[2] > box[0] and box[3] > box[1] and continuity(r.box, box, self.cfg, age)[0]
                projected.append(replace(r, box=box if valid else r.box, age=age, stable=r.stable and valid))
            return projected
        result = self.model.predict(frame, imgsz=640, conf=.35, iou=.5, classes=[0, 2, 3, 5, 7],
                                    device=self.cfg.plate_device, **self.precision,
                                    verbose=False)[0]
        self.calls += 1
        current, used = [], set()
        for row in result.boxes.data.cpu().numpy():
            box, confidence, class_id = tuple(map(float, row[:4])), float(row[4]), int(row[5])
            matches = [(iou(p.box, box), p) for p in self.previous if p.class_id == class_id
                       and p.roi_id not in used and continuity(p.box, box, self.cfg, self.cfg.object_interval)[0]]
            prior = max(matches, key=lambda x: x[0])[1] if matches else None
            if prior is not None:
                stable, stability = continuity(prior.box, box, self.cfg, self.cfg.object_interval)
                elapsed = max(1, index - self.last_frame)
                bound = .25 * max(box[2] - box[0], box[3] - box[1])
                velocity = tuple(float(np.clip((a - b) / elapsed, -bound, bound)) for a, b in zip(box, prior.box))
                r = ObjectROI(box, confidence, class_id, prior.roi_id, prior.observations + 1,
                              stable and stability >= .5, motion=1 - stability,
                              velocity=velocity if stable and stability >= .5 else (0., 0., 0., 0.))
                used.add(prior.roi_id)
            else:
                r = ObjectROI(box, confidence, class_id, self.next_id)
                self.next_id += 1
            current.append(r)
        self.previous, self.last_frame = current, index
        return current


class Refinement:
    def __init__(self, cfg):
        self.cfg = cfg
        self.face_calls = self.plate_calls = self.head_masks = 0
        self.face_evidence, self.plate_evidence = [], []
        self.plate_frame = 0
        self.plate_history = {}
        self.plate_failures = {}

    def reset(self):
        self.face_evidence.clear()
        self.plate_evidence.clear()
        self.plate_history.clear()
        self.plate_failures.clear()
        self.plate_frame = 0

    def select(self, rois, global_detections, kind, shape):
        cfg = self.cfg
        candidates = []
        for roi in rois:
            if (kind == "face" and roi.class_id != 0) or (kind == "plate" and roi.class_id not in {2, 3, 5, 7}):
                continue
            crop = crop_box(roi, shape, cfg, kind)
            w, h = crop[2] - crop[0], crop[3] - crop[1]
            if min(w, h) < (12 if kind == "face" else 24):
                continue
            high = cfg.face_high_conf if kind == "face" else cfg.plate_high_conf
            resolved = any(d.confidence >= high and crop[0] <= (d.box[0] + d.box[2]) / 2 <= crop[2]
                           and crop[1] <= (d.box[1] + d.box[3]) / 2 <= crop[3] for d in global_detections)
            if cfg.small_object_rescue and resolved and roi.observations > 1 and roi.motion < .2:
                continue
            priority = (not resolved, roi.observations == 1, roi.motion >= .2, roi.confidence,
                        -math.sqrt(w * h), -roi.roi_id)
            candidates.append((priority, roi, crop))
        candidates.sort(key=lambda x: x[0], reverse=True)
        budget = cfg.max_face_rois_per_frame if kind == "face" else cfg.max_plate_rois_per_frame
        return [(roi, crop) for _, roi, crop in candidates[:budget]]

    def faces(self, frame, rois, face):
        cfg = self.cfg
        self.face_evidence = []
        detections = face.detect(frame)
        if not (cfg.face_roi or cfg.small_object_rescue or cfg.head_fallback):
            return detections
        if cfg.face_roi or cfg.small_object_rescue:
            for roi, crop in self.select(rois, detections, "face", frame.shape):
                image = frame[crop[1]:crop[3], crop[0]:crop[2]]
                scale = min(4., cfg.face_roi_size / max(image.shape[:2]))
                image = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
                self.face_calls += 1
                for d in face.detect(image):
                    mapped = remap(d, crop, image.shape, frame.shape, roi.roi_id)
                    self.face_evidence.append({"parent_roi_id": roi.roi_id, "crop": crop, "box": mapped.box,
                                               "confidence": d.confidence, "geometry": geometry(mapped, frame.shape),
                                               "accepted": d.confidence >= cfg.roi_face_score and geometry(mapped, frame.shape) >= .55})
                    if d.confidence >= cfg.roi_face_score and geometry(mapped, frame.shape) >= .55:
                        detections.append(mapped)
        if cfg.head_fallback:
            for roi, _ in self.select(rois, [], "face", frame.shape):
                d = head_fallback(roi, detections, frame.shape)
                if d is not None:
                    detections.append(d)
                    self.head_masks += 1
        return fuse_detections(detections)

    def plates(self, frame, rois, detections, plate):
        cfg = self.cfg
        self.plate_evidence = []
        self.plate_frame += 1
        visible = {r.roi_id for r in rois}
        self.plate_history = {k: v for k, v in self.plate_history.items() if k in visible}
        self.plate_failures = {k: v for k, v in self.plate_failures.items() if k in visible}
        if cfg.plate_roi or cfg.small_object_rescue:
            for roi, crop in self.select(rois, detections, "plate", frame.shape):
                prior = self.plate_history.get(roi.roi_id)
                failure = self.plate_failures.get(roi.roi_id)
                low_evidence = any(crop[0] <= (d.box[0] + d.box[2]) / 2 <= crop[2]
                                   and crop[1] <= (d.box[1] + d.box[3]) / 2 <= crop[3] for d in detections)
                if (cfg.small_object_rescue and failure and failure[1] >= 2 and roi.stable and roi.motion < .2
                        and not low_evidence and self.plate_frame - failure[0] < cfg.object_interval):
                    continue
                # Two consistent crop hits permit a bounded refinement cooldown;
                # globals still run every frame. New/moving/unresolved candidates
                # are refined immediately until reliable evidence exists.
                if (cfg.small_object_rescue and prior and prior[1] >= 2 and roi.stable and roi.motion < .2
                        and self.plate_frame - prior[0] < cfg.object_interval):
                    continue
                self.plate_calls += 1
                image = frame[crop[1]:crop[3], crop[0]:crop[2]]
                accepted = False
                for d in plate.detect_roi(image, cfg.plate_roi_size):
                    mapped = remap(d, crop, image.shape, frame.shape, roi.roi_id)
                    self.plate_evidence.append({"parent_roi_id": roi.roi_id, "crop": crop, "box": mapped.box,
                                                "confidence": d.confidence, "geometry": geometry(mapped, frame.shape),
                                                "accepted": d.confidence >= cfg.roi_plate_score and geometry(mapped, frame.shape) >= .5})
                    if d.confidence >= cfg.roi_plate_score and geometry(mapped, frame.shape) >= .5:
                        accepted = True
                        detections.append(mapped)
                        count = prior[1] + 1 if prior and continuity(prior[2], mapped.box, cfg)[0] else 1
                        self.plate_history[roi.roi_id] = (self.plate_frame, count, mapped.box)
                if accepted:
                    self.plate_failures.pop(roi.roi_id, None)
                else:
                    self.plate_failures[roi.roi_id] = (self.plate_frame, failure[1] + 1 if failure else 1)
        return fuse_detections(detections)
