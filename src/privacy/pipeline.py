from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from .config import PrivacyConfig, hardware
from .tracker import PrivacyTracker
from .tracks import cache_format, probe, save_tracks, write_json


def scan_video_privacy(video, cache, config=None, *, face_detector=None, plate_detector=None, object_guide=None, max_frames=None, segment_manifest=None):
    """Read ORIGINAL frames only. Writes only the caller's separate privacy directory.

    max_frames is for benchmarks; partial scans are deliberately unrenderable.
    The offline cache backfills a new detection by at most N-1 frames within its
    scene, limiting entry gaps without claiming any detector recall guarantee.
    """
    from .detectors import YoloPlateDetector, YuNetFaceDetector

    info = probe(video)
    from hybrid_v2.segments import boundaries
    boundary_frames = boundaries(segment_manifest, info["frame_count"] / info["fps"], info["fps"]) if segment_manifest is not None else []
    boundary_set = set(boundary_frames)
    reset_evidence = []
    capabilities = hardware()
    cfg = (config or PrivacyConfig()).resolved(capabilities)
    existing = Path(cache) / "privacy_meta.json"
    if existing.exists():
        version = cache_format(asdict(cfg))
        if json.loads(existing.read_text(encoding="utf-8")).get("format") != version:
            raise ValueError("Use a separate --cache directory: existing privacy cache version is preserved")
    cv2.setNumThreads(cfg.cpu_threads)
    loaded = time.perf_counter()
    face = face_detector or YuNetFaceDetector(cfg)
    plate = plate_detector or YoloPlateDetector(cfg)
    tracker = PrivacyTracker(cfg, info["fps"])
    from .refinement import ObjectGuide, Refinement
    refinement = Refinement(cfg)
    guided = cfg.face_roi or cfg.plate_roi or cfg.small_object_rescue or cfg.head_fallback
    guide = object_guide or (ObjectGuide(cfg) if guided else None)
    model_load_seconds = time.perf_counter() - loaded
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="privacy-face") if cfg.overlap else None
    import torch

    if capabilities["cuda"]:
        torch.cuda.reset_peak_memory_stats()
    metrics = dict(
        decode_seconds=0.0,
        face_seconds=0.0,
        plate_seconds=0.0,
        tracking_seconds=0.0,
        detection_frames=0,
        tracker_only_frames=0,
        object_seconds=0.0,
    )
    cap = cv2.VideoCapture(str(video))
    boxes, risks, previous, scene_start, frame_index = [], [], None, 0, 0
    seen, active = set(), set()
    last_boxes = {}
    started = time.perf_counter()
    limit = min(info["frame_count"], max_frames) if max_frames is not None else info["frame_count"]
    if limit <= 0:
        raise ValueError("max-frames must be positive")
    try:
        while frame_index < limit:
            if frame_index in boundary_set:
                tracker.reset()
                refinement.reset()
                if guide is not None:
                    guide.reset()
                seen.clear()
                active.clear()
                last_boxes.clear()
                previous = None
                scene_start = frame_index
                state_count = sum(len(getattr(tracker, k)) for k in ("last", "smooth", "direct", "candidates", "backend_keys"))
                state_count += sum(len(b.tracked_stracks)+len(b.lost_stracks)+len(b.removed_stracks) for b in tracker.backends.values())
                reset_evidence.append({"frame_index": frame_index, "reset_applied": True, "remaining_tracker_state": state_count})
                risks.append({"frame_index": frame_index, "reasons": ["explicit_segment_boundary"]})
            # Batching only when EVERY frame is detected. Adaptive refresh and
            # frame skipping use one frame at a time to preserve causal behavior.
            batch = min(cfg.batch_size if cfg.detect_every == 1 else 1, limit - frame_index)
            # Stateful refinement must observe resets before its next detection.
            if guide is not None or boundary_frames:
                batch = 1
            next_boundary = next((b for b in boundary_frames if b > frame_index), limit)
            batch = min(batch, next_boundary - frame_index)
            frames = []
            for _ in range(batch):
                t = time.perf_counter()
                ok, frame = cap.read()
                metrics["decode_seconds"] += time.perf_counter() - t
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Privacy decode stopped at frame {frame_index + len(frames)}; cache not committed"
                    )
                # Reject VFR rather than misalign a CFR output with the source.
                pts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000
                expected = (frame_index + len(frames)) / info["fps"]
                if abs(pts - expected) > max(0.05, 1.5 / info["fps"]):
                    raise ValueError("Variable frame timing is unsupported by privacy CFR rendering")
                frames.append(frame)
            batch_plates = None
            rois = [[] for _ in frames]
            if guide is not None:
                t = time.perf_counter()
                rois = [guide.detect(frame, frame_index + offset) for offset, frame in enumerate(frames)]
                metrics["object_seconds"] += time.perf_counter() - t
            def face_batch():
                t = time.perf_counter()
                groups = [refinement.faces(frame, hints, face) for frame, hints in zip(frames, rois)]
                return groups, time.perf_counter() - t
            future = pool.submit(face_batch) if pool else None
            if cfg.detect_every == 1 and hasattr(plate, "detect_batch"):
                t = time.perf_counter()
                batch_plates = plate.detect_batch(frames)
                metrics["plate_seconds"] += time.perf_counter() - t
                if len(batch_plates) != len(frames):
                    raise RuntimeError("Privacy plate batch/frame alignment mismatch")
                if guided:
                    t = time.perf_counter()
                    batch_plates = [refinement.plates(frame, hints, group, plate)
                                    for frame, hints, group in zip(frames, rois, batch_plates)]
                    metrics["plate_seconds"] += time.perf_counter() - t
            batch_faces = None
            if future is not None:
                batch_faces, seconds = future.result()
                metrics["face_seconds"] += seconds
            for offset, frame in enumerate(frames):
                small = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2GRAY)
                cut = (
                    previous is not None
                    and np.mean(np.abs(small.astype(float) - previous.astype(float))) / 255 > cfg.scene_cut_threshold
                )
                previous = small
                if cut:
                    tracker.reset()
                    refinement.reset()
                    if guide is not None:
                        guide.reset()
                    seen.clear()
                    active.clear()
                    last_boxes.clear()
                    scene_start = frame_index
                    risks.append({"frame_index": frame_index, "reasons": ["scene_transition"]})
                refresh = tracker.refresh_needed(frame_index)
                detect = frame_index % cfg.detect_every == 0 or cut or refresh
                detections = []
                if detect:
                    metrics["detection_frames"] += 1
                    t = time.perf_counter()
                    detections.extend(batch_faces[offset] if batch_faces is not None else refinement.faces(frame, rois[offset], face))
                    if batch_faces is None:
                        metrics["face_seconds"] += time.perf_counter() - t
                    if batch_plates is None:
                        t = time.perf_counter()
                        detections.extend(plate.detect(frame))
                        metrics["plate_seconds"] += time.perf_counter() - t
                    else:
                        detections.extend(batch_plates[offset])
                t = time.perf_counter()
                current = tracker.update(frame_index, detections, frame.shape)
                metrics["tracking_seconds"] += time.perf_counter() - t
                reasons = set(tracker.events)
                keys = {(b.kind, b.track_id) for b in current}
                if active - keys:
                    reasons.add("track_disappeared")
                for box in current:
                    if box.origin == "head_fallback":
                        reasons.add("head_fallback")
                    elif box.origin.startswith("roi_"):
                        reasons.add("roi_refinement")
                    key = box.kind, box.track_id
                    if key not in seen:
                        reasons.add("new_track")
                        seen.add(key)
                    # Offline look-ahead protects skipped frames for BOTH new and
                    # continuing objects. Existing Kalman boxes remain as well,
                    # so rendering uses the union of past/future coverage.
                    if box.source == "detector":
                        for prior in range(max(scene_start, frame_index - cfg.detect_every + 1), frame_index):
                            boxes.append(
                                replace(
                                    box, frame_index=prior, timestamp_sec=prior / info["fps"], source="interpolated"
                                )
                            )
                    if box.confidence < 0.5:
                        reasons.add("low_confidence")
                    prior = last_boxes.get(key)
                    if prior is not None:
                        old_center = np.array([(prior.box[0] + prior.box[2]) / 2, (prior.box[1] + prior.box[3]) / 2])
                        center = np.array([(box.box[0] + box.box[2]) / 2, (box.box[1] + box.box[3]) / 2])
                        if np.linalg.norm(center - old_center) > max(20, prior.box[2] - prior.box[0]):
                            reasons.add("large_motion")
                    last_boxes[key] = box
                    if box.source == "tracker":
                        reasons.add("tracker_only")
                        age = frame_index - tracker.last.get(key, frame_index)
                        if age > cfg.detect_every:
                            reasons.add("detector_gap")
                        if age >= 4:
                            reasons.add("long_tracker_only_run")
                if any(b.source == "tracker" for b in current):
                    metrics["tracker_only_frames"] += 1
                if refresh:
                    reasons.add("forced_refresh")
                if reasons:
                    risks.append({"frame_index": frame_index, "reasons": sorted(reasons)})
                boxes.extend(current)
                active = keys
                last_boxes = {key: value for key, value in last_boxes.items() if key in active}
                frame_index += 1
    finally:
        cap.release()
        if pool:
            pool.shutdown(wait=True, cancel_futures=True)
    elapsed = time.perf_counter() - started
    import psutil

    metrics.update(
        scan_seconds=elapsed,
        scan_fps=frame_index / elapsed,
        decode_fps=frame_index / max(metrics["decode_seconds"], 1e-9),
        face_ms_per_frame=1000 * metrics["face_seconds"] / frame_index,
        face_ms_per_detection_frame=1000 * metrics["face_seconds"] / metrics["detection_frames"],
        plate_ms_per_detection_frame=1000 * metrics["plate_seconds"] / metrics["detection_frames"],
        tracking_ms_per_frame=1000 * metrics["tracking_seconds"] / frame_index,
        validation_ms_per_frame=1000 * tracker.validation_seconds / frame_index,
        peak_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if capabilities["cuda"] else 0,
        peak_torch_reserved_mb=torch.cuda.max_memory_reserved() / 2**20 if capabilities["cuda"] else 0,
        cpu_rss_mb=psutil.Process().memory_info().rss / 2**20,
        model_load_seconds=model_load_seconds,
        face_roi_calls_per_frame=refinement.face_calls / frame_index,
        plate_roi_calls_per_frame=refinement.plate_calls / frame_index,
        head_fallback_masks=refinement.head_masks,
        object_calls=getattr(guide, "calls", 0),
        object_ms_per_frame=1000 * metrics["object_seconds"] / frame_index,
    )
    meta = {
        "video": info,
        "covered_frames": frame_index,
        "complete": frame_index == info["frame_count"],
        "created_at": datetime.now(UTC).isoformat(),
        "hardware": capabilities,
        "config": asdict(cfg),
        "face_detector": getattr(face, "descriptor", {"name": type(face).__name__}),
        "plate_detector": getattr(plate, "descriptor", {"name": type(plate).__name__}),
        "object_guide": getattr(guide, "descriptor", None),
        "tracking_backend": "Ultralytics ByteTrack CPU, separate face/plate, no ReID",
        "known_boundaries": {"expected": len([b for b in boundary_frames if b < limit]), "applied": len(reset_evidence),
                             "stale_carry_over": sum(r["remaining_tracker_state"] for r in reset_evidence), "resets": reset_evidence},
        "metrics": metrics,
        "risk_frames": len(risks),
        "track_counts": {kind: len({b.track_id for b in boxes if b.kind == kind}) for kind in ("face", "plate")},
    }
    t = time.perf_counter()
    result = save_tracks(cache, meta, boxes, risks)
    result.meta["metrics"]["serialization_seconds"] = time.perf_counter() - t
    write_json(Path(cache) / "privacy_meta.json", result.meta)
    return result
