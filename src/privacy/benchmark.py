from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from .detectors import YoloPlateDetector, YuNetFaceDetector
from .metrics import MemorySampler
from .pipeline import scan_video_privacy
from .render import apply_privacy_to_frame
from .tracks import PrivacyTracks, write_json


def benchmark(video, output, config, max_frames=300):
    output = Path(output)
    results = []
    cfg = config.resolved()
    face, plate = YuNetFaceDetector(cfg), YoloPlateDetector(cfg)
    warm = cv2.VideoCapture(str(video))
    try:
        ok, frame = warm.read()
        if not ok:
            raise ValueError("Cannot decode benchmark video")
        for _ in range(3):
            face.detect(frame)
            plate.detect(frame)
    finally:
        warm.release()
    intervals = (1,) if any((cfg.face_roi, cfg.plate_roi, cfg.head_fallback, cfg.small_object_rescue, cfg.overlap)) else (1, 2, 3)
    for interval in intervals:
        sampler = MemorySampler()
        try:
            result = scan_video_privacy(
                video,
                output / f"detect_{interval}",
                replace(cfg, detect_every=interval),
                face_detector=face,
                plate_detector=plate,
                max_frames=max_frames,
            )
        finally:
            driver_peak = sampler.stop()
        count = result.meta["covered_frames"]
        offsets = np.concatenate(([0], np.cumsum(np.bincount(result.tracks["frame_index"], minlength=count))))
        cache = PrivacyTracks(result.meta, result.tracks, offsets)
        cap = cv2.VideoCapture(str(video))
        blur_seconds = 0.0
        started = time.perf_counter()
        try:
            for i in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError("Benchmark render decode failed")
                t = time.perf_counter()
                apply_privacy_to_frame(frame, cache.boxes(i))
                blur_seconds += time.perf_counter() - t
        finally:
            cap.release()
        metrics = {
            **result.meta["metrics"],
            "detect_every": interval,
            "cached_render_fps": count / (time.perf_counter() - started),
            "blur_ms_per_frame": blur_seconds * 1000 / count,
            "risk_frames": result.meta["risk_frames"],
            "track_counts": result.meta["track_counts"],
            "resolution": [result.meta["video"]["width"], result.meta["video"]["height"]],
            "source_fps": result.meta["video"]["fps"],
            "frames": count,
            "sampled_driver_vram_peak_mb": driver_peak,
            "driver_memory_sampling_ms": 100,
        }
        results.append(metrics)
        print(
            f"detect-every={interval}: scan={metrics['scan_fps']:.2f} FPS, cached render={metrics['cached_render_fps']:.2f} FPS",
            flush=True,
        )
    report = {
        "video": str(video),
        "config": result.meta["config"],
        "hardware": result.meta["hardware"],
        "warmup_detection_calls": 3,
        "results": results,
        "cached_render_includes_encoding": False,
        "note": "Sequential decode+anonymization, excluding encoding/audio; not a privacy recall benchmark",
    }
    write_json(output / "benchmark.json", report)
    return report
