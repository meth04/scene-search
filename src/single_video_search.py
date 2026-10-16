"""Bounded-memory single-video windowing, ranking, and clip extraction."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Window:
    window_id: int
    start_sec: float
    end_sec: float
    duration_sec: float
    partial: bool


def timestamp(seconds: float) -> str:
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("Timestamp must be finite and non-negative")
    milliseconds = round(seconds * 1000)
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_windows(
    duration_sec: float, window_sec: float = 10.0, stride_sec: float = 5.0, minimum_coverage: float = 0.5
) -> list[Window]:
    if not all(math.isfinite(x) and x > 0 for x in (duration_sec, window_sec, stride_sec)):
        raise ValueError("Duration, window, and stride must be finite positive numbers")
    if not 0 < minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be in (0, 1]")
    if (window_sec, stride_sec, minimum_coverage) == (10.0, 5.0, 0.5):
        from src.indexing.video.window_sampling import generate_windows as fixed_windows
        return [Window(w.window_index, w.start_ms/1000, w.end_ms/1000,
                       (w.end_ms-w.start_ms)/1000, w.partial) for w in fixed_windows(duration_sec*1000)]
    windows = []
    index = 0
    while True:
        start = index * stride_sec
        remaining = duration_sec - start
        if remaining <= 1e-6:
            break
        if index and remaining < window_sec * minimum_coverage - 1e-6:
            break
        end = min(duration_sec, start + window_sec)
        windows.append(Window(index, start, end, end - start, end - start < window_sec - 1e-6))
        index += 1
    return windows


def temporal_iou(a: Window, b: Window) -> float:
    intersection = max(0.0, min(a.end_sec, b.end_sec) - max(a.start_sec, b.start_sec))
    union = a.duration_sec + b.duration_sec - intersection
    return intersection / union if union else 0.0


def rank_windows(windows: list[Window], scores: list[float], top_k: int, dedup_iou: float | None = 0.25) -> list[dict]:
    if len(windows) != len(scores) or len(set(w.window_id for w in windows)) != len(windows):
        raise ValueError("Windows and scores must match with unique window IDs")
    if top_k <= 0 or dedup_iou is not None and not 0 < dedup_iou <= 1:
        raise ValueError("Invalid top_k or dedup_iou")
    if not all(math.isfinite(float(score)) for score in scores):
        raise ValueError("Scores contain non-finite values")
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), windows[i].window_id))
    chosen: list[dict] = []
    for raw_rank, index in enumerate(order, 1):
        window = windows[index]
        if dedup_iou is not None and any(temporal_iou(window, windows[item["_index"]]) >= dedup_iou for item in chosen):
            continue
        chosen.append(
            {
                "rank": len(chosen) + 1,
                "final_rank": len(chosen) + 1,
                "raw_rank": raw_rank,
                "window_id": window.window_id,
                "score": float(scores[index]),
                "start_sec": window.start_sec,
                "end_sec": window.end_sec,
                "duration_sec": round(window.duration_sec, 1),
                "start_timestamp": timestamp(window.start_sec),
                "end_timestamp": timestamp(window.end_sec),
                "partial": window.partial,
                "_index": index,
            }
        )
        if len(chosen) == top_k:
            break
    for item in chosen:
        del item["_index"]
    return chosen


def probe_video(path: Path) -> dict:
    import cv2

    if not path.is_file():
        raise FileNotFoundError(f"Input video does not exist: {path}")
    metadata = {}
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        command = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,codec_name,avg_frame_rate,nb_frames,duration:format=duration",
            "-of",
            "json",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            try:
                data = json.loads(result.stdout)
                stream = data["streams"][0]
                duration = float(stream.get("duration") or data["format"]["duration"])
                numerator, denominator = map(float, stream["avg_frame_rate"].split("/"))
                metadata = {
                    "duration_sec": duration,
                    "fps": numerator / denominator,
                    "frame_count": int(stream.get("nb_frames") or 0),
                    "width": int(stream["width"]),
                    "height": int(stream["height"]),
                    "codec": stream.get("codec_name", "unknown"),
                }
            except (KeyError, ValueError, ZeroDivisionError, IndexError):
                metadata = {}
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Video cannot be opened: {path}")
        ok, first = capture.read()
        if not ok or first is None:
            raise ValueError(f"Video has no decodable frames: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not metadata:
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
            metadata = {
                "duration_sec": count / fps if fps > 0 else 0,
                "fps": fps,
                "frame_count": count,
                "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "codec": codec or "unknown",
            }
        if metadata["duration_sec"] <= 0 or metadata["fps"] <= 0:
            raise ValueError(f"Invalid video timing metadata: {path}")
        # A container duration beyond the last decodable frame creates fictional windows.
        if count > 0 and fps > 0:
            decodable_estimate = count / fps
            if abs(metadata["duration_sec"] - decodable_estimate) > max(1.0, 2 / fps):
                metadata["duration_sec"] = decodable_estimate
        metadata["width"] = metadata["width"] or first.shape[1]
        metadata["height"] = metadata["height"] or first.shape[0]
        return metadata
    finally:
        capture.release()


class WindowDecoder:
    def __init__(self, path: Path, fps: float, num_frames: int):
        import cv2

        self.cv2 = cv2
        self.capture = cv2.VideoCapture(str(path))
        if not self.capture.isOpened():
            self.capture.release()
            raise ValueError(f"Cannot decode video: {path}")
        self.fps = fps
        self.num_frames = num_frames

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self) -> None:
        self.capture.release()

    def decode(self, window: Window) -> list:
        import numpy as np

        # Sample bin centers: all requested timestamps are strictly inside real content.
        times = window.start_sec + (np.arange(self.num_frames) + 0.5) * window.duration_sec / self.num_frames
        frames = []
        for target in times:
            frame = None
            for offset in (0.0, 1 / self.fps, 2 / self.fps):
                seek = max(window.start_sec, float(target) - offset)
                self.capture.set(self.cv2.CAP_PROP_POS_MSEC, seek * 1000)
                ok, bgr = self.capture.read()
                if ok and bgr is not None:
                    frame = self.cv2.cvtColor(bgr, self.cv2.COLOR_BGR2RGB)
                    break
            if frame is None:
                raise ValueError(f"Could not decode frame near {target:.3f}s in window {window.window_id}")
            frames.append(frame)
        return frames


def extract_clip(source: Path, target: Path, start_sec: float, end_sec: float, fps: float) -> None:
    import cv2

    duration = end_sec - start_sec
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{start_sec:.3f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.3f}",
            "-map",
            "0:v:0",
            "-map",
            "0:a?",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-c:a",
            "aac",
            str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg extraction failed: {result.stderr.strip()}")
    else:
        capture = cv2.VideoCapture(str(source))
        try:
            capture.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)
            ok, first = capture.read()
            if not ok or first is None:
                raise RuntimeError(f"Cannot read clip start at {start_sec:.3f}s")
            writer = cv2.VideoWriter(
                str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (first.shape[1], first.shape[0])
            )
            if not writer.isOpened():
                raise RuntimeError("OpenCV MP4 writer unavailable; install FFmpeg for extraction")
            try:
                writer.write(first)
                maximum_frames = max(1, math.ceil(duration * fps))
                for _ in range(maximum_frames - 1):
                    ok, frame = capture.read()
                    if not ok or frame is None:
                        break
                    position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                    if position > end_sec + 1 / fps:
                        break
                    writer.write(frame)
            finally:
                writer.release()
        finally:
            capture.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError(f"Clip extraction created no data: {target}")
    actual = probe_video(target)["duration_sec"]
    if abs(actual - duration) > max(0.5, 3 / fps):
        raise RuntimeError(f"Clip duration mismatch: expected {duration:.3f}s, got {actual:.3f}s at {target}")


def window_dicts(windows: list[Window]) -> list[dict]:
    return [asdict(window) for window in windows]
