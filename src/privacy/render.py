from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2

from .config import ROOT
from .tracks import load_privacy_tracks


def ffmpeg_binary():
    managed = ROOT / "data/bin/ffmpeg/bin/ffmpeg.exe"
    if managed.is_file():
        return str(managed)
    found = shutil.which("ffmpeg")
    local = ROOT / "outputs/ffmpeg-main/package/ffmpeg.exe"
    return found or (str(local) if local.is_file() else None)


@dataclass
class PrivacyRenderResult:
    path: Path
    frames: int
    fps: float
    render_fps: float
    blur_ms_per_frame: float
    audio_preserved: bool


def apply_privacy_to_frame(frame, boxes, mode="gaussian", strength="strong", block_size=16, *, debug_overlay=False):
    if mode not in {"gaussian", "pixelate"} or strength != "strong" or block_size < 2:
        raise ValueError("Privacy rendering requires gaussian/pixelate, strong strength and block-size >=2")
    output = frame.copy()
    height, width = output.shape[:2]
    boxes = list(boxes)
    for box in boxes:
        coords = box.box if hasattr(box, "box") else box
        x1, y1, x2, y2 = coords
        x1, y1, x2, y2 = (
            max(0, math.floor(x1)),
            max(0, math.floor(y1)),
            min(width, math.ceil(x2)),
            min(height, math.ceil(y2)),
        )
        if x2 <= x1 or y2 <= y1:
            continue
        roi = output[y1:y2, x1:x2]
        h, w = roi.shape[:2]
        if mode == "pixelate":
            # At most 6x4 cells even for a large ROI; tiny plates collapse to one cell.
            tiny = cv2.resize(
                roi, (max(1, min(6, w // block_size)), max(1, min(4, h // block_size))), interpolation=cv2.INTER_AREA
            )
            output[y1:y2, x1:x2] = cv2.resize(tiny, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            kernel = max(31, int(max(w, h) * 0.65)) | 1
            output[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (kernel, kernel), max(10.0, max(w, h) * 0.25))
    if debug_overlay:
        for box in boxes:
            if not hasattr(box, "state"):
                continue
            x1, y1, x2, y2 = map(round, box.box)
            color = (0, 210, 255) if box.state in {"PROVISIONAL", "SUSPICIOUS", "STALE"} else (80, 255, 80)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 1)
            label = f"{box.kind.upper()} {box.state} {box.confidence:.2f} Q{box.quality_score:.2f}"
            if box.origin != "legacy":
                label += f" {box.origin}"
            cv2.putText(output, label, (max(0, x1), max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    return output


def render_private_clip(video, cache, target, start, end, *, mode="gaussian", strength="strong", block_size=16, debug_overlay=False):
    """No detector or VLM imports. Frame-index aligned rendering, atomic publish.

    FFmpeg encodes anonymized BGR frames and remuxes source audio. With no FFmpeg,
    OpenCV writes video only (the existing raw extractor has the same fallback).
    """
    tracks = load_privacy_tracks(cache, video)
    info = tracks.meta["video"]
    fps, count = info["fps"], info["frame_count"]
    if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= count / fps + 1e-6:
        raise ValueError("Invalid private extraction bounds")
    first, stop = max(0, round(start * fps)), min(count, round(end * fps))
    if stop <= first:
        raise ValueError("Requested range is shorter than one video frame")
    target = Path(target)
    if target.exists() or target.resolve() == Path(video).resolve():
        raise FileExistsError(f"Refusing to overwrite media: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{uuid4().hex}.mp4")
    ffmpeg = ffmpeg_binary()
    cap, process, writer = cv2.VideoCapture(str(video)), None, None
    blur_seconds = 0.0
    started = time.perf_counter()
    try:
        # Decode from frame zero: no seek/keyframe rounding can change alignment.
        for i in range(first):
            if not cap.grab():
                raise RuntimeError(f"Privacy decoding failed before frame {i}")
        with tempfile.TemporaryFile() as errors:
            if ffmpeg:
                args = [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-n",
                    "-f",
                    "rawvideo",
                    "-pix_fmt",
                    "bgr24",
                    "-s",
                    f"{info['width']}x{info['height']}",
                    "-r",
                    str(fps),
                    "-i",
                    "pipe:0",
                    "-ss",
                    str(first / fps),
                    "-i",
                    str(video),
                    "-map",
                    "0:v:0",
                    "-map",
                    "1:a:0?",
                    "-t",
                    str((stop - first) / fps),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "20",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-movflags",
                    "+faststart",
                    str(temporary),
                ]
                process = subprocess.Popen(
                    args,
                    stdin=subprocess.PIPE,
                    stderr=errors,
                    stdout=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            else:
                writer = cv2.VideoWriter(
                    str(temporary), cv2.VideoWriter_fourcc(*"mp4v"), fps, (info["width"], info["height"])
                )
                if not writer.isOpened():
                    raise RuntimeError("Private MP4 encoder unavailable; install FFmpeg")
            for i in range(first, stop):
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"Privacy decoding failed at frame {i}")
                t = time.perf_counter()
                private = apply_privacy_to_frame(frame, tracks.boxes(i), mode, strength, block_size, debug_overlay=debug_overlay)
                blur_seconds += time.perf_counter() - t
                if process:
                    process.stdin.write(private.tobytes())
                else:
                    writer.write(private)
            if process:
                process.stdin.close()
                if process.wait(timeout=120):
                    errors.seek(0)
                    raise RuntimeError("Private encoding failed: " + errors.read().decode(errors="replace"))
            else:
                writer.release()
                writer = None
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError("Private encoder created no media")
        verification = cv2.VideoCapture(str(temporary))
        try:
            if not verification.read()[0] or int(verification.get(cv2.CAP_PROP_FRAME_COUNT)) != stop - first:
                raise RuntimeError("Private encoded media failed frame-count validation")
        finally:
            verification.release()
        # Exclusive reservation prevents concurrent requests from overwriting one another.
        with target.open("xb"):
            pass
        os.replace(temporary, target)
        elapsed = time.perf_counter() - started
        return PrivacyRenderResult(
            target, stop - first, fps, (stop - first) / elapsed, blur_seconds * 1000 / (stop - first), bool(ffmpeg)
        )
    finally:
        cap.release()
        if writer:
            writer.release()
        if process and process.poll() is None:
            process.kill()
            process.wait()
        temporary.unlink(missing_ok=True)
