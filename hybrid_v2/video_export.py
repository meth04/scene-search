"""Optional result extraction: FFmpeg with audio, otherwise bounded OpenCV."""
from pathlib import Path
import math
import shutil
import subprocess
import cv2


def extract_clip(video, target, start, end, fps, *, privacy=False, privacy_cache=None,
                 privacy_mode="gaussian", privacy_strength="strong", privacy_block_size=16):
    if privacy:
        from src.privacy import render_private_clip
        if privacy_cache is None:
            raise ValueError("PRIVACY_CACHE_REQUIRED: raw output is disabled")
        return render_private_clip(video, privacy_cache, target, start, end, mode=privacy_mode,
                                   strength=privacy_strength, block_size=privacy_block_size)
    target = Path(target)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing result clip: {target}")
    if not all(math.isfinite(x) for x in (start, end, fps)) or not 0 <= start < end or fps <= 0:
        raise ValueError("Invalid extraction bounds")
    target.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-n", "-ss", str(start), "-i", str(video),
                        "-t", str(end-start), "-map", "0:v:0", "-map", "0:a?", "-c:v", "libx264",
                        "-preset", "fast", "-crf", "20", "-c:a", "aac", str(target)], check=True)
    else:
        cap = cv2.VideoCapture(str(video))
        writer = None
        try:
            cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
            for _ in range(max(1, int(math.ceil((end-start)*fps)))):
                ok, image = cap.read()
                if not ok or image is None:
                    raise RuntimeError("Video ended before the requested clip was decoded")
                if writer is None:
                    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError("OpenCV MP4 encoder unavailable; use an FFmpeg installation")
                writer.write(image)
        finally:
            cap.release()
            if writer is not None:
                writer.release()
    if not target.is_file() or target.stat().st_size == 0:
        raise RuntimeError("No extracted clip was created")
    cap = cv2.VideoCapture(str(target))
    try:
        rate = cap.get(cv2.CAP_PROP_FPS)
        duration = cap.get(cv2.CAP_PROP_FRAME_COUNT)/rate if rate else 0
        if abs(duration-(end-start)) > max(.5, 3/fps):
            raise RuntimeError("Extracted clip duration is inconsistent with requested bounds")
    finally:
        cap.release()
