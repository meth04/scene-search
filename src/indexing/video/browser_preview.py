"""Prepare browser media without changing the original or its scene timeline."""

from __future__ import annotations

import os
import subprocess
from uuid import uuid4

from src.indexing.models.driveretriever_adapter import file_checksum


def prepare_browser_media(asset, reader):
    source = reader.path(asset.uri)
    streams = asset.video_metadata.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio = [s for s in streams if s.get("codec_type") == "audio"]
    native = (
        source.suffix.lower() in {".mp4", ".m4v"}
        and video.get("codec_name") == "h264"
        and video.get("pix_fmt") in {"yuv420p", "yuvj420p"}
        and abs(float(video.get("start_time", 0))) < 0.001
        and all(s.get("codec_name") in {"aac", "mp3"} for s in audio)
    )
    if native:
        return {
            "path": str(source),
            "sha256": asset.checksum,
            "source_checksum": asset.checksum,
            "media_type": "video/mp4",
            "converted": False,
        }
    directory = source.parent / ".scene-previews"
    directory.mkdir(exist_ok=True)
    target = directory / (asset.asset_id + ".mp4")
    temporary = directory / (uuid4().hex + ".mp4")
    try:
        # Rebase both streams to zero, matching the window pipeline's relative PTS.
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-vf",
                "setpts=PTS-STARTPTS,pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-af",
                "asetpts=PTS-STARTPTS",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "23",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(temporary),
            ],
            check=True,
            capture_output=True,
            timeout=7200,
        )
        os.replace(temporary, target)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("BROWSER_PREVIEW_FAILED: không thể tạo video MP4 để xem; kiểm tra FFmpeg") from exc
    finally:
        temporary.unlink(missing_ok=True)
    return {
        "path": str(target),
        "sha256": file_checksum(target),
        "source_checksum": asset.checksum,
        "media_type": "video/mp4",
        "converted": True,
    }


def browser_media_valid(record, asset, reader):
    try:
        return bool(
            record
            and record["source_checksum"] == asset.checksum
            and file_checksum(reader.path(record["path"])) == record["sha256"]
        )
    except (OSError, ValueError, KeyError):
        return False
