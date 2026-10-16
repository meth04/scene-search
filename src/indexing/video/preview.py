from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np

from src.indexing.domain.entities import Frame


class PreviewEncoder:
    def encode(self, frames: list[Frame], fps: float) -> bytes:
        if not frames:
            raise ValueError("preview requires frames")
        width, height = frames[0].source_size
        temporary = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        path = Path(temporary.name)
        temporary.close()
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        try:
            if not writer.isOpened():
                raise RuntimeError("OpenCV cannot initialize MP4 preview encoder")
            for frame in frames:
                rgb = np.asarray(frame.pixels, dtype=np.uint8)
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        try:
            return path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
