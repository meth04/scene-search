from pathlib import Path

import cv2
from pydantic import BaseModel, Field

from src.indexing.domain.errors import ItemIndexingError


class VideoInfo(BaseModel):
    uri: str
    duration_ms: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec: str


class VideoProbe:
    def probe(self, uri: str) -> VideoInfo:
        if "://" not in uri and not Path(uri).is_file():
            raise ItemIndexingError(f"Video does not exist: {uri}")
        capture = cv2.VideoCapture(uri)
        try:
            if not capture.isOpened():
                raise ItemIndexingError(f"Cannot open video: {uri}")
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or count <= 0:
                raise ItemIndexingError(f"Invalid video timing metadata: {uri}")
            fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
            codec = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00")
            return VideoInfo(
                uri=uri,
                duration_ms=max(1, round(count / fps * 1000)),
                fps=fps,
                frame_count=count,
                width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                codec=codec or "unknown",
            )
        finally:
            capture.release()
