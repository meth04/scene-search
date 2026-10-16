from __future__ import annotations

import cv2
import numpy as np

from src.indexing.domain.entities import Frame, SensitiveDetection


def expand_and_clamp(
    box: tuple[float, float, float, float], width: int, height: int, ratio: float
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return (
        max(0, min(width - 1, int(x1 - dx))),
        max(0, min(height - 1, int(y1 - dy))),
        max(1, min(width, int(np.ceil(x2 + dx)))),
        max(1, min(height, int(np.ceil(y2 + dy)))),
    )


class NoOpTracker:
    def track(self, detections: list[list[SensitiveDetection]]) -> list[list[SensitiveDetection]]:
        return detections


class Anonymizer:
    def __init__(self, expansion: float = 0.12, tracker=None):
        self.expansion = expansion
        self.tracker = tracker or NoOpTracker()

    def anonymize(
        self, frames: list[Frame], detections: list[list[SensitiveDetection]]
    ) -> list[Frame]:
        if len(frames) != len(detections):
            raise ValueError("privacy detection output count must match preview frames")
        tracked = self.tracker.track(detections)
        output = []
        for frame, frame_detections in zip(frames, tracked):
            pixels = np.asarray(frame.pixels).copy()
            height, width = pixels.shape[:2]
            for detection in frame_detections:
                x1, y1, x2, y2 = expand_and_clamp(
                    detection.box_xyxy, width, height, self.expansion
                )
                region = pixels[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                kernel = max(3, (min(region.shape[:2]) // 3) | 1)
                pixels[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)
            output.append(frame.model_copy(update={"pixels": pixels}))
        return output
