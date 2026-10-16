from __future__ import annotations

import cv2
import numpy as np

from src.indexing.domain.entities import Frame, SceneRange
from src.indexing.domain.errors import ItemIndexingError
from src.indexing.video.probe import VideoInfo


class FrameSampler:
    @staticmethod
    def uniform_timestamps(scene: SceneRange, count: int) -> list[int]:
        if count <= 0:
            raise ValueError("sample count must be positive")
        if count == 1:
            return [(scene.start_ms + scene.end_ms) // 2]
        values = np.linspace(scene.start_ms, max(scene.start_ms, scene.end_ms - 1), count)
        return sorted(set(int(round(value)) for value in values))

    def sample(self, info: VideoInfo, scene: SceneRange, count: int) -> list[Frame]:
        return self.decode_at(info, self.uniform_timestamps(scene, count))

    def decode_preview(self, info: VideoInfo, scene: SceneRange, preview_fps: float) -> list[Frame]:
        interval = max(1, round(1000 / preview_fps))
        timestamps = list(range(scene.start_ms, scene.end_ms, interval))
        if not timestamps:
            timestamps = [scene.start_ms]
        return self.decode_at(info, timestamps)

    def decode_at(self, info: VideoInfo, timestamps: list[int]) -> list[Frame]:
        capture = cv2.VideoCapture(info.uri)
        frames: list[Frame] = []
        try:
            if not capture.isOpened():
                raise ItemIndexingError(f"Cannot decode video: {info.uri}")
            for timestamp in timestamps:
                capture.set(cv2.CAP_PROP_POS_MSEC, timestamp)
                ok, bgr = capture.read()
                if not ok or bgr is None:
                    continue
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                actual = max(0, int(round(capture.get(cv2.CAP_PROP_POS_MSEC))))
                index = max(0, int(round(capture.get(cv2.CAP_PROP_POS_FRAMES))) - 1)
                frames.append(Frame(
                    pixels=rgb, timestamp_ms=actual or timestamp, frame_index=index,
                    source_size=(info.width, info.height),
                ))
        finally:
            capture.release()
        if not frames:
            raise ItemIndexingError(f"No decodable frames in {info.uri}")
        return frames

    @staticmethod
    def resample_frames(frames: list[Frame], count: int) -> list[Frame]:
        if not frames:
            raise ItemIndexingError("Cannot sample empty anonymized frame list")
        indices = np.linspace(0, len(frames) - 1, min(count, len(frames))).round().astype(int)
        return [frames[index] for index in indices]
