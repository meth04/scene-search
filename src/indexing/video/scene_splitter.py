from src.indexing.domain.entities import SceneRange
from src.indexing.video.probe import VideoInfo


class SlidingWindowSceneSplitter:
    def __init__(self, duration_ms: int, stride_ms: int):
        self.duration_ms = duration_ms
        self.stride_ms = stride_ms

    def split(self, info: VideoInfo) -> list[SceneRange]:
        scenes = []
        start = 0
        while start < info.duration_ms:
            end = min(info.duration_ms, start + self.duration_ms)
            scenes.append(SceneRange(start_ms=start, end_ms=end))
            if end == info.duration_ms:
                break
            start += self.stride_ms
        return scenes
