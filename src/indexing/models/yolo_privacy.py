from src.indexing.config import ModelConfig
from src.indexing.domain.entities import Frame, SensitiveDetection
from src.indexing.models.model_artifacts import local_model as _local_model
from src.indexing.models.yolo_object import YoloObjectAdapter


class YoloPrivacyAdapter(YoloObjectAdapter):
    @classmethod
    def from_config(cls, config: ModelConfig):
        root, revision = _local_model(config)
        return cls(root, config, revision)

    def detect(self, frames: list[Frame]) -> list[list[SensitiveDetection]]:
        raw = super().detect(frames)
        return [[SensitiveDetection(**item.model_dump()) for item in detections] for detections in raw]
