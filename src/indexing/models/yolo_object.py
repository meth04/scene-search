from __future__ import annotations

from src.indexing.config import ModelConfig
from src.indexing.domain.entities import Detection, Frame
from src.indexing.domain.errors import FatalIndexingError
from src.indexing.models.base import descriptor_from_manifest, read_model_manifest
from src.indexing.models.model_artifacts import local_model as _local_model


class YoloObjectAdapter:
    def __init__(self, root, config: ModelConfig, revision: str):
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise FatalIndexingError("Install ultralytics for YOLO object detection") from exc
        manifest = read_model_manifest(root)
        self.descriptor = descriptor_from_manifest(
            manifest, config.provider, config.revision, revision, config.device, config.repo_id
        )
        candidates = [name for name in manifest.artifact_files if name.endswith((".pt", ".onnx"))]
        if not candidates:
            raise FatalIndexingError("YOLO manifest has no .pt or .onnx weights artifact")
        weights = root / candidates[0]
        self._model = YOLO(weights)
        self._device = config.device
        self._labels = manifest.output.labels or []

    @classmethod
    def from_config(cls, config: ModelConfig):
        root, revision = _local_model(config)
        return cls(root, config, revision)  # pragma: no cover

    def detect(self, frames: list[Frame]) -> list[list[Detection]]:
        results = self._model.predict([f.pixels for f in frames], device=self._device, verbose=False)
        output = []
        for result in results:
            detections = []
            for box, score, class_id in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                index = int(class_id)
                if index >= len(self._labels):
                    raise ValueError(f"class ID {index} has no canonical label")
                detections.append(
                    Detection(
                        box_xyxy=tuple(float(v) for v in box.tolist()),
                        label=self._labels[index],
                        score=float(score),
                    )
                )
            output.append(detections)
        return output
