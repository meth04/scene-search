from __future__ import annotations

import numpy as np

from src.indexing.config import ModelConfig
from src.indexing.domain.entities import ClassProbabilities, Frame
from src.indexing.domain.errors import FatalIndexingError
from src.indexing.models.base import descriptor_from_manifest, read_model_manifest
from src.indexing.models.model_artifacts import local_model as _local_model


class WeatherAdapter:
    def __init__(self, root, config: ModelConfig, revision: str):
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModelForImageClassification
        except ImportError as exc:
            raise FatalIndexingError("Install torch and transformers for weather classification") from exc
        manifest = read_model_manifest(root)
        self.descriptor = descriptor_from_manifest(
            manifest, config.provider, config.revision, revision, config.device, config.repo_id
        )
        self._torch, self._device = torch, config.device
        self._processor = AutoImageProcessor.from_pretrained(root, local_files_only=True)
        self._model = (
            AutoModelForImageClassification.from_pretrained(root, local_files_only=True).to(config.device).eval()
        )
        self._labels = manifest.output.labels or []

    @classmethod
    def from_config(cls, config: ModelConfig):
        root, revision = _local_model(config)
        return cls(root, config, revision)

    def classify(self, frames: list[Frame]) -> list[ClassProbabilities]:
        inputs = self._processor(images=[np.asarray(f.pixels) for f in frames], return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}
        with self._torch.inference_mode():
            probabilities = self._model(**inputs).logits.softmax(dim=-1).cpu().tolist()
        return [ClassProbabilities(probabilities=dict(zip(self._labels, row))) for row in probabilities]
