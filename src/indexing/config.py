from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, SerializerFunctionWrapHandler, model_serializer, model_validator

from src.indexing.domain.entities import stable_hash
from src.indexing.metadata_config import SceneMetadataConfig


class ModelConfig(BaseModel):
    provider: Literal["fake", "local", "huggingface"] = "fake"
    repo_id: str | None = None
    revision: str = "fake-v1"
    cache_dir: str = "data/model-cache"
    device: str = "cpu"
    dtype: str = "float32"
    batch_size: int = Field(default=4, gt=0)
    allow_patterns: list[str] | None = None
    local_path: str | None = None
    trust_remote_code: bool = False

    @model_validator(mode="after")
    def provider_requirements(self) -> ModelConfig:
        if self.provider == "huggingface" and not self.repo_id:
            raise ValueError("repo_id is required for huggingface provider")
        if self.provider == "local" and not self.local_path:
            raise ValueError("local_path is required for local provider")
        if self.provider != "fake" and self.revision in {"", "latest", "main", "master"}:
            raise ValueError("model revision must be pinned; branch/latest values are forbidden")
        if self.trust_remote_code:
            raise ValueError("trust_remote_code requires a reviewed custom adapter and is disabled")
        return self


class ModelsConfig(BaseModel):
    """Retain serialized historical fields so saved DriveRetriever config hashes remain valid.

    The xclip field is snapshot compatibility data only; no runtime model uses it.
    """

    xclip: ModelConfig = Field(default_factory=ModelConfig)
    object: ModelConfig = Field(default_factory=ModelConfig)
    privacy: ModelConfig = Field(default_factory=ModelConfig)
    weather: ModelConfig = Field(default_factory=ModelConfig)


class DriveRetrieverConfig(BaseModel):
    provider: Literal["real", "fake"] = "real"
    checkpoint: str = "data/model-cache/driveretriever-epoch4/best.pt"
    checkpoint_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    revision: str = Field(default="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", pattern=r"^[a-f0-9]{40}$")
    device: str = "cpu"
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    batch_size: int = Field(default=2, ge=1, le=32)
    local_files_only: bool = True
    source_roots: list[str] = Field(default_factory=lambda: ["data/videos"])
    source_registry_path: str | None = None
    window_group_size: int = Field(default=16, ge=1, le=128)


class WorkflowConfig(BaseModel):
    provider: Literal["local", "temporal"] = "local"
    address: str = "localhost:7233"
    namespace: str = "default"
    task_queue: str = "driveretriever"


class QualityConfig(BaseModel):
    """Window geometry and historical quality fields retained in saved config hashes."""

    weather_confidence: float = Field(default=0.55, ge=0, le=1)
    privacy_box_expansion: float = Field(default=0.12, ge=0, le=1)
    scene_duration_ms: int = Field(default=10000, gt=0)
    scene_stride_ms: int = Field(default=5000, gt=0)
    analytic_frames: int = Field(default=8, gt=0)
    embedding_frames: int = Field(default=12, gt=0)
    preview_fps: float = Field(default=5.0, gt=0)
    max_failed_items: int = Field(default=0, ge=0)


class StorageConfig(BaseModel):
    provider: Literal["local", "production"] = "local"
    root_dir: str = "data/indexing"
    production_alias: str = "scenes_production"
    artifact_bucket: str = "scene-previews"


class UploadConfig(BaseModel):
    root_dir: str = "data/videos/uploads"
    max_file_bytes: int = Field(default=1024**3, gt=0, le=1024**3)
    max_files: int = Field(default=10000, gt=0, le=10000)


class PipelineConfig(BaseModel):
    """Pipeline settings with complete historical snapshot serialization preserved.

    Attributes:
        scene_metadata: Optional pinned label-generation settings; None preserves the old pipeline.
    """

    model_family: Literal["driveretriever"] = "driveretriever"
    driveretriever: DriveRetrieverConfig = Field(default_factory=DriveRetrieverConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    contract_version: Literal["1.0"] = "1.0"
    taxonomy_version: str = "traffic-v1"
    preprocessing_version: str = "source-rgb-siglip2-v1"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    uploads: UploadConfig = Field(default_factory=UploadConfig)
    scene_metadata: SceneMetadataConfig | None = None

    @model_serializer(mode="wrap")
    def historical_serialization(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Omit disabled metadata from every snapshot and config hash.

        Args:
            handler: Pydantic serializer for the remaining historical fields.

        Returns:
            Serialized settings with no new null key for legacy configurations.
        """
        value = handler(self)
        if self.scene_metadata is None:
            value.pop("scene_metadata", None)
        return value

    @model_validator(mode="after")
    def fixed_window_contract(self):
        if self.model_family == "driveretriever":
            q = self.quality
            if (q.scene_duration_ms, q.scene_stride_ms, q.embedding_frames) != (10000, 5000, 12):
                raise ValueError("DriveRetriever requires 10000ms windows, 5000ms stride and 12 frames")
            if self.storage.provider == "production" and (
                self.driveretriever.provider == "fake" or self.workflow.provider != "temporal"
            ):
                raise ValueError("production requires real DriveRetriever and Temporal")
            if self.storage.provider == "production" and not self.driveretriever.source_registry_path:
                raise ValueError("production requires a trusted source_registry_path")
        return self

    @property
    def config_hash(self) -> str:
        return stable_hash(self)

    @property
    def index_version(self) -> str:
        return self.config_hash[:16]


def load_structured(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read YAML configuration") from exc
    value = yaml.safe_load(text)
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain an object")
    return value


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    return PipelineConfig.model_validate(load_structured(path))
