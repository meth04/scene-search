from __future__ import annotations

import hashlib
import json
import math
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.indexing.domain.states import JobStatus


def canonical_json(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json", exclude_none=False)
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def deterministic_scene_id(source_id: str, video_id: str, start_ms: int, end_ms: int, config_hash: str) -> str:
    raw = f"{source_id}\x1f{video_id}\x1f{start_ms}\x1f{end_ms}\x1f{config_hash}"
    return hashlib.sha256(raw.encode()).hexdigest()


class VideoManifestItem(BaseModel):
    video_id: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetManifest(BaseModel):
    source_id: str = Field(min_length=1)
    dataset_version: str = Field(min_length=1)
    videos: list[VideoManifestItem] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_video_ids(self) -> DatasetManifest:
        ids = [item.video_id for item in self.videos]
        if len(ids) != len(set(ids)):
            raise ValueError("video_id must be unique inside a manifest")
        return self


class ModelInputContract(BaseModel):
    color_space: Literal["RGB"] = "RGB"
    image_size: tuple[int, int]
    num_frames: int | None = Field(default=None, gt=0)
    sampling: str | None = None


class ModelOutputContract(BaseModel):
    embedding_dim: int | None = Field(default=None, gt=0)
    normalized: bool | None = None
    labels: list[str] | None = None


class ModelManifest(BaseModel):
    """Checksummed handoff manifest for independent detector and weather adapters."""

    contract_version: Literal["1.0"]
    task: Literal[
        "object_detection",
        "privacy_detection",
        "weather_classification",
    ]
    framework: Literal["transformers", "ultralytics", "pytorch", "fake"]
    model_version: str = Field(min_length=1)
    artifact_files: list[str] = Field(min_length=1)
    sha256: dict[str, str]
    input: ModelInputContract
    output: ModelOutputContract
    recommended_thresholds: dict[str, float] = Field(default_factory=dict)
    training_data_version: str | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    license: str | None = None

    @model_validator(mode="after")
    def checksums_cover_artifacts(self) -> ModelManifest:
        """Validate artifact checksum coverage and detector label contracts.

        Returns:
            The validated manifest.

        Raises:
            ValueError: If persisted inputs, configuration or operation invariants are invalid.
        """
        missing = set(self.artifact_files) - set(self.sha256)
        if missing:
            raise ValueError(f"missing sha256 for artifacts: {sorted(missing)}")
        for digest in self.sha256.values():
            if len(digest) != 64 or any(c not in "0123456789abcdefABCDEF" for c in digest):
                raise ValueError("artifact sha256 must contain 64 hexadecimal characters")
        labels = set(self.output.labels or [])
        if self.task in {"object_detection", "weather_classification"} and not labels:
            raise ValueError(f"{self.task} requires a non-empty label map")
        if self.task == "privacy_detection" and labels != {"face", "plate"}:
            raise ValueError("privacy labels must map exactly to face and plate")
        return self


class ModelDescriptor(BaseModel):
    task: str
    provider: Literal["fake", "local", "huggingface"]
    repo_id: str | None = None
    requested_revision: str
    resolved_revision: str
    model_version: str
    artifact_checksums: dict[str, str] = Field(default_factory=dict)
    framework: str
    device: str = "cpu"
    contract_version: str = "1.0"


class Frame(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    pixels: Any
    timestamp_ms: int = Field(ge=0)
    frame_index: int = Field(ge=0)
    source_size: tuple[int, int]


class Detection(BaseModel):
    box_xyxy: tuple[float, float, float, float]
    label: str = Field(min_length=1)
    score: float = Field(ge=0, le=1)

    @field_validator("box_xyxy")
    @classmethod
    def valid_box(cls, box: tuple[float, float, float, float]):
        if not all(math.isfinite(v) for v in box) or box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("box must be finite xyxy with positive area")
        return box


class SensitiveDetection(Detection):
    label: Literal["face", "plate"]


class ClassProbabilities(BaseModel):
    probabilities: dict[str, float]

    @model_validator(mode="after")
    def valid_probabilities(self) -> ClassProbabilities:
        if not self.probabilities:
            raise ValueError("probabilities cannot be empty")
        if any(not math.isfinite(v) or v < 0 or v > 1 for v in self.probabilities.values()):
            raise ValueError("probabilities must be finite and in [0, 1]")
        total = sum(self.probabilities.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError("probabilities must sum to 1")
        return self


class Embedding(BaseModel):
    vector: list[float]
    dimension: int = Field(gt=0)
    normalized: bool

    @model_validator(mode="after")
    def valid_vector(self) -> Embedding:
        if len(self.vector) != self.dimension:
            raise ValueError("embedding dimension does not match vector")
        if any(not math.isfinite(v) for v in self.vector):
            raise ValueError("embedding contains NaN or Inf")
        return self


class SceneRange(BaseModel):
    start_ms: int = Field(ge=0)
    end_ms: int = Field(gt=0)

    @model_validator(mode="after")
    def ordered(self) -> SceneRange:
        if self.end_ms <= self.start_ms:
            raise ValueError("scene end must be after start")
        return self


class ObjectMetadata(BaseModel):
    classes: list[str]
    max_counts: dict[str, int]
    presence_ratio: dict[str, float]


class WeatherMetadata(BaseModel):
    weather: str
    confidence: float = Field(ge=0, le=1)
    lighting: str = "unknown"


class IndexJob(BaseModel):
    job_id: str
    index_version: str
    dataset_version: str
    config_snapshot: dict[str, Any]
    pipeline_config_hash: str
    status: JobStatus = JobStatus.PENDING
    total_items: int
    processed_items: int = 0
    quarantined_scenes: int = 0
    failed_items: int = 0
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ValidationGate(BaseModel):
    name: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    index_version: str
    gates: list[ValidationGate]
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def passed(self) -> bool:
        return bool(self.gates) and all(gate.passed for gate in self.gates)
