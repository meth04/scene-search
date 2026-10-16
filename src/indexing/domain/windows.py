from __future__ import annotations

from typing import Literal
from uuid import UUID, uuid5

from pydantic import BaseModel, Field, field_validator, model_validator

from src.indexing.domain.entities import Embedding, stable_hash
from src.indexing.domain.scene_metadata import AGGREGATION_VERSION, PAYLOAD_VERSION, SCHEMA_VERSION, SceneMetadata

WINDOW_NAMESPACE = UUID("671c9951-a413-5e6e-9c61-2f22be3c501b")


def source_uuid(kind: str, *parts) -> str:
    return str(uuid5(WINDOW_NAMESPACE, stable_hash({"kind": kind, "parts": parts})))


class VideoAsset(BaseModel):
    asset_id: str
    source_id: str
    video_id: str
    checksum: str = Field(pattern=r"^[a-f0-9]{64}$")
    uri: str
    origin_uri: str | None = None
    size_bytes: int = Field(default=0, ge=0)
    mtime_ns: int = Field(default=0, ge=0)
    video_metadata: dict = Field(default_factory=dict)
    duration_ms: float = Field(gt=0, allow_inf_nan=False)
    stream_index: int = Field(ge=0)
    timeline_origin_pts: int
    time_base_num: int = Field(gt=0)
    time_base_den: int = Field(gt=0)
    tenant_id: str
    access_scope_id: str
    camera: str | None = None
    captured_at: str | None = None
    region: str | None = None
    available: bool = True
    media_policy_version: Literal["source_unmodified_internal_v1"] = "source_unmodified_internal_v1"

    @field_validator("asset_id")
    @classmethod
    def valid_uuid(cls, value):
        if UUID(value).version != 5:
            raise ValueError("source IDs must be UUIDv5")
        return value

    @field_validator("captured_at")
    @classmethod
    def valid_captured_at(cls, value):
        if value is not None:
            from datetime import datetime
            if datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is None:
                raise ValueError("captured_at must include a time zone")
        return value


class SourceFrameRef(BaseModel):
    frame_id: str
    asset_id: str
    stream_index: int
    pts: int
    time_base_num: int = Field(gt=0)
    time_base_den: int = Field(gt=0)
    timestamp_ms: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("frame_id", "asset_id")
    @classmethod
    def valid_uuid(cls, value):
        if UUID(value).version != 5:
            raise ValueError("frame/source IDs must be UUIDv5")
        return value


class WindowSample(BaseModel):
    sample_index: int = Field(ge=0, le=11)
    target_ms: float = Field(ge=0, allow_inf_nan=False)
    frame: SourceFrameRef


class WindowResult(BaseModel):
    """Persist one source window, its twelve samples and a normalized embedding.

    Attributes:
        scene_id: Stable UUIDv5 for this source interval.
        index_version: Immutable owning index version.
        samples: Twelve ordered midpoint samples, retaining original source PTS.
        embedding: Finite normalized 512-dimensional DriveRetriever vector.
        status: Staged until SQL and vector payload have been reconciled.
        metadata: Optional independent scene labels; missing differs from unknown.
        metadata_schema_version: Required supported schema when metadata exists.
        aggregation_version: Required supported aggregation rules when metadata exists.
    """

    scene_id: str
    index_version: str
    dataset_version: str
    asset_id: str
    video_id: str
    window_index: int = Field(ge=0)
    start_ms: float = Field(ge=0, allow_inf_nan=False)
    end_ms: float = Field(gt=0, allow_inf_nan=False)
    partial: bool
    tenant_id: str
    access_scope_id: str
    camera: str | None = None
    samples: list[WindowSample] = Field(min_length=12, max_length=12)
    embedding: Embedding
    embedding_contract_hash: str
    pipeline_config_hash: str
    checkpoint_sha256: str
    status: Literal["staged", "checkpointed"] = "staged"
    privacy_status: Literal["not_processed"] = "not_processed"
    media_kind: Literal["source_unmodified"] = "source_unmodified"
    media_policy_version: Literal["source_unmodified_internal_v1"] = "source_unmodified_internal_v1"
    metadata: SceneMetadata | None = None
    metadata_schema_version: Literal["scene_metadata_v1"] | None = None
    aggregation_version: Literal["metadata-aggregation-v1"] | None = None

    @model_validator(mode="after")
    def metadata_versions(self) -> WindowResult:
        """Require metadata and both versions together.

        Returns:
            Validated window without inventing unknown labels for old records.

        Raises:
            ValueError: If metadata and version fields are incomplete.
        """
        if self.metadata is None:
            if self.metadata_schema_version is not None or self.aggregation_version is not None:
                raise ValueError("metadata versions require metadata")
        elif (self.metadata_schema_version, self.aggregation_version) != (SCHEMA_VERSION, AGGREGATION_VERSION):
            raise ValueError("metadata requires supported schema and aggregation versions")
        return self

    @field_validator("scene_id", "asset_id")
    @classmethod
    def valid_uuid(cls, value):
        if UUID(value).version != 5:
            raise ValueError("scene/source IDs must be UUIDv5")
        return value

    @model_validator(mode="after")
    def samples_match_window(self):
        import math

        if (self.end_ms <= self.start_ms or self.end_ms-self.start_ms > 10000
                or self.start_ms != self.window_index*5000 or self.partial != (self.end_ms - self.start_ms < 10000)):
            raise ValueError("invalid window range/partial flag")
        previous = -1.0
        for i, sample in enumerate(self.samples):
            if sample.sample_index != i or sample.frame.asset_id != self.asset_id:
                raise ValueError("samples must have indices 0..11 and the same asset")
            target = self.start_ms + (i + 0.5) * (self.end_ms - self.start_ms) / 12
            if abs(sample.target_ms - target) > 1e-6:
                raise ValueError("sample target is not a midpoint")
            timestamp = sample.frame.timestamp_ms
            if not self.start_ms <= timestamp < self.end_ms or timestamp < previous:
                raise ValueError("sample PTS must be ordered and inside the window")
            previous = timestamp
        e = self.embedding
        if e.dimension != 512 or not e.normalized or abs(math.sqrt(sum(v*v for v in e.vector))-1) > 1e-3:
            raise ValueError("window vector must be 512-dimensional and L2 normalized")
        return self

    def payload(self, asset: VideoAsset) -> dict:
        """Build the exact searchable payload while preserving legacy keys.

        Args:
            asset: Source record providing optional region and capture time.

        Returns:
            JSON-compatible payload, with v2 metadata only when generated.
        """
        value = self.model_dump(mode="json", exclude={"samples", "embedding", "status", "start_ms", "end_ms", "asset_id", "media_kind"})
        if self.metadata is None:
            for field in ("metadata", "metadata_schema_version", "aggregation_version"):
                value.pop(field, None)
        value.update(
            scene_start_ms=self.start_ms, scene_end_ms=self.end_ms,
            payload_schema_version=PAYLOAD_VERSION if self.metadata else "window_payload_siglip2_only_v1",
            embedding_contract_version="driveretriever_window_epoch4_source_v1",
            duration_ms=self.end_ms-self.start_ms, num_frames=12, media_input="source_unmodified",
            sample_frame_ids=[s.frame.frame_id for s in self.samples],
            sample_target_timestamps_ms=[s.target_ms for s in self.samples],
            sample_source_pts_ms=[s.frame.timestamp_ms for s in self.samples],
        )
        for field in ("captured_at", "region"):
            if getattr(asset, field) is not None:
                value[field] = getattr(asset, field)
        return value
