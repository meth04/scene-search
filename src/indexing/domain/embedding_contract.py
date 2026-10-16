"""Shared, immutable description of the exact video/text embedding pipeline."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.indexing.domain.entities import stable_hash


class DriveRetrieverContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    contract_version: Literal["driveretriever_window_epoch4_source_v1"] = "driveretriever_window_epoch4_source_v1"
    model_family: Literal["driveretriever"] = "driveretriever"
    model_id: Literal["google/siglip2-base-patch16-224"] = "google/siglip2-base-patch16-224"
    requested_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    resolved_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    checkpoint_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    backbone_sha256: dict[str, str] = Field(default_factory=dict)
    architecture: dict[str, Any]
    processor: dict[str, Any]
    tokenizer: dict[str, Any]
    runtime: dict[str, Any]
    runtime_lock_hash: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    max_text_length: int = Field(default=64, gt=0)
    text_projection: Literal["heads.text_proj"] = "heads.text_proj"
    video_projection: Literal["heads.clip_proj"] = "heads.clip_proj"
    feature_api: Literal["siglip2-pooler-output"] = "siglip2-pooler-output"
    video_feature_api: Literal["DriveRetriever.encode_video"] = "DriveRetriever.encode_video"
    text_feature_api: Literal["DriveRetriever.encode_text"] = "DriveRetriever.encode_text"
    text_padding: Literal["max_length"] = "max_length"
    query_overlength_policy: Literal["reject_before_truncation"] = "reject_before_truncation"
    query_prompt_template: Literal["identity"] = "identity"
    nearest_frame_policy: Literal["nearest_pts_tie_earlier"] = "nearest_pts_tie_earlier"
    embedding_dimension: Literal[512] = 512
    storage_dtype: Literal["float32"] = "float32"
    normalization: Literal["l2"] = "l2"
    distance_metric: Literal["cosine"] = "cosine"
    num_frames: Literal[12] = 12
    window_ms: Literal[10000] = 10000
    stride_ms: Literal[5000] = 5000
    segmentation_version: Literal["sliding_10s_stride5s_v1"] = "sliding_10s_stride5s_v1"
    sampling_version: Literal["uniform_bin_centers_12_v1"] = "uniform_bin_centers_12_v1"
    preprocessing_version: Literal["source-rgb-siglip2-v1"] = "source-rgb-siglip2-v1"
    query_language_policy: Literal["direct"] = "direct"
    query_normalizer_version: Literal["unicode-nfc-whitespace-v1"] = "unicode-nfc-whitespace-v1"
    payload_schema_version: Literal["window_payload_siglip2_only_v1"] = "window_payload_siglip2_only_v1"
    media_policy_version: Literal["source_unmodified_internal_v1"] = "source_unmodified_internal_v1"
    is_fake: bool = False

    @model_validator(mode="after")
    def pinned_artifacts(self):
        if self.requested_revision != self.resolved_revision:
            raise ValueError("resolved backbone differs from the pinned revision")
        if not self.is_fake and not self.backbone_sha256:
            raise ValueError("real contracts require backbone artifact checksums")
        if not self.is_fake and not self.runtime_lock_hash:
            raise ValueError("real contracts require the dependency lock checksum")
        return self

    @property
    def hash(self) -> str:
        return stable_hash(self)
