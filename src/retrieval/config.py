from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from src.indexing.config import DriveRetrieverConfig, WorkflowConfig, load_structured


class SearchConfig(BaseModel):
    production_alias: str = "scenes_production"
    page_size_max: int = Field(default=100, ge=1)
    candidate_k_max: int = Field(default=100, ge=1)
    score_threshold: float | None = None
    cursor_ttl_seconds: int = Field(default=3600, gt=0)
    ranking_config_version: str = "baseline-v1"


class PreviewConfig(BaseModel):
    url_ttl_seconds: int = Field(default=300, gt=0)


class ExportSettings(BaseModel):
    formats: list[Literal["manifest_only", "clips_and_manifest"]] = ["manifest_only"]
    artifact_ttl_seconds: int = Field(default=86400, gt=0)
    download_ttl_seconds: int = Field(default=60, gt=0)
    max_items: int = Field(default=1000, gt=0)
    max_total_duration_ms: int = Field(default=3600000, gt=0)
    staging_prefix: str = "exports/staging"
    published_prefix: str = "exports/published"


class RetrievalConfig(BaseModel):
    """DriveRetriever settings, source access, query limits and export configuration."""

    model_family: Literal["driveretriever"] = "driveretriever"
    driveretriever: DriveRetrieverConfig = Field(default_factory=DriveRetrieverConfig)
    workflow: WorkflowConfig = Field(default_factory=WorkflowConfig)
    initial_candidates: int = Field(default=300, ge=1, le=1200)
    maximum_candidates: int = Field(default=1200, ge=1, le=1200)
    result_limit: int = Field(default=300, ge=1, le=300)
    nms_iou: float | None = Field(default=0.25, ge=0, le=1)
    query_cache_ttl_seconds: int = Field(default=300, gt=0)
    query_cache_max_entries: int = Field(default=256, gt=0)
    max_concurrent_searches: int = Field(default=4, ge=1, le=32)
    candidate_budget_seconds: int = Field(default=15, ge=1, le=60)
    contract_version: Literal["1.0"] = "1.0"
    provider: Literal["local", "production"] = "local"
    root_dir: str = "data/indexing"
    cursor_signing_key: str = "local-development-only-key"
    search: SearchConfig = Field(default_factory=SearchConfig)
    preview: PreviewConfig = Field(default_factory=PreviewConfig)
    export: ExportSettings = Field(default_factory=ExportSettings)

    @model_validator(mode="after")
    def safe_driveretriever(self):
        if self.maximum_candidates < self.initial_candidates:
            raise ValueError("maximum candidates must be at least initial candidates")
        if self.model_family == "driveretriever" and self.provider == "production":
            if not self.driveretriever.source_registry_path:
                raise ValueError("production requires trusted source registration")
            if self.driveretriever.provider == "fake" or self.workflow.provider != "temporal":
                raise ValueError("production requires real encoder and Temporal")
            if self.cursor_signing_key == "local-development-only-key" or len(self.cursor_signing_key) < 32:
                raise ValueError("production requires a private cursor signing key of at least 32 characters")
        return self


def load_retrieval_config(path: str | Path) -> RetrievalConfig:
    import os

    value = load_structured(path)
    if os.getenv("RETRIEVAL_CURSOR_SIGNING_KEY"):
        value["cursor_signing_key"] = os.environ["RETRIEVAL_CURSOR_SIGNING_KEY"]
    return RetrievalConfig.model_validate(value)
