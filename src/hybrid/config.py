from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from src.indexing.config import load_structured


class HybridConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index_root: str = "outputs/local_hybrid_indexes"
    job_root: str = "data/hybrid-api-jobs"
    source_roots: list[str] = Field(default_factory=lambda: ["data/videos", "data/test-video"])
    tenant_id: str = "default"
    access_scope_id: str = "default"
    stage2b_checkpoint: str = "outputs/driveretriever_stage2b_vision/best_composite.pt"
    stage3a_checkpoint: str = "outputs/driveretriever_stage3a/best_composite.pt"
    stage5_checkpoint: str = "outputs/driveretriever_hybrid_v2/stage5/best_stage5.pt"
    hf_cache_dir: str = "data/model-cache/huggingface"
    siglip_revision: str | None = None
    bge_model: str = "BAAI/bge-base-en-v1.5"
    bge_revision: str | None = None
    detector: str = "data/model-cache/yolo11n.pt"
    qwen_mode: Literal["local", "server"] | None = None
    qwen_config: str | None = None
    qwen_base_url: str | None = None
    vlm_model: str | None = None
    qwen_server_base_url: str | None = None
    qwen_server_model: str | None = None
    device: Literal["cpu", "cuda"] = "cpu"
    precision: Literal["fp32", "fp16"] = "fp32"
    ranker: Literal["caption", "stage3", "base", "stage5", "dense", "dense-entity", "embeddinggemma2", "driveevidence", "driveevidence_v2", "driveevidence_v3", "driveevidence_v4", "driveevidence_v5", "driveevidence_v6"] = "stage5"
    cpu_threads: int = Field(default=4, ge=1, le=32)
    search_cache_entries: int = Field(default=2, ge=1, le=8)
    decision_provider: Literal["none", "laya"] | None = None
    deployment_mode: Literal["eg2_only", "shadow", "active"] | None = None
    laya_device: Literal["auto", "cpu", "cuda"] | None = None
    laya_model_path: str | None = None
    laya_batch_size: int | None = Field(default=None, ge=1, le=15)
    laya_candidate_count: int | None = Field(default=None, ge=1, le=15)
    privacy_on_ingest: bool = False
    privacy_profile: Literal["auto", "cpu", "gtx1650", "rtx3090"] = "auto"


def load_hybrid_config(path: str | Path) -> HybridConfig:
    return HybridConfig.model_validate(load_structured(path))
