"""Request identity, export settings and object-store metadata shared by window services."""

from typing import Literal

from pydantic import BaseModel, Field


class ActorContext(BaseModel):
    actor_id: str
    tenant_id: str = "default"
    access_scope_ids: list[str] = Field(default_factory=lambda: ["default"])
    roles: list[str] = Field(default_factory=list)


class ExportConfig(BaseModel):
    format: Literal["manifest_only", "clips_and_manifest"] = "manifest_only"
    manifest_format: Literal["jsonl", "csv"] = "jsonl"


class ObjectInfo(BaseModel):
    key: str
    size_bytes: int
    sha256: str
    metadata: dict[str, str] = Field(default_factory=dict)
