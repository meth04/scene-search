from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from src.retrieval.domain.entities import ExportConfig
from src.retrieval.domain.states import ReviewStatus


class CreateSessionCommand(BaseModel):
    title: str = ""
    index_id: str | None = None


class SearchCommand(BaseModel):
    session_id: str
    index_id: str = "scenes_production"
    request_key: str
    query_text: str = Field(min_length=1, max_length=1000)
    filters: dict[str, Any] = Field(default_factory=dict)
    page_size: int = Field(default=20, ge=1, le=100)


class ReviewCommand(BaseModel):
    result_id: str
    status: ReviewStatus
    expected_version: int = Field(gt=0)
    note: str | None = Field(default=None, max_length=2000)
    request_id: str
    subject_version: str | None = None
    start_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class CreateResultSetCommand(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ResultSetCommand(BaseModel):
    result_set_id: str
    result_id: str
    operation: Literal["add", "remove"]
    expected_version: int = Field(gt=0)


class ExportCommand(BaseModel):
    result_set_id: str
    expected_version: int = Field(gt=0)
    idempotency_key: str = Field(min_length=1, max_length=200)
    request_id: str
    config: ExportConfig = Field(default_factory=ExportConfig)
