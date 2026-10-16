from __future__ import annotations

from datetime import datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.indexing.domain.scene_metadata import LIGHTING_LABELS, WEATHER_LABELS, LightingLabel, WeatherLabel
from src.retrieval.domain.errors import ForbiddenError, RetrievalError

IDENTITY_FIELDS = ("tenant_id", "index_version", "embedding_contract_hash", "privacy_status", "media_policy_version")
MEMBERSHIP_FIELDS = (
    ("video_ids", "video_id"),
    ("camera_ids", "camera"),
    ("regions", "region"),
    ("dataset_versions", "dataset_version"),
)


class WindowFilter(BaseModel):
    """Trusted tenant/scope boundaries and supported optional metadata predicates.

    Attributes:
        weather: OR selection of weather labels; an empty list leaves weather unrestricted.
        lighting: OR selection of lighting labels, combined with weather using AND.
        access_scope_ids: Mandatory source membership injected from the authenticated actor.
        index_version: Exact index retained by the search session.
    """

    model_config = ConfigDict(extra="forbid")
    tenant_id: str
    access_scope_ids: list[str] = Field(min_length=1)
    index_version: str
    embedding_contract_hash: str
    video_ids: list[str] = Field(default_factory=list, max_length=100)
    camera_ids: list[str] = Field(default_factory=list, max_length=100)
    dataset_versions: list[str] = Field(default_factory=list, max_length=100)
    regions: list[str] = Field(default_factory=list, max_length=100)
    weather: list[WeatherLabel] = Field(default_factory=list, max_length=100)
    lighting: list[LightingLabel] = Field(default_factory=list, max_length=100)
    start_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    start_time: datetime | None = None
    end_time: datetime | None = None
    privacy_status: str = "not_processed"
    media_policy_version: str = "source_unmodified_internal_v1"

    @model_validator(mode="after")
    def valid_interval(self):
        """Require timezone-aware capture bounds and a valid video interval.

        Returns:
            This validated filter without changing its bounds.

        Raises:
            ValueError: If capture times lack a timezone, video bounds are
                incomplete or either interval is reversed or empty.
        """
        if any(value is not None and value.tzinfo is None for value in (self.start_time, self.end_time)):
            raise ValueError("captured_at filters require a time zone")
        if (self.start_ms is None) != (self.end_ms is None) or (
            self.start_ms is not None and self.end_ms <= self.start_ms
        ):
            raise ValueError("time filter must specify 0 <= start_ms < end_ms")
        if self.start_time and self.end_time and self.start_time >= self.end_time:
            raise ValueError("invalid captured_at interval")
        return self

    @classmethod
    def from_request(cls, value: dict, actor: Any, catalog: dict) -> Self:
        """Validate requested filters and inject trusted actor and catalog boundaries.

        Args:
            value: Record or request value to inspect.
            actor: Authenticated actor, tenant and current source access scopes.
            catalog: Pinned index catalog, capabilities and embedding contract.

        Returns:
            Validated window filter.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
            RetrievalError: If the query, filter or export request violates its supported contract.
        """
        if not actor.access_scope_ids:
            raise ForbiddenError("empty source access scope")
        data = validate_filter_request(value, catalog)
        try:
            return cls(
                **data,
                tenant_id=actor.tenant_id,
                access_scope_ids=actor.access_scope_ids,
                index_version=catalog["index_version"],
                embedding_contract_hash=catalog["embedding_contract_hash"],
            )
        except ValueError as exc:
            raise RetrievalError("invalid filter", details={"error_code": "FILTER_INVALID"}) from exc

    def matches(self, payload: dict) -> bool:
        """Check a local payload against the same predicates sent to Qdrant.

        Args:
            payload: Stored window metadata with capture times as ISO strings.

        Returns:
            Whether identity, membership and half-open time bounds all match.

        Raises:
            ValueError: If a filtered capture timestamp cannot be parsed.
        """
        for key in IDENTITY_FIELDS:
            if payload.get(key) != getattr(self, key):
                return False
        if payload.get("access_scope_id") not in self.access_scope_ids:
            return False
        for attribute, field in MEMBERSHIP_FIELDS:
            allowed = getattr(self, attribute)
            if allowed and payload.get(field) not in allowed:
                return False
        metadata = payload.get("metadata")
        for group in ("weather", "lighting"):
            allowed = getattr(self, group)
            if allowed:
                prediction = metadata.get(group) if isinstance(metadata, dict) else None
                if not isinstance(prediction, dict) or prediction.get("value") not in allowed:
                    return False
        if self.start_ms is not None and not (
            payload.get("scene_start_ms", float("inf")) < self.end_ms
            and payload.get("scene_end_ms", -1) > self.start_ms
        ):
            return False
        return self._matches_capture_time(payload.get("captured_at"))

    def _matches_capture_time(self, captured_at):
        """Apply inclusive start and exclusive end bounds to a capture time.

        Args:
            captured_at: ISO timestamp or a missing payload value.

        Returns:
            True when no capture filter is set or the timestamp is within it.

        Raises:
            ValueError: If the supplied timestamp cannot be parsed.
        """
        if not (self.start_time or self.end_time):
            return True
        if not captured_at:
            return False
        captured = datetime.fromisoformat(captured_at.replace("Z", "+00:00"))
        if self.start_time and captured < self.start_time:
            return False
        if self.end_time and captured >= self.end_time:
            return False
        return True

    def qdrant(self) -> Any:
        """Translate the window filter to equivalent Qdrant field conditions.

        Returns:
            Qdrant filter with mandatory identity and half-open time bounds.
        """
        from qdrant_client import models

        conditions = qdrant_identity_conditions(self)
        for group in ("weather", "lighting"):
            values = getattr(self, group)
            if values:
                conditions.append(models.FieldCondition(key=f"metadata.{group}.value", match=models.MatchAny(any=values)))
        if self.start_ms is not None:
            conditions.extend(
                [
                    models.FieldCondition(key="scene_start_ms", range=models.Range(lt=self.end_ms)),
                    models.FieldCondition(key="scene_end_ms", range=models.Range(gt=self.start_ms)),
                ]
            )
        if self.start_time or self.end_time:
            conditions.append(
                models.FieldCondition(
                    key="captured_at", range=models.DatetimeRange(gte=self.start_time, lt=self.end_time)
                )
            )
        return models.Filter(must=conditions)


def validate_filter_request(value: dict | None, catalog: dict) -> dict:
    """Reject unsupported filter keys and unavailable catalog capabilities.

    Args:
        value: Record or request value to inspect.
        catalog: Pinned index catalog, capabilities and embedding contract.

    Returns:
        Copied filter request containing only permitted fields.

    Raises:
        RetrievalError: If the query, filter or export request violates its supported contract.
    """
    if value is not None and not isinstance(value, dict):
        raise RetrievalError("filters must be an object", details={"error_code": "FILTER_INVALID"})
    data = dict(value or {})
    capability_keys = {
        "video_ids": "video",
        "camera_ids": "camera",
        "regions": "region",
        "start_time": "captured_at",
        "end_time": "captured_at",
    }
    for key in ("objects", "object_classes"):
        if key in data:
            raise RetrievalError(
                "this filter has no generated metadata", details={"error_code": "FILTER_NOT_AVAILABLE", "filter": key}
            )
    allowed = {"video_ids", "camera_ids", "regions", "dataset_versions", "start_ms", "end_ms", "start_time", "end_time", "weather", "lighting"}
    if set(data) - allowed:
        raise RetrievalError("unsupported filter", details={"error_code": "FILTER_INVALID"})
    for group, labels in (("weather", WEATHER_LABELS), ("lighting", LIGHTING_LABELS)):
        if group not in data:
            continue
        values = data[group]
        if (not isinstance(values, list) or len(values) > 100
                or any(not isinstance(v, str) or v not in (*labels, "unknown") for v in values)):
            raise RetrievalError("invalid metadata filter", details={"error_code": "FILTER_INVALID", "filter": group})
        if values and catalog.get("capabilities", {}).get(group) != "generated":
            raise RetrievalError("metadata not generated", details={"error_code": "FILTER_NOT_AVAILABLE", "filter": group})
        available = catalog.get("metadata_filter_values", {}).get(group, [])
        if values and (not isinstance(available, list) or any(v not in available for v in values)):
            raise RetrievalError("metadata value unavailable in this index", details={"error_code": "FILTER_NOT_AVAILABLE", "filter": group})
    for key, capability in capability_keys.items():
        if data.get(key) and not catalog.get("capabilities", {}).get(capability):
            raise RetrievalError(
                "filter is unavailable in this index", details={"error_code": "FILTER_NOT_AVAILABLE", "filter": key}
            )
    return data


def qdrant_identity_conditions(filters):
    """Build mandatory identity and optional membership predicates for Qdrant.

    Args:
        filters: Validated tenant, scope and metadata filter.

    Returns:
        Field conditions enforcing tenant, scope, index, contract and selected metadata.
    """
    from qdrant_client import models

    conditions = [
        models.FieldCondition(key=key, match=models.MatchValue(value=getattr(filters, key))) for key in IDENTITY_FIELDS
    ]
    for attribute, key in (("access_scope_ids", "access_scope_id"), *MEMBERSHIP_FIELDS):
        values = getattr(filters, attribute)
        if values:
            conditions.append(models.FieldCondition(key=key, match=models.MatchAny(any=values)))
    return conditions
