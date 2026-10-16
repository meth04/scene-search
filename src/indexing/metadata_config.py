"""Versioned SigLIP metadata configuration with no implicit thresholds."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.indexing.domain.entities import stable_hash
from src.indexing.domain.scene_metadata import AGGREGATION_VERSION, SCHEMA_VERSION, AggregationPolicy

WEATHER_PROMPTS = {
    "clear": "a dashcam image of a road in clear weather",
    "cloudy": "a dashcam image of a road in cloudy weather",
    "rain": "a dashcam image of a road in rainy weather",
    "fog": "a dashcam image of a road in foggy weather",
}
LIGHTING_PROMPTS = {
    "day": "a dashcam image of a road during daytime",
    "night": "a dashcam image of a road at night",
}


class SceneMetadataConfig(BaseModel):
    """Pinned model, prompts and mandatory per-group aggregation policies.

    Attributes:
        model_id: Original SigLIP 2 repository.
        revision: Immutable model commit, never a moving branch.
        device: Torch device used by the indexing worker.
        precision: Model inference precision.
        batch_size: Maximum number of RGB frames scored per batch.
        cache_dir: Existing Hugging Face cache directory.
        local_files_only: Whether loading must avoid downloads.
        weather_prompts: Exactly one description per concrete weather label.
        lighting_prompts: Exactly one description per concrete lighting label.
        weather_prompt_version: Version of weather descriptions.
        lighting_prompt_version: Version of lighting descriptions.
        weather_policy: Explicit weather acceptance thresholds.
        lighting_policy: Explicit lighting acceptance thresholds.
        quality_status: Unevaluated configurations cannot publish production indexes.
        evaluation_report_sha256: Checksum of the reviewed quality report, required for evaluated status.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: Literal["google/siglip2-base-patch16-224"] = "google/siglip2-base-patch16-224"
    revision: str = Field(default="75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2", pattern=r"^[a-f0-9]{40}$")
    device: str = "cpu"
    precision: Literal["fp32", "bf16", "fp16"] = "fp32"
    batch_size: int = Field(default=4, ge=1, le=128)
    cache_dir: str = "data/model-cache/huggingface"
    local_files_only: bool = True
    weather_prompts: dict[str, str] = Field(default_factory=lambda: dict(WEATHER_PROMPTS))
    lighting_prompts: dict[str, str] = Field(default_factory=lambda: dict(LIGHTING_PROMPTS))
    weather_prompt_version: str = "weather-prompts-v1"
    lighting_prompt_version: str = "lighting-prompts-v1"
    weather_policy: AggregationPolicy
    lighting_policy: AggregationPolicy
    quality_status: Literal["not_evaluated", "evaluated"] = "not_evaluated"
    evaluation_report_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def valid_prompts(self) -> Self:
        """Keep v1 descriptions fixed and require evidence for evaluated status.

        Returns:
            Configuration with complete, versioned descriptions.

        Raises:
            ValueError: If descriptions violate the contract or evaluation evidence is missing.
        """
        for prompts, baseline, version, initial in (
            (self.weather_prompts, WEATHER_PROMPTS, self.weather_prompt_version, "weather-prompts-v1"),
            (self.lighting_prompts, LIGHTING_PROMPTS, self.lighting_prompt_version, "lighting-prompts-v1"),
        ):
            if set(prompts) != set(baseline) or any(not text.strip() for text in prompts.values()) or not version.strip():
                raise ValueError("metadata descriptions must cover exact concrete labels with a version")
            if version == initial and prompts != baseline:
                raise ValueError("changing metadata descriptions requires a new prompt version")
        if self.quality_status == "evaluated" and not self.evaluation_report_sha256:
            raise ValueError("evaluated metadata requires an evaluation report checksum")
        return self

    def contract(self) -> dict:
        """Snapshot configuration and versions for indexing and pinned searches.

        Returns:
            JSON-compatible contract including its reproducible hash.
        """
        value = dict(config=self.model_dump(mode="json"), metadata_schema_version=SCHEMA_VERSION,
                     aggregation_version=AGGREGATION_VERSION)
        return value | {"hash": stable_hash(value)}
