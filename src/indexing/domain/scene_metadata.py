"""Typed scene labels and finite scores, independent of models and storage."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

WeatherLabel = Literal["clear", "cloudy", "rain", "fog", "unknown"]
LightingLabel = Literal["day", "night", "unknown"]
WEATHER_LABELS = ("clear", "cloudy", "rain", "fog")
LIGHTING_LABELS = ("day", "night")
SCHEMA_VERSION = "scene_metadata_v1"
AGGREGATION_VERSION = "metadata-aggregation-v1"
PAYLOAD_VERSION = "window_payload_scene_metadata_v2"


class AggregationPolicy(BaseModel):
    """Explicit thresholds for accepting a window label.

    Attributes:
        min_score: Minimum mean similarity score, in [0, 1].
        min_margin: Minimum difference between the two highest means.
        min_agreement: Minimum fraction of frames with a unique winning label.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    min_score: float = Field(ge=0, le=1, allow_inf_nan=False)
    min_margin: float = Field(ge=0, le=1, allow_inf_nan=False)
    min_agreement: float = Field(ge=0, le=1, allow_inf_nan=False)


class FrameMetadataScores(BaseModel):
    """Scores for one RGB frame without an unknown model label.

    Attributes:
        weather: All four concrete weather labels and their scores in [0, 1].
        lighting: Both concrete lighting labels and their scores in [0, 1].
    """

    model_config = ConfigDict(extra="forbid")
    weather: dict[str, float]
    lighting: dict[str, float]

    @model_validator(mode="after")
    def valid_scores(self) -> Self:
        """Require complete groups and finite bounded scores.

        Returns:
            Validated frame scores, preserving label order.

        Raises:
            ValueError: If labels are incomplete or a score is invalid.
        """
        import math

        for scores, labels in ((self.weather, WEATHER_LABELS), (self.lighting, LIGHTING_LABELS)):
            if set(scores) != set(labels) or any(not math.isfinite(v) or not 0 <= v <= 1 for v in scores.values()):
                raise ValueError("metadata scores require exact labels and finite values in [0, 1]")
        return self


class MetadataPrediction(BaseModel):
    """Accepted label or unknown with reproducible inference provenance.

    Attributes:
        value: Label chosen by aggregation, rather than by the model directly.
        confidence: Highest mean similarity score; not a calibrated probability.
        model_id: Model repository used for scoring.
        model_revision: Immutable 40-character model revision.
        prompt_version: Version of the group's English descriptions.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    value: str
    confidence: float = Field(ge=0, le=1, allow_inf_nan=False)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(pattern=r"^[a-f0-9]{40}$")
    prompt_version: str = Field(min_length=1)


class WeatherPrediction(MetadataPrediction):
    """Prediction restricted to the five weather values.

    Attributes:
        value: Weather label; remaining fields retain prediction provenance.
    """

    value: WeatherLabel


class LightingPrediction(MetadataPrediction):
    """Prediction restricted to day, night or unknown.

    Attributes:
        value: Lighting label; remaining fields retain prediction provenance.
    """

    value: LightingLabel


class SceneMetadata(BaseModel):
    """Independent weather and lighting predictions for one source window.

    Attributes:
        weather: Weather prediction and provenance.
        lighting: Lighting prediction and provenance.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    weather: WeatherPrediction
    lighting: LightingPrediction
