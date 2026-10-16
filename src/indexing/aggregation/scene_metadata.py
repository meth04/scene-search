"""Pure independent label decisions over the twelve source-frame scores."""

import math

from src.indexing.domain.scene_metadata import (
    AggregationPolicy,
    FrameMetadataScores,
    LightingPrediction,
    SceneMetadata,
    WeatherPrediction,
)
from src.indexing.metadata_config import SceneMetadataConfig


def _aggregate_group(scores: list[dict[str, float]], policy: AggregationPolicy) -> tuple[str, float]:
    """Accept a unique winner when all thresholds are met within 1e-12 rounding.

    Args:
        scores: Validated concrete scores in source-frame order.
        policy: Minimum mean score, margin and frame agreement.

    Returns:
        Accepted label or unknown, plus the highest mean similarity score.
    """
    means = {label: math.fsum(frame[label] for frame in scores) / len(scores) for label in scores[0]}
    ordered = sorted(means, key=lambda label: (-means[label], label))
    winner, runner_up = ordered[:2]
    confidence = means[winner]
    margin = confidence - means[runner_up]
    agreement = sum(all(frame[winner] > value for label, value in frame.items() if label != winner)
                    for frame in scores) / len(scores)
    thresholds_met = all(value >= minimum or math.isclose(value, minimum, rel_tol=0, abs_tol=1e-12)
                         for value, minimum in ((confidence, policy.min_score), (margin, policy.min_margin),
                                                (agreement, policy.min_agreement)))
    accepted = margin > 0 and thresholds_met
    return (winner if accepted else "unknown"), confidence


def aggregate_scene_metadata(scores: list[FrameMetadataScores], config: SceneMetadataConfig) -> SceneMetadata:
    """Aggregate exactly twelve frames without inference or storage effects.

    Args:
        scores: Complete per-frame scores in sampled source order.
        config: Pinned provenance and explicit independent group policies.

    Returns:
        Weather and lighting predictions, independently falling back to unknown.

    Raises:
        ValueError: If there are not exactly twelve valid score records.
    """
    if len(scores) != 12:
        raise ValueError("scene metadata requires exactly twelve frame scores")
    checked = [FrameMetadataScores.model_validate(s.model_dump() if isinstance(s, FrameMetadataScores) else s) for s in scores]
    weather, weather_score = _aggregate_group([s.weather for s in checked], config.weather_policy)
    lighting, lighting_score = _aggregate_group([s.lighting for s in checked], config.lighting_policy)
    provenance = dict(model_id=config.model_id, model_revision=config.revision)
    return SceneMetadata(
        weather=WeatherPrediction(value=weather, confidence=weather_score,
                                  prompt_version=config.weather_prompt_version, **provenance),
        lighting=LightingPrediction(value=lighting, confidence=lighting_score,
                                    prompt_version=config.lighting_prompt_version, **provenance),
    )
