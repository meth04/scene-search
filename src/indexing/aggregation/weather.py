from collections import defaultdict

from src.indexing.domain.entities import ClassProbabilities, WeatherMetadata


def aggregate_weather(
    results: list[ClassProbabilities], confidence_threshold: float
) -> WeatherMetadata:
    sums: dict[str, float] = defaultdict(float)
    for result in results:
        for label, probability in result.probabilities.items():
            sums[label] += probability
    if not sums:
        return WeatherMetadata(weather="unknown", confidence=0, lighting="unknown")
    denominator = len(results)
    averages = {label: value / denominator for label, value in sums.items()}
    label, confidence = max(averages.items(), key=lambda item: item[1])
    weather = label if confidence >= confidence_threshold else "unknown"
    lighting = "night" if averages.get("night", 0) >= confidence_threshold else "unknown"
    return WeatherMetadata(weather=weather, confidence=confidence, lighting=lighting)
