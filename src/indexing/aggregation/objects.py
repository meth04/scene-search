from collections import Counter, defaultdict

from src.indexing.domain.entities import Detection, ObjectMetadata


def aggregate_objects(frames: list[list[Detection]]) -> ObjectMetadata:
    totals: Counter[str] = Counter()
    maximums: dict[str, int] = defaultdict(int)
    for detections in frames:
        counts = Counter(item.label for item in detections)
        totals.update(counts.keys())
        for label, count in counts.items():
            maximums[label] = max(maximums[label], count)
    denominator = max(1, len(frames))
    labels = sorted(totals)
    return ObjectMetadata(
        classes=labels,
        max_counts={label: maximums[label] for label in labels},
        presence_ratio={label: totals[label] / denominator for label in labels},
    )
