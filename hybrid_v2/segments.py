"""Validated source segments shared by indexing and explicit privacy resets."""

from __future__ import annotations

import json
import math
from pathlib import Path


def load_segments(manifest, duration, fps):
    value = json.loads(Path(manifest).read_text(encoding="utf-8")) if isinstance(manifest, (str, Path)) else manifest
    value = value.get("segments") if isinstance(value, dict) else value
    if not isinstance(value, list) or not value:
        raise ValueError("Segment manifest must contain a nonempty segment list")
    result, seen = [], set()
    prior = 0.0
    for i, row in enumerate(value):
        start, end = row.get("global_start_sec"), row.get("global_end_sec")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in (start, end)):
            raise ValueError("Segment times must be finite numbers")
        if abs(start - prior) > 0.5 / fps or end <= start or end > duration + 0.5 / fps:
            raise ValueError("Segments must cover the timeline contiguously without overlap")
        if any(abs(v * fps - round(v * fps)) > 1e-3 for v in (start, end)):
            raise ValueError("Segment boundaries must align with source frames")
        identity = str(row.get("segment_id") or row.get("sequence_id") or f"segment_{i:06d}")
        if identity in seen:
            raise ValueError("Duplicate segment id")
        seen.add(identity)
        result.append(
            {
                **row,
                "segment_id": identity,
                "source_clip_id": row.get("source_clip_id") or row.get("hf_path") or identity,
                "global_start_sec": float(start),
                "global_end_sec": float(end),
            }
        )
        prior = end
    if abs(prior - duration) > 0.5 / fps:
        raise ValueError("Segment manifest does not cover the complete source")
    return result


def boundaries(manifest, duration, fps):
    return [int(round(s["global_start_sec"] * fps)) for s in load_segments(manifest, duration, fps)[1:]]


def validation_artifact(windows, segments, samples=12):
    from .common import chronological_frame_ids

    results = []
    by_id = {s["segment_id"]: s for s in segments}
    for w in windows:
        s = by_id[w["segment_id"]]
        sf, ef = round(s["global_start_sec"] * w["fps"]), round(s["global_end_sec"] * w["fps"])
        ids = chronological_frame_ids(w["start_frame"], w["end_frame"], samples).tolist()
        ok = sf <= w["start_frame"] < w["end_frame"] <= ef and all(sf <= i < ef for i in ids)
        results.append(
            {
                "row": w["row"],
                "segment_id": s["segment_id"],
                "source_clip_id": s["source_clip_id"],
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "sample_frame_ids": ids,
                "single_source": ok,
            }
        )
    return {
        "status": "PASS" if all(r["single_source"] for r in results) else "FAIL",
        "windows": len(windows),
        "source_segments": len(segments),
        "cross_source_windows": sum(not r["single_source"] for r in results),
        "rows": results,
    }
