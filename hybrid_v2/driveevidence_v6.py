"""Frozen V6 evidence construction and representation studies (no final policy)."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from .embeddinggemma2 import DIMENSION, REVISION, digest, identity

FORMAT = "driveevidence-v6"
FACTOR_SCHEMA = "driveevidence-v6-atoms-1.1"
REPRESENTATION = {
    "frame_count": 8, "frame_sampling": "uniform_endpoints",
    "window_length": 2.0, "window_stride": 1.5, "window_frames": 3,
    "tube_frames": 4, "tube_context_scale": 1.75, "max_tubes": 3,
    "track_min_observations": 4, "track_min_duration": .5,
    "track_min_confidence": .3, "track_min_area": .001,
    "visual_tokens": 140,
}
COMPATIBILITY = {
    "GLOBAL": ("GLOBAL", "WINDOW"),
    "OBJECT": ("FRAME", "TUBE", "WINDOW"),
    "ACTION": ("WINDOW", "TUBE"),
    "SPATIAL": ("TUBE", "WINDOW", "FRAME"),
    "SCENE": ("GLOBAL", "WINDOW", "FRAME"),
    "STATE_A": ("WINDOW", "FRAME"), "STATE_B": ("WINDOW", "FRAME"),
}


def validate_representation(cfg):
    if set(cfg) != set(REPRESENTATION):
        raise ValueError("V6_REPRESENTATION_FIELDS")
    for key, lo, hi in (("frame_count", 1, 16), ("window_frames", 2, 8),
                        ("tube_frames", 3, 8), ("max_tubes", 1, 6), ("track_min_observations", 2, 50)):
        if type(cfg[key]) is not int or not lo <= cfg[key] <= hi:
            raise ValueError("V6_REPRESENTATION_BOUND " + key)
    for key, lo, hi in (("window_length", .5, 5), ("window_stride", .25, 5),
                        ("tube_context_scale", 1., 4.), ("track_min_duration", 0., 5.),
                        ("track_min_confidence", 0., 1.), ("track_min_area", 0., 1.)):
        if not isinstance(cfg[key], (int, float)) or not math.isfinite(cfg[key]) or not lo <= cfg[key] <= hi:
            raise ValueError("V6_REPRESENTATION_BOUND " + key)
    if cfg["frame_sampling"] != "uniform_endpoints" or cfg["visual_tokens"] not in (70, 140, 280):
        raise ValueError("V6_REPRESENTATION_POLICY")
    temporal_windows(5., cfg["window_length"], cfg["window_stride"])
    return cfg


def sample_positions(timestamps, count, start=None, end=None):
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("INVALID_CHRONOLOGICAL_TIMESTAMPS")
    if count < 1:
        raise ValueError("INVALID_FRAME_COUNT")
    lo, hi = times[0] if start is None else start, times[-1] if end is None else end
    eligible = np.flatnonzero((times >= lo - 1e-8) & (times <= hi + 1e-8))
    if len(eligible) < count:
        raise ValueError("INSUFFICIENT_DISTINCT_FRAMES")
    targets = np.linspace(times[eligible[0]], times[eligible[-1]], count)
    positions = [int(eligible[np.argmin(np.abs(times[eligible] - target))]) for target in targets]
    if len(set(positions)) != count:
        positions = eligible[np.linspace(0, len(eligible) - 1, count, dtype=int)].tolist()
    return positions


def temporal_windows(duration, length, stride):
    if not all(math.isfinite(x) and x > 0 for x in (duration, length, stride)) or length > duration:
        raise ValueError("INVALID_WINDOW_CONFIG")
    starts = list(np.arange(0, duration - length + 1e-8, stride))
    if abs(starts[-1] - (duration - length)) > 1e-6:
        starts.append(duration - length)
    if len(starts) > 12:
        raise ValueError("UNBOUNDED_WINDOWS")
    return [(round(float(s), 6), round(float(s + length), 6)) for s in starts]


def expanded_box(box, width, height, scale):
    if not math.isfinite(scale) or scale < 1 or scale > 4:
        raise ValueError("INVALID_CONTEXT_SCALE")
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(x) for x in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        raise ValueError("INVALID_BBOX")
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    halfw, halfh = (x2 - x1) * scale / 2, (y2 - y1) * scale / 2
    result = [max(0, int(math.floor(cx - halfw))), max(0, int(math.floor(cy - halfh))),
              min(width, int(math.ceil(cx + halfw))), min(height, int(math.ceil(cy + halfh)))]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("EMPTY_CONTEXT_CROP")
    return result


def select_tracks(tracks, cfg):
    selected = []
    for track in tracks:
        observations = sorted(track["observations"], key=lambda o: (o["timestamp"], o["frame_position"]))
        # Observations, not interpolated/fabricated detections, construct tubes.
        if len(observations) < max(cfg["track_min_observations"], cfg["tube_frames"]):
            continue
        duration = observations[-1]["timestamp"] - observations[0]["timestamp"]
        confidence = float(np.mean([o["confidence"] for o in observations]))
        area = float(np.mean([o["xywh"][2] * o["xywh"][3] for o in observations]))
        if duration < cfg["track_min_duration"] or confidence < cfg["track_min_confidence"] or area < cfg["track_min_area"]:
            continue
        visibility = len(observations) / 50.0
        quality = confidence * min(duration / 3, 1) * min(visibility / .5, 1) * math.sqrt(min(area / .02, 1))
        selected.append((quality, int(track["slot"]), {**track, "observations": observations, "quality": quality}))
    selected.sort(key=lambda x: (-x[0], x[1]))
    return [t for _, _, t in selected[:cfg["max_tubes"]]]


def tube_spec(track, width, height, cfg):
    obs = track["observations"]
    positions = sample_positions([o["timestamp"] for o in obs], cfg["tube_frames"])
    chosen = [obs[p] for p in positions]
    return {"family": "TUBE", "track_id": int(track["slot"]), "detector_class": track["class"],
            "indices": [int(o["frame_position"]) for o in chosen],
            "timestamps": [float(o["timestamp"]) for o in chosen],
            "boxes": [expanded_box(o["bbox"], width, height, cfg["tube_context_scale"]) for o in chosen],
            "original_boxes": [o["bbox"] for o in chosen], "quality": track["quality"],
            "start": float(chosen[0]["timestamp"]), "end": float(chosen[-1]["timestamp"])}


def complexity(query):
    q = query.casefold()
    signals = {
        "action": bool(re.search(r"\b(overtak\w*|cross\w*|pass\w*|turn\w*|approach\w*|leav\w*|enter\w*|mov\w*|walk\w*|ride\w*)\b", q)),
        "relation": bool(re.search(r"\b(left of|right of|beside|next to|behind|in front of)\b", q)),
        "spatial": bool(re.search(r"\b(left|right|ahead|shoulder|foreground|background)\b", q)),
        "attribute": bool(re.search(r"\b(white|black|red|blue|green|yellow|silver|gray|orange|shirt|clothing|small|sign|light)\b", q)),
        "temporal": bool(re.search(r"\b(then|before|after|while|from .+ to|briefly|disappear\w*)\b", q)),
    }
    return {"route": "complex" if any(signals.values()) else "simple", "signals": signals}


def factor_atoms(query, plan, template="concise", temporal=True):
    """Phrases grounded in query-only frozen Laya fields; no candidate inputs."""
    atoms = [{"type": "GLOBAL", "text": query, "weight": 1.0}]
    if plan is None:
        return atoms
    entities = plan.get("entities", [])
    subject = entities[0].get("object_phrase", "vehicle") if entities else "vehicle"
    if entities:
        atoms.append({"type": "OBJECT", "text": subject, "weight": 1.0})
    action = plan.get("action")
    if isinstance(action, dict):
        action = action.get("value")
    if action and action != "unknown":
        verb = str(action).replace("_", " ")
        generic_subject = "person" if entities and entities[0].get("type") == "person" else "vehicle"
        action_subject = subject if template == "bound" else generic_subject
        atoms.append({"type": "ACTION", "text": f"{action_subject} {verb}", "weight": 1.0})
    graph, side = plan.get("relation_graph"), plan.get("spatial", {}).get("side", "unknown")
    if graph and len(entities) > 1:
        relation = str(graph["relation"]).replace("_", " ")
        atoms.append({"type": "SPATIAL", "text": f"{subject} {relation} {entities[1]['object_phrase']}", "weight": 1.0})
    elif side != "unknown":
        atoms.append({"type": "SPATIAL", "text": f"{subject} on the {side}", "weight": 1.0})
    # Explicit transition endpoints only. Inferred actor states are not certified.
    q = query.casefold()
    if temporal and "left to right" in q:
        atoms += [{"type": "STATE_A", "text": f"{subject} on the left", "weight": .5},
                  {"type": "STATE_B", "text": f"{subject} on the right", "weight": .5}]
    elif temporal and "right to left" in q:
        atoms += [{"type": "STATE_A", "text": f"{subject} on the right", "weight": .5},
                  {"type": "STATE_B", "text": f"{subject} on the left", "weight": .5}]
    # Bound to five factors; states replace the redundant static side factor.
    if len(atoms) > 5:
        atoms = [a for a in atoms if a["type"] != "SPATIAL"][:5]
    return atoms


def aggregate(values, weights, method="weighted_mean"):
    x, w = np.asarray(values, float), np.asarray(weights, float)
    if not len(x) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError("INVALID_FACTOR_SCORES")
    if method == "mean":
        return float(x.mean())
    if method == "weighted_mean":
        return float(np.average(x, weights=w))
    if method == "min_aware":
        return float(.75 * np.average(x, weights=w) + .25 * x.min())
    if method == "top_factor":
        return float(np.sort(x)[-min(2, len(x)):].mean())
    if method == "geometric":
        return float(2 * np.exp(np.average(np.log(np.clip((x + 1) / 2, 1e-8, 1)), weights=w)) - 1)
    raise ValueError("INVALID_AGGREGATION")


def late_interaction(atoms, vectors, evidence, *, typed=True, aggregation="weighted_mean", families=None):
    values, weights, matches = [], [], []
    for atom in atoms:
        query_vector = np.asarray(vectors.get(atom["text"], []), np.float32)
        if (query_vector.shape != (DIMENSION,) or not np.isfinite(query_vector).all() or
                not np.isclose(np.linalg.norm(query_vector), 1., atol=.003)):
            return {"valid": False, "score": None, "reason": "INVALID_FACTOR_VECTOR", "matches": matches}
        choices = [e for e in evidence if (families is None or e["family"] in families) and
                   (not typed or e["family"] in COMPATIBILITY[atom["type"]])]
        if not choices:
            return {"valid": False, "score": None, "reason": "FACTOR_EVIDENCE_MISSING", "matches": matches}
        matrix = np.asarray([e["vector"] for e in choices], np.float32)
        if matrix.shape != (len(choices), DIMENSION) or not np.isfinite(matrix).all():
            return {"valid": False, "score": None, "reason": "INVALID_EVIDENCE", "matches": matches}
        scores = matrix @ query_vector
        if not np.isfinite(scores).all():
            return {"valid": False, "score": None, "reason": "INVALID_EVIDENCE", "matches": matches}
        best = int(np.argmax(scores))
        match = {k: v for k, v in choices[best].items() if k != "vector"}
        matches.append({"factor_type": atom["type"], "score": float(scores[best]), **match})
        values.append(float(scores[best])); weights.append(atom["weight"])
    return {"valid": True, "score": aggregate(values, weights, aggregation), "matches": matches}


def temporal_support(matches):
    stamps = {m["factor_type"]: (m["start"] + m["end"]) / 2 for m in matches if m["family"] in ("WINDOW", "FRAME")}
    if "STATE_A" not in stamps or "STATE_B" not in stamps:
        return {"available": False, "score": 0., "order_valid": None}
    gap = stamps["STATE_B"] - stamps["STATE_A"]
    if "ACTION" in stamps:
        valid = stamps["STATE_A"] < stamps["ACTION"] < stamps["STATE_B"]
    else:
        valid = gap > 0
    return {"available": True, "order_valid": bool(valid), "order_margin": float(gap),
            "score": min(gap / 5, 1.) if valid else -min(abs(gap) / 5, 1.)}


def stable_order(scores):
    return np.argsort(-np.asarray(scores), kind="stable")


def cache_key(query, eg2_revision, laya_revision, config):
    # Literal EG2 input is included; normalized Laya cache has a separate identity.
    return identity({"query": query, "normalized_query": " ".join(query.casefold().split()),
                     "EG2": eg2_revision, "Laya": laya_revision, "schema": FACTOR_SCHEMA,
                     "config": config})


def load_index(path):
    """Validate global core independently; corrupt optional evidence cannot rerank."""
    root = Path(path)
    meta = json.loads((root / "index_meta.json").read_text(encoding="utf-8"))
    if meta.get("format") != FORMAT or meta.get("schema") != 1 or meta.get("config", {}).get("revision") != REVISION:
        raise ValueError("V6_INDEX_IDENTITY_MISMATCH")
    if identity(meta["representation"]) != meta.get("representation_sha256"):
        raise ValueError("V6_REPRESENTATION_IDENTITY_MISMATCH")
    validate_representation(meta["representation"])
    core = ("global_vectors.npy", "manifest.json", "provenance.json")
    optional = [prefix + suffix for prefix in ("window", "frame", "tube")
                for suffix in ("_vectors.npy", "_offsets.npy", "_metadata.json", "_meta.npz")]
    bad = [name for name in optional if name not in meta["hashes"]]
    for name, expected in meta["hashes"].items():
        if Path(name).name != name:
            raise ValueError("V6_INDEX_PATH_ESCAPE")
        if not (root / name).is_file() or digest(root / name) != expected:
            if name in core:
                raise ValueError("V6_CORE_HASH_MISMATCH " + name)
            bad.append(name)
    if not all(name in meta["hashes"] for name in core):
        raise ValueError("V6_MISSING_CORE_IDENTITIES")
    rows = json.loads((root / "manifest.json").read_text(encoding="utf-8"))["clips"]
    global_vectors = np.load(root / "global_vectors.npy", mmap_mode="r", allow_pickle=False)
    if global_vectors.shape != (len(rows), DIMENSION) or not np.isfinite(global_vectors).all() or not np.allclose(np.linalg.norm(global_vectors, axis=1), 1, atol=.003):
        raise ValueError("V6_INVALID_GLOBAL_VECTORS")
    out = {"rows": rows, "global_vectors": global_vectors, "meta": meta, "evidence_fallback": bad, "families": {}}
    if bad:
        return out
    for family in ("WINDOW", "FRAME", "TUBE"):
        prefix = family.casefold()
        try:
            matrix = np.load(root / (prefix + "_vectors.npy"), mmap_mode="r", allow_pickle=False)
            offsets = np.load(root / (prefix + "_offsets.npy"), allow_pickle=False)
            metadata = json.loads((root / (prefix + "_metadata.json")).read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            out["evidence_fallback"] = ["V6_INVALID_" + family]
            out["families"] = {}
            return out
        valid = (isinstance(metadata, list) and matrix.shape == (len(metadata), DIMENSION) and offsets.shape == (len(rows) + 1,) and
                 offsets.dtype.kind in "iu" and offsets[0] == 0 and offsets[-1] == len(matrix) and
                 np.all(np.diff(offsets) >= 0) and np.isfinite(matrix).all() and
                 (not len(matrix) or np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.003)))
        if not valid:
            out["evidence_fallback"] = ["V6_INVALID_" + family]
            out["families"] = {}
            return out
        for row, clip in enumerate(rows):
            for i in range(int(offsets[row]), int(offsets[row + 1])):
                try:
                    m = metadata[i]
                    stamps = m["timestamps"]
                    invalid = (m["clip_id"] != clip["clip_id"] or m["source_sha256"] != clip["source_sha256"] or
                        m["family"] != family or not 0 <= m["start"] <= m["end"] <= 5 or
                        not len(stamps) or not all(math.isfinite(t) and 0 <= t <= 5 for t in stamps) or
                        any(b <= a for a, b in zip(stamps, stamps[1:])) or
                        family == "FRAME" and (len(stamps) != 1 or m["start"] != m["end"]) or
                        family == "WINDOW" and len(stamps) < 2 or family == "TUBE" and len(stamps) < 3)
                except (KeyError, TypeError, ValueError):
                    invalid = True
                if invalid:
                    out["evidence_fallback"] = ["V6_EVIDENCE_PROVENANCE_MISMATCH_" + family]
                    out["families"] = {}
                    return out
        out["families"][family] = {"vectors": matrix, "offsets": offsets, "metadata": metadata}
    return out


def candidate_evidence(loaded, row):
    evidence = [{"family": "GLOBAL", "vector": loaded["global_vectors"][row], "start": 0., "end": 5., "row": row}]
    for family, data in loaded["families"].items():
        for i in range(int(data["offsets"][row]), int(data["offsets"][row + 1])):
            evidence.append({**data["metadata"][i], "vector": data["vectors"][i], "row": i})
    return evidence
