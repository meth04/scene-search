"""Versioned single-video index, phase receipts, and strict validation."""
from __future__ import annotations

from datetime import datetime, timezone
import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np

from .build_features import FEATURE_NAMES
from .local_runtime import FORMAT, atomic_json, check_model_identity, load_checkpoint, sha256

PHASES = ["prepare", "visual", "motion", "fuse", "frame-project", "caption", "text-index", "validate"]
ARTIFACTS = {
    "prepare": ["manifest.jsonl"],
    "visual": ["visual/raw_frame_features.npy", "visual/base_clip_embeddings.npy"],
    "motion": ["visual/tracks.npy", "visual/track_masks.npy"],
    "fuse": ["visual/fused_embeddings.npy"],
    "frame-project": ["visual/projected_frame_embeddings.npy"],
    "caption": ["captions/captions.jsonl"],
    "text-index": [f"text/caption_{name}.npy" for name in ("full", "action", "relation", "context")],
}


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_meta(index):
    meta = json.loads((Path(index) / "index_meta.json").read_text(encoding="utf-8"))
    from .dense_metadata import DENSE_FORMAT
    from .embeddinggemma2 import FORMAT as EG2_FORMAT
    if meta.get("format") not in (FORMAT, DENSE_FORMAT, EG2_FORMAT, "driveevidence-v1", "driveevidence-v2", "driveevidence-v3", "driveevidence-v4", "driveevidence-v5", "driveevidence-v6"):
        raise ValueError(f"Unsupported index format: {meta.get('format')!r}")
    return meta


def load_stage5(path):
    ck = load_checkpoint(path, "driveretriever-stage5-hybrid-v1")
    if ck.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Stage5 feature order mismatch: checkpoint.feature_names != runtime FEATURE_NAMES")
    for name in ("feature_mean", "feature_std"):
        arr = np.asarray(ck[name])
        if arr.shape != (18,) or not np.isfinite(arr).all():
            raise ValueError(f"Invalid Stage5 {name}")
    if (np.asarray(ck["feature_std"]) <= 0).any() or not math.isfinite(float(ck["alpha"])):
        raise ValueError("Invalid Stage5 std/alpha; refusing to alter saved calibration")
    return ck


def verify_checkpoints(meta):
    for name, info in meta["checkpoints"].items():
        if sha256(info["path"]) != info["sha256"]:
            raise ValueError(f"Checkpoint provenance mismatch: {name}")
    load_stage5(meta["checkpoints"]["stage5"]["path"])


def phase_valid(index, meta, phase):
    receipt = meta.get("phases", {}).get(phase)
    if not receipt:
        return False
    expected = ARTIFACTS.get(phase, [])
    if set(receipt["artifacts"]) != set(expected):
        raise ValueError(f"Incomplete artifact receipt: {phase}")
    for path, digest in receipt["artifacts"].items():
        file = Path(index) / path
        if not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Index artifact changed/incomplete: {path}")
    return True


def complete_phase(index, meta, phase, details=None):
    meta.setdefault("phases", {})[phase] = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": {p: sha256(Path(index) / p) for p in ARTIFACTS.get(phase, [])},
        "details": details or {},
    }
    atomic_json(Path(index) / "index_meta.json", meta)


def array(index, path):
    return np.load(Path(index) / path, mmap_mode="r", allow_pickle=False)


def caption_prompt_info(meta):
    """Legacy indexes may only have the nested Qwen prompt identity, or neither."""
    qwen = meta.get("models", {}).get("qwen", {})
    version = meta.get("caption_prompt_version", qwen.get("prompt_version", "unknown/legacy-unversioned"))
    digest = meta.get("caption_prompt_sha256", qwen.get("prompt_sha256"))
    if "caption_prompt_version" in meta or "caption_prompt_sha256" in meta:
        if not isinstance(version, str) or not version or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("Invalid caption prompt metadata")
        if any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Invalid caption prompt SHA256")
        if qwen and (qwen.get("prompt_version") != version or qwen.get("prompt_sha256") != digest):
            raise ValueError("Caption prompt provenance mismatch")
    info = {"caption_prompt_version": version, "caption_prompt_sha256": digest}
    if "caption_frame_count" in meta:
        count = meta["caption_frame_count"]
        if not isinstance(count, int) or isinstance(count, bool) or count < 2:
            raise ValueError("Invalid caption frame count")
        if qwen.get("frame_count", 6) != count:
            raise ValueError("Caption frame count provenance mismatch")
        info["caption_frame_count"] = count
    return info


def reuse_visual_index(source, target):
    """Copy verified independent artifacts through frame projection into a NEW index."""
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Source and target indexes must be independent directories")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite index: {target}")
    validate_index(source)
    original = read_meta(source)
    if sha256(original["video"]["path"]) != original["video"]["sha256"]:
        raise ValueError("Source video changed; refusing representation reuse")
    manifest = rows(source / "manifest.jsonl")
    for row in manifest:
        if (abs(row["start_sec"]*row["fps"]-row["start_frame"]) > .51
                or abs(row["end_sec"]*row["fps"]-row["end_frame"]) > .51):
            raise ValueError("Source window timing/frame alignment mismatch")
    meta = copy.deepcopy(original)
    meta["format"] = FORMAT
    for key in ("index_format", "index_version", "dense_metadata_version", "dense_metadata_format", "dense_recap", "dense_provenance", "dense_stats", "dense_receipts", "dense_text_timing", "contact_sheet_layout"):
        meta.pop(key, None)
    meta["created_at"] = datetime.now(timezone.utc).isoformat()
    meta["phases"] = {}
    meta["models"] = {"siglip_name": original["models"].get("siglip_name"), "siglip": original["models"]["siglip"]}
    for key in ("caption_stats", "caption_timing", "caption_in_progress", "caption_prompt_version", "caption_prompt_sha256",
                "caption_frame_count", "caption_cache_reuse", "recaption"):
        meta.pop(key, None)
    meta["representation_reuse"] = {"source_index": str(source), "source_meta_sha256": sha256(source/"index_meta.json"),
                                    "source_video_sha256": original["video"]["sha256"], "row_count": len(manifest),
                                    "window_timing": [{k:r[k] for k in ("row", "clip_id", "start_sec", "end_sec", "start_frame", "end_frame")} for r in manifest],
                                    "array_shapes": {p:list(array(source,p).shape) for phase in PHASES[1:5] for p in ARTIFACTS[phase]},
                                    "artifacts": {}}
    target.mkdir(parents=True)
    for phase in PHASES[:5]:
        for relative, digest in original["phases"][phase]["artifacts"].items():
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
            if sha256(destination) != digest:
                raise ValueError(f"Copied representation hash mismatch: {relative}")
            meta["representation_reuse"]["artifacts"][relative] = digest
        complete_phase(target, meta, phase, {"reused_from": str(source)})
    return target


def validate_index(index, check_models=True):
    index = Path(index)
    meta = read_meta(index)
    verify_checkpoints(meta)
    if meta.get("feature_names") != FEATURE_NAMES:
        raise ValueError("Index feature schema mismatch")
    for phase in PHASES[:-1]:
        if not phase_valid(index, meta, phase):
            raise ValueError(f"Index incomplete: phase {phase} has not completed")
    manifest, captions = rows(index / "manifest.jsonl"), rows(index / "captions/captions.jsonl")
    n = meta["num_windows"]
    if n <= 0 or len(manifest) != n or len(captions) != n:
        raise ValueError("Manifest/caption row count mismatch")
    for i, (m, c) in enumerate(zip(manifest, captions)):
        if m.get("row") != i or c.get("row") != i or c.get("clip_id") != m.get("clip_id"):
            raise ValueError(f"Manifest/caption row alignment mismatch at row {i}")
        if m.get("video_id") != meta["video_id"] or m.get("video_path") != meta["video"]["path"]:
            raise ValueError(f"Manifest video provenance mismatch at row {i}")
        if not (0 <= m["start_sec"] < m["end_sec"] <= meta["video"]["duration_sec"] + 1e-5):
            raise ValueError(f"Invalid timestamps at row {i}")
        if not (0 <= m["start_frame"] < m["end_frame"] <= meta["video"]["frame_count"]):
            raise ValueError(f"Invalid frame bounds at row {i}")
        if not math.isfinite(m["fps"]) or m["fps"] != meta["video"]["fps"]:
            raise ValueError(f"Invalid frame rate at row {i}")
        if not str(c.get("description", "")).strip():
            raise ValueError(f"Empty caption at row {i}")
        if c.get("fallback") and (c.get("queries") != [] or not c.get("fallback_reason")):
            raise ValueError(f"Invalid fallback caption at row {i}")
    dims = meta["dimensions"]
    t, h, e, d = dims["frames"], dims["hidden"], dims["embedding"], dims["bge"]
    if (t, h, e) != (12, 768, 512) or not isinstance(d, int) or d <= 0:
        raise ValueError(f"Unsupported index dimensions: {dims}")
    expected = {
        "visual/raw_frame_features.npy": ((n, t, h), False),
        "visual/base_clip_embeddings.npy": ((n, e), True),
        "visual/fused_embeddings.npy": ((n, e), True),
        "visual/projected_frame_embeddings.npy": ((n, t, e), True),
        "visual/tracks.npy": ((n, 24, 29), False),
        "visual/track_masks.npy": ((n, 24), False),
        **{f"text/caption_{x}.npy": ((n, d), True) for x in ("full", "action", "relation", "context")},
    }
    for path, (shape, normalized) in expected.items():
        data = array(index, path)
        if data.shape != shape:
            raise ValueError(f"Array row alignment/dimension mismatch: {path}: {data.shape} != {shape}")
        if path.endswith("track_masks.npy"):
            if data.dtype != np.bool_:
                raise ValueError("Track masks must be bool")
            continue
        if data.dtype.kind != "f":
            raise ValueError(f"Expected floating-point representation: {path}")
        for s in range(0, n, 128):
            chunk = np.asarray(data[s:s+128], dtype=np.float32)
            if not np.isfinite(chunk).all():
                raise ValueError(f"Non-finite index values: {path}")
            if normalized and not np.allclose(np.linalg.norm(chunk, axis=-1), 1, atol=2e-3):
                raise ValueError(f"Non-normalized embeddings: {path}")
            if path.endswith("raw_frame_features.npy") and (np.linalg.norm(chunk, axis=-1) == 0).any():
                raise ValueError("Zero raw frame embedding")
    base = array(index, "visual/base_clip_embeddings.npy")
    fused = array(index, "visual/fused_embeddings.npy")
    masks = array(index, "visual/track_masks.npy")
    for s in range(0, n, 128):
        empty = ~masks[s:s+128].any(axis=1)
        b = np.asarray(base[s:s+128], dtype=np.float32)[empty]
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
        if not np.allclose(fused[s:s+128][empty], b, atol=1e-6):
            raise ValueError("Stage3A no-track fallback violated in saved index")
    if check_models:
        for name in ("siglip", "bge"):
            if not meta["models"].get(name):
                raise ValueError(f"Missing model provenance: {name}")
            check_model_identity(meta["models"][name])
    fallback = sum(bool(c.get("fallback")) for c in captions)
    stats = meta.get("caption_stats", {})
    if (stats.get("fallback") != fallback or stats.get("total") != n or stats.get("ok") != n-fallback
            or not math.isclose(stats.get("fallback_rate", -1), fallback/n, abs_tol=1e-8)):
        raise ValueError("Caption fallback statistics mismatch")
    from .dense_metadata import DENSE_FORMAT
    dense = {}
    if meta["format"] == DENSE_FORMAT or (index/"metadata/dense_metadata.jsonl").exists() or meta.get("dense_metadata_version"):
        from .dense_index import validate_dense
        dense = validate_dense(index, meta, manifest)
    return {"status": "PASS", "windows": n, "fallback": fallback, "fallback_rate": fallback/n, **dense,
            **caption_prompt_info(meta)}
