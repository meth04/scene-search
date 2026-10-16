"""Build a V4.1 retrieval index from a verified visual subset.

Reuses the frozen visual/motion arrays (they are video-derived, not metadata-derived)
and regenerates every text/BGE array with the exact BGE recipe. The final record is
the deterministic local merge of the three passes; Qwen never emits the final JSON.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import metadata_v4_1 as contract
from .common import chronological_frame_ids
from .dense_index import METADATA_FILES, TEXT_FILES, receipt, verify_receipt
from .dense_metadata import BGE_RECIPE, DENSE_FORMAT, FIELDS, NORMALIZATION_VERSION
from .local_index import complete_phase, read_meta, rows
from .local_runtime import atomic_array, atomic_jsonl
from .metadata_v4_1_pipeline import MultiPassStore
from .qwen_store import atomic_json

V4_1_NATIVE = "v4_1_native"
FRAME_COUNT = 8
JPEG_QUALITY = 85
TILE = [256, 144]
SHEET_LAYOUT = [2, 4]


def read(path, default=None):
    """Read a JSON artifact, or `default` when it is absent (declared placeholders)."""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _dense_record(value, window, contract_module=contract):
    record = contract_module.to_dense(value)
    record.update({k: window[k] for k in ("row", "clip_id", "start_sec", "end_sec")})
    record["caption_sample_frame_ids"] = chronological_frame_ids(
        window["start_frame"], window["end_frame"], FRAME_COUNT
    ).tolist()
    record["caption_request_seconds"] = 0.0
    return record


def _placeholder_record(window, reason):
    """Alignment-preserving, non-searchable row for a window that never reached
    FINAL_SUCCESS. It keeps the index row-aligned (so arrays/manifest still zip)
    without fabricating actor facts: it is the dense-contract fallback record, so
    the retriever zeroes its scores exactly like any other missing metadata row.
    """
    from .dense_metadata import empty_metadata, searchable_fields, validate_record

    record = empty_metadata(reason)
    record.update({k: window[k] for k in ("row", "clip_id", "start_sec", "end_sec")})
    record["caption_sample_frame_ids"] = chronological_frame_ids(
        window["start_frame"], window["end_frame"], FRAME_COUNT
    ).tolist()
    record["caption_request_seconds"] = 0.0
    record["searchable_texts"] = searchable_fields(record)
    validate_record(record)
    return record


def finalize(index, *, selection=None, qwen_config=None, workdir=None, allow_partial=False,
             contract_module=contract, store_cls=MultiPassStore, store_relative="metadata/v4_1",
             native_flag=V4_1_NATIVE):
    """Assemble the index from committed multi-pass rows.

    Strict by default: any row without FINAL_SUCCESS writes `metadata/incomplete.json`
    and raises, so a partial index is never silently published (§25).

    With `allow_partial=True` (used only after the §14 >=49/50 gate passes) the
    index is still assembled row-aligned, but each missing row becomes a declared
    non-searchable placeholder recorded in `metadata/unpublished.json`; the index
    is then reported UNPUBLISHED and the API readiness gate rejects it.
    """
    contract = contract_module
    index = Path(index)
    meta = read_meta(index)
    windows = rows(index / "manifest.jsonl")
    store = store_cls(index / store_relative)
    records, missing = [], []
    for w in windows:
        window_id = f"{w['row']:06d}"
        value, saved = store._read(window_id, "final")
        if value is None or not saved or saved["state"] != "FINAL_SUCCESS":
            missing.append(w)
            continue
        records.append(_dense_record(value, w, contract))
    if missing and not allow_partial:
        atomic_json(index / "metadata/incomplete.json", {"missing_final": [f"{w['row']:06d}" for w in missing], "expected": len(windows)})
        raise RuntimeError(f"V4.1 generation incomplete: {len(missing)} rows lack FINAL_SUCCESS")

    unpublished = []
    for w in missing:
        records.append(_placeholder_record(w, f"v4.1 generation incomplete for window {w['row']:06d}"))
        unpublished.append(f"{w['row']:06d}")
    records.sort(key=lambda r: r["row"])
    if unpublished:
        atomic_json(index / "metadata/unpublished.json", {"rows": sorted(unpublished), "expected": len(windows),
                                                          "successful_windows": len(records) - len(unpublished)})
        (index / "metadata/incomplete.json").unlink(missing_ok=True)

    atomic_jsonl(index / METADATA_FILES[0], records)
    captions = []
    for record in records:
        text = record["searchable_texts"]
        caption = {
            **{k: record[k] for k in ("row", "clip_id", "start_sec", "end_sec", "caption_sample_frame_ids")},
            "description": text["full_text"] or record["description"],
            "objects": [],
            "ego_action": "",
            "other_actions": [text["action_text"]],
            "relations": [text["relation_text"]],
            "road_context": [text["scene_text"]],
            "traffic_control": [text["infrastructure_text"]],
            "weather": "",
            "events": [],
            "queries": [],
            "fallback": bool(record.get("fallback")),
        }
        if caption["fallback"]:
            caption["fallback_reason"] = record["fallback_reason"]
        captions.append(caption)
    atomic_jsonl(index / "captions/captions.jsonl", captions)

    fallback = sum(bool(r.get("fallback")) for r in records)
    stats = {
        "total": len(records),
        "ok": len(records) - fallback,
        "fallback": fallback,
        "fallback_rate": fallback / len(records),
        "warning_count": sum(len(r["metadata_warnings"]) for r in records),
        "warning_rows": sum(bool(r["metadata_warnings"]) for r in records),
    }
    identity = {
        "schema_version": contract.VERSION,
        "prompt_version": contract.VERSION,
        "prompt_sha256": contract.COMBINED_PROMPT_SHA256,
        "pass_prompt_sha256": contract.PROMPT_SHA256,
        "served_model": meta.get("models", {}).get("qwen", {}).get("served_model"),
        "base_url": meta.get("models", {}).get("qwen", {}).get("base_url"),
        "mode": meta.get("models", {}).get("qwen", {}).get("mode"),
        "frame_count": FRAME_COUNT,
        "jpeg_quality": JPEG_QUALITY,
        "tile_size": TILE,
        "sheet_layout": SHEET_LAYOUT,
    }
    meta.update(
        format=DENSE_FORMAT,
        index_format=DENSE_FORMAT,
        index_version=1,
        v3_native=False,
        v4_native=False,
        **{native_flag: True},
        dense_metadata_version=contract.VERSION,
        caption_prompt_version=contract.VERSION,
        caption_prompt_sha256=contract.COMBINED_PROMPT_SHA256,
        caption_frame_count=FRAME_COUNT,
        contact_sheet_layout=SHEET_LAYOUT,
        caption_in_progress=identity,
        caption_stats={k: stats[k] for k in ("total", "ok", "fallback", "fallback_rate")},
        dense_stats=stats,
    )
    meta["models"]["qwen"] = identity
    meta["dense_provenance"] = {
        "schema_version": contract.VERSION,
        "qwen": identity,
        "bge_recipe": BGE_RECIPE,
        "normalization_version": NORMALIZATION_VERSION,
        "stats": stats,
    }
    atomic_json(index / METADATA_FILES[1], meta["dense_provenance"])
    complete_phase(index, meta, "caption", {"schema_version": contract.VERSION})
    receipt(index, meta, "metadata", METADATA_FILES)
    return records


def embed(index, meta=None):
    """Regenerate all 12 BGE text arrays with the exact recipe (no source arrays)."""
    from .text_encoder import BGEEncoder

    index = Path(index)
    meta = meta or read_meta(index)
    records = rows(index / METADATA_FILES[0])
    settings = meta["settings"]
    encoder = BGEEncoder(
        settings.get("bge_model", "BAAI/bge-base-en-v1.5"),
        "cpu",
        cache_dir=settings["hf_cache_dir"],
        revision=settings.get("bge_revision"),
        local_files_only=True,
    )
    started = time.perf_counter()
    truncated = {}
    for field in FIELDS:
        texts = [r["searchable_texts"][field + "_text"] or "none" for r in records]
        truncated[field] = sum(len(encoder.tok.encode(t)) > 128 for t in texts)
        atomic_array(index / f"text/dense_{field}.npy", encoder.encode(texts, batch_size=settings.get("text_batch_size", 8)))
    for field, dense in (("full", "full"), ("action", "action"), ("relation", "relation"), ("context", "scene")):
        atomic_array(
            index / f"text/caption_{field}.npy",
            encoder.encode([r["searchable_texts"][dense + "_text"] or "none" for r in records], batch_size=settings.get("text_batch_size", 8)),
        )
    meta["models"]["bge"] = encoder.identity
    meta["dimensions"]["bge"] = int(encoder.model.config.hidden_size)
    meta["dense_provenance"]["bge_identity"] = encoder.identity
    meta["dense_text_timing"] = {
        "seconds": time.perf_counter() - started,
        "truncated_rows_by_field": truncated,
        "max_length": 128,
    }
    atomic_json(index / METADATA_FILES[1], meta["dense_provenance"])
    receipt(index, meta, "metadata", METADATA_FILES)
    receipt(index, meta, "text", TEXT_FILES)
    complete_phase(index, meta, "text-index", meta["dense_text_timing"])
    return meta["dense_text_timing"]


def validate(index, meta, manifest, *, contract_module=contract, store_cls=MultiPassStore,
             store_relative="metadata/v4_1", native_flag=V4_1_NATIVE):
    import numpy as np

    from .local_index import array

    contract = contract_module
    index = Path(index)
    if not meta.get(native_flag) or meta["caption_prompt_version"] != contract.VERSION:
        raise ValueError("V4.1 prompt/schema mismatch")
    if meta["models"]["qwen"] != meta["caption_in_progress"] or meta["models"]["qwen"] != meta["dense_provenance"]["qwen"]:
        raise ValueError("V4.1 provider provenance mismatch")
    if meta["models"]["bge"] != meta["dense_provenance"]["bge_identity"] or meta["dense_provenance"]["bge_recipe"] != BGE_RECIPE:
        raise ValueError("V4.1 BGE recipe/provenance mismatch")
    for phase, files in (("metadata", METADATA_FILES), ("text", TEXT_FILES)):
        if not verify_receipt(index, meta, phase, files):
            raise ValueError("V4.1 index not ready: " + phase)
    records = rows(index / METADATA_FILES[0])
    if len(records) != len(manifest):
        raise ValueError("V4.1 row count mismatch")
    # Declared non-searchable placeholders keep the index row-aligned but make it
    # UNPUBLISHED; only rows named here may lack FINAL_SUCCESS (§25).
    unpublished = set(read(index / "metadata/unpublished.json", {}).get("rows", []))
    store = store_cls(index / store_relative)
    for record, w in zip(records, manifest):
        window_id = f"{w['row']:06d}"
        value, saved = store._read(window_id, "final")
        if value is None or saved["state"] != "FINAL_SUCCESS":
            if window_id not in unpublished or not record.get("fallback"):
                raise ValueError("V4.1 missing FINAL_SUCCESS row: " + window_id)
            if any(record[k] != w[k] for k in ("row", "clip_id", "start_sec", "end_sec")):
                raise ValueError("V4.1 placeholder alignment mismatch: " + window_id)
            continue
        if record["searchable_texts"] != contract.to_dense(value)["searchable_texts"] or record.get("fallback"):
            raise ValueError("V4.1 deterministic search text mismatch")
        if any(record[k] != w[k] for k in ("row", "clip_id", "start_sec", "end_sec")) or record[
            "caption_sample_frame_ids"
        ] != chronological_frame_ids(w["start_frame"], w["end_frame"], FRAME_COUNT).tolist():
            raise ValueError("V4.1 alignment mismatch")
        # The committed final record must itself satisfy the V4.1 schema.
        if contract.validate_metadata(value)["schema_version"] != contract.VERSION:
            raise ValueError("V4.1 record schema mismatch: " + window_id)
    for file in TEXT_FILES:
        v = array(index, file)
        if v.dtype != np.float32 or v.shape != (len(manifest), 768) or not np.isfinite(v).all() or not np.allclose(
            np.linalg.norm(v, axis=1), 1, atol=2e-3
        ):
            raise ValueError("V4.1 invalid embedding: " + file)
    from .segments import validation_artifact

    if validation_artifact(manifest, meta["segments"])["status"] != "PASS":
        raise ValueError("V4.1 cross-clip contamination")
    return {"dense_metadata": "PASS", "schema_version": contract.VERSION,
            "successful_windows": len(records) - len(unpublished),
            "unpublished_rows": sorted(unpublished),
            "publication": "UNPUBLISHED" if unpublished else "PUBLISHED"}
