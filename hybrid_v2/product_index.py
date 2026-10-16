"""Validated v3 indexing shared by local and server modes; never commits failed rows."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from .common import chronological_frame_ids
from .dense_metadata import (
    BGE_RECIPE,
    DENSE_FORMAT,
    FIELDS,
    METADATA_VERSION,
    NORMALIZATION_VERSION,
    legacy_caption,
    validate_record,
)
from .local_index import complete_phase, rows
from .local_runtime import atomic_array, atomic_jsonl
from .qwen_metadata import PROMPT_SHA256, VERSION, to_dense
from .qwen_provider import QwenError, create_provider
from .qwen_store import MetadataStore, atomic_json


def extract(index, meta, settings):
    from .dense_index import METADATA_FILES, receipt
    from .local_video_index import contact_sheet

    index = Path(index)
    provider = create_provider(settings, evidence_dir=index / "metadata/failures")
    try:
        provider.healthcheck()
    except Exception:
        provider.close()
        raise
    store = MetadataStore(index / "metadata/v3")
    records, captions, failed = [], [], []
    count = settings.get("caption_frames", 6)
    columns = math.ceil(count / 2)
    identity = {
        "served_model": provider.model,
        "base_url": provider.config.base_url,
        "mode": provider.config.mode,
        "prompt_version": VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "frame_count": count,
        "jpeg_quality": settings.get("caption_jpeg_quality", 85),
        "schema_version": VERSION,
        "sheet_layout": [math.ceil(count / columns), columns],
        "tile_size": [settings["caption_tile_width"], settings["caption_tile_height"]],
    }
    if meta.get("caption_in_progress") and meta["caption_in_progress"] != identity:
        raise ValueError("Qwen metadata provenance changed; choose a new output")
    meta["caption_in_progress"] = identity
    atomic_json(index / "index_meta.json", meta)
    try:
        for window in rows(index / "manifest.jsonl"):
            started = time.perf_counter()
            sheet = contact_sheet(
                window["video_path"], window, settings["caption_tile_width"], settings["caption_tile_height"], count
            )
            import base64

            from .qwen_provider import jpeg_data_url

            url = jpeg_data_url(sheet, identity["jpeg_quality"])
            target = index / f"metadata/contact_sheets/{window['row']:06d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(base64.b64decode(url.split(",", 1)[1]))
            try:
                value = store.generate(
                    f"{window['row']:06d}",
                    [url],
                    provider,
                    {
                        "index_id": index.name,
                        "source_identity": {
                            "sha256": meta["video"]["sha256"],
                            "clip_id": window["clip_id"],
                            "segment_id": window.get("segment_id"),
                        },
                    },
                )
            except QwenError as exc:
                failed.append({"row": window["row"], "error_code": exc.code})
                print(f"[metadata] {window['row'] + 1}/{meta['num_windows']} {exc.code}", flush=True)
                if provider.clock() < provider.open_until:
                    break
                continue
            record = to_dense(value)
            record.update({k: window[k] for k in ("row", "clip_id", "start_sec", "end_sec")})
            record.update(
                caption_sample_frame_ids=chronological_frame_ids(
                    window["start_frame"], window["end_frame"], count
                ).tolist(),
                caption_request_seconds=time.perf_counter() - started,
            )
            validate_record(record)
            atomic_json(index / f"metadata/rows/{window['row']:06d}.json", record)
            records.append(record)
            caption = legacy_caption(record)
            caption.update(
                {
                    k: record[k]
                    for k in (
                        "row",
                        "clip_id",
                        "start_sec",
                        "end_sec",
                        "caption_sample_frame_ids",
                        "caption_request_seconds",
                    )
                }
            )
            captions.append(caption)
            print(f"[metadata] {window['row'] + 1}/{meta['num_windows']} SUCCESS", flush=True)
    finally:
        atomic_json(index / "metadata/metrics.json", provider.metrics())
        atomic_json(index / "metadata/window_states.json", store.status())
        provider.close()
    if failed or len(records) != meta["num_windows"]:
        atomic_json(
            index / "metadata/incomplete.json",
            {"failed": failed, "success": len(records), "expected": meta["num_windows"]},
        )
        raise QwenError("QWEN_SCHEMA_FAILURE", retryable=True)
    # Atomic aggregates/phase receipts are created only when all windows are valid.
    atomic_jsonl(index / METADATA_FILES[0], records)
    atomic_jsonl(index / "captions/captions.jsonl", captions)
    meta.update(
        format=DENSE_FORMAT,
        index_format=DENSE_FORMAT,
        index_version=1,
        v3_native=True,
        dense_metadata_format=DENSE_FORMAT,
        dense_metadata_version=METADATA_VERSION,
        caption_prompt_version=VERSION,
        caption_prompt_sha256=PROMPT_SHA256,
        caption_frame_count=count,
        contact_sheet_layout=identity["sheet_layout"],
    )
    meta["models"]["qwen"] = identity
    meta["caption_stats"] = {"total": len(records), "ok": len(records), "fallback": 0, "fallback_rate": 0.0}
    meta["dense_stats"] = {
        **meta["caption_stats"],
        "warning_count": sum(len(r["metadata_warnings"]) for r in records),
        "warning_rows": sum(bool(r["metadata_warnings"]) for r in records),
    }
    meta["dense_provenance"] = {
        "format": DENSE_FORMAT,
        "version": METADATA_VERSION,
        "qwen": identity,
        "bge_recipe": BGE_RECIPE,
        "stats": meta["dense_stats"],
        "normalization_version": NORMALIZATION_VERSION,
    }
    atomic_json(index / METADATA_FILES[1], meta["dense_provenance"])
    complete_phase(index, meta, "caption", {"schema_version": VERSION})
    receipt(index, meta, "metadata", METADATA_FILES)


def embed(index, meta, settings):
    from .dense_index import METADATA_FILES, TEXT_FILES, receipt
    from .embed_captions import fields
    from .text_encoder import BGEEncoder

    index = Path(index)
    encoder = BGEEncoder(
        settings["bge_model"],
        "cpu",
        cache_dir=settings["hf_cache_dir"],
        revision=settings.get("bge_revision"),
        local_files_only=True,
    )
    records = rows(index / METADATA_FILES[0])
    captions = rows(index / "captions/captions.jsonl")
    meta["models"]["bge"] = encoder.identity
    meta["dimensions"]["bge"] = int(encoder.model.config.hidden_size)
    for name, texts in zip(("full", "action", "relation", "context"), zip(*[fields(r) for r in captions])):
        atomic_array(index / f"text/caption_{name}.npy", encoder.encode(texts, batch_size=settings["text_batch_size"]))
    for field in FIELDS:
        atomic_array(
            index / f"text/dense_{field}.npy",
            encoder.encode(
                [r["searchable_texts"][field + "_text"] or "none" for r in records],
                batch_size=settings["text_batch_size"],
            ),
        )
    meta["dense_provenance"]["bge_identity"] = encoder.identity
    atomic_json(index / METADATA_FILES[1], meta["dense_provenance"])
    receipt(index, meta, "metadata", METADATA_FILES)
    receipt(index, meta, "text", TEXT_FILES)


def validate(index, meta, manifest):
    import numpy as np

    from .dense_index import METADATA_FILES, TEXT_FILES, verify_receipt
    from .local_index import array
    from .qwen_metadata import validate_metadata

    index = Path(index)
    if meta["caption_prompt_version"] != VERSION or meta["caption_prompt_sha256"] != PROMPT_SHA256:
        raise ValueError("v3 Qwen prompt provenance mismatch")
    if (
        meta["models"]["qwen"] != meta["caption_in_progress"]
        or meta["dense_provenance"]["qwen"] != meta["models"]["qwen"]
    ):
        raise ValueError("v3 Qwen model/provenance mismatch")
    if (
        meta["dense_provenance"].get("bge_identity") != meta["models"]["bge"]
        or meta["dense_provenance"]["bge_recipe"] != BGE_RECIPE
    ):
        raise ValueError("v3 BGE provenance mismatch")
    for phase, files in (("metadata", METADATA_FILES), ("text", TEXT_FILES)):
        if not verify_receipt(index, meta, phase, files):
            raise ValueError("v3 index not ready: " + phase)
    records = rows(index / METADATA_FILES[0])
    if len(records) != len(manifest) or MetadataStore(index / "metadata/v3").status()["SUCCESS"] != len(manifest):
        raise ValueError("v3 metadata row count mismatch")
    for record, window in zip(records, manifest):
        validate_record(record)
        if record.get("fallback") or any(
            record.get(k) != window[k] for k in ("row", "clip_id", "start_sec", "end_sec")
        ):
            raise ValueError("v3 row alignment or validity mismatch")
        original = json.loads((index / f"metadata/v3/rows/{record['row']:06d}.json").read_text())
        expected = to_dense(validate_metadata(original))
        if record["searchable_texts"] != expected["searchable_texts"]:
            raise ValueError("v3 dense adaptation mismatch")
        if (
            record["caption_sample_frame_ids"]
            != chronological_frame_ids(window["start_frame"], window["end_frame"], meta["caption_frame_count"]).tolist()
        ):
            raise ValueError("v3 frame sampling mismatch")
    for path in TEXT_FILES:
        value = array(index, path)
        if (
            value.shape != (len(manifest), meta["dimensions"]["bge"])
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or not np.allclose(np.linalg.norm(value, axis=1), 1, atol=2e-3)
        ):
            raise ValueError("v3 dense embedding invalid: " + path)
    if meta.get("segments"):
        from .segments import validation_artifact

        if validation_artifact(manifest, meta["segments"])["status"] != "PASS":
            raise ValueError("Cross-source semantic windows")
    return {"dense_metadata": "PASS", "schema_version": VERSION, "successful_windows": len(manifest)}
