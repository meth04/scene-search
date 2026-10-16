"""New V4 derivatives reuse verified visual artifacts; V3 indexes remain immutable."""

from __future__ import annotations

import base64
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from . import metadata_v4 as contract
from .common import chronological_frame_ids
from .dense_index import METADATA_FILES, TEXT_FILES, receipt, verify_receipt
from .dense_metadata import BGE_RECIPE, DENSE_FORMAT, FIELDS
from .local_index import array, complete_phase, read_meta, reuse_visual_index, rows
from .local_runtime import atomic_array, atomic_jsonl
from .qwen_provider import QwenError, create_provider, jpeg_data_url
from .qwen_store import MetadataStore, atomic_json, digest


def generate(source_index, output, *, selection=None, full=False, qwen_config=None):
    """Resume only in a distinct V4 store bound to exact images/settings/prompt."""
    from .local_video_index import contact_sheet

    source, output = Path(source_index), Path(output)
    if full and not output.exists():
        reuse_visual_index(source, output)
    meta = read_meta(output if full else source)
    if full and meta.get("v4_native") and "validate" in meta.get("phases", {}):
        raise ValueError("Published V4 index is immutable; choose a new output")
    windows = rows(source / "manifest.jsonl")
    if selection is not None:
        windows = [windows[i] for i in selection]
    settings = {
        "qwen_mode": "server",
        "caption_frames": 8,
        "jpeg_quality": 85,
        "tile_size": [256, 144],
        "concurrency": 1,
        "temperature": 0,
        "max_tokens": contract.MAX_TOKENS,
    }
    provider = create_provider(
        qwen_config or {"qwen_mode": "server"}, contract=contract, evidence_dir=output / "metadata/requests"
    )
    settings["qwen_mode"] = provider.config.mode
    store = MetadataStore(output / "metadata/v4", contract=contract)
    values, errors = [], []
    started = time.perf_counter()
    try:
        health = provider.healthcheck()
        atomic_json(output / "metadata/health.json", health)
        for w in windows:
            sheet = contact_sheet(w["video_path"], w, 256, 144, 8)
            url = jpeg_data_url(sheet, 85)
            target = output / f"metadata/contact_sheets/{w['row']:06d}.jpg"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(base64.b64decode(url.split(",", 1)[1]))
            try:
                value = store.generate(
                    f"{w['row']:06d}",
                    [url],
                    provider,
                    {
                        "index_id": output.name,
                        "source_identity": {
                            "sha256": meta["video"]["sha256"],
                            "clip_id": w["clip_id"],
                            "segment_id": w.get("segment_id"),
                        },
                        "generation_settings": settings,
                    },
                )
                record = contract.to_dense(value)
                record.update({k: w[k] for k in ("row", "clip_id", "start_sec", "end_sec")})
                record["caption_sample_frame_ids"] = chronological_frame_ids(
                    w["start_frame"], w["end_frame"], 8
                ).tolist()
                values.append(record)
                print(f"[V4] {w['row']:03d} SUCCESS bytes={len(json.dumps(value).encode())}", flush=True)
            except QwenError as exc:
                errors.append({"row": w["row"], "code": exc.code})
                print(f"[V4] {w['row']:03d} {exc.code}", flush=True)
                if provider.clock() < provider.open_until:
                    break
    finally:
        atomic_json(
            output / "metadata/metrics.json",
            {
                **provider.metrics(),
                "wall_seconds": time.perf_counter() - started,
                "success": len(values),
                "total": len(windows),
                "errors": errors,
                "settings": settings,
            },
        )
        atomic_json(output / "metadata/window_states.json", store.status())
        provider.close()
    if not full:
        return values
    if len(values) != len(windows) or errors:
        raise RuntimeError("V4 generation incomplete; unpublished resumable rows retained")
    atomic_jsonl(output / METADATA_FILES[0], values)
    captions = []
    for record in values:
        text = record["searchable_texts"]
        captions.append(
            {
                **{k: record[k] for k in ("row", "clip_id", "start_sec", "end_sec", "caption_sample_frame_ids")},
                "description": text["full_text"],
                "objects": [],
                "ego_action": "",
                "other_actions": [text["action_text"]],
                "relations": [text["relation_text"]],
                "road_context": [text["scene_text"]],
                "traffic_control": [text["infrastructure_text"]],
                "weather": "",
                "events": [],
                "queries": [],
                "fallback": False,
            }
        )
    atomic_jsonl(output / "captions/captions.jsonl", captions)
    identity = {
        "schema_version": contract.VERSION,
        "prompt_version": contract.VERSION,
        "prompt_sha256": contract.PROMPT_SHA256,
        "served_model": provider.model,
        "base_url": provider.config.base_url,
        "mode": provider.config.mode,
        "frame_count": 8,
        "jpeg_quality": 85,
        "tile_size": [256, 144],
        "sheet_layout": [2, 4],
    }
    meta.update(
        format=DENSE_FORMAT,
        index_format=DENSE_FORMAT,
        index_version=1,
        v3_native=False,
        v4_native=True,
        dense_metadata_version=contract.VERSION,
        caption_prompt_version=contract.VERSION,
        caption_prompt_sha256=contract.PROMPT_SHA256,
        caption_frame_count=8,
        caption_in_progress=identity,
        caption_stats={"total": len(values), "ok": len(values), "fallback": 0, "fallback_rate": 0.0},
    )
    meta["models"]["qwen"] = identity
    meta["dense_provenance"] = {"schema_version": contract.VERSION, "qwen": identity, "bge_recipe": BGE_RECIPE}
    atomic_json(output / METADATA_FILES[1], meta["dense_provenance"])
    complete_phase(output, meta, "caption", {"schema_version": contract.VERSION})
    receipt(output, meta, "metadata", METADATA_FILES)
    return values


def embed(output):
    from .text_encoder import BGEEncoder

    output = Path(output)
    meta = read_meta(output)
    records = rows(output / METADATA_FILES[0])
    encoder = BGEEncoder(
        meta.get("settings", {}).get("bge_model", "BAAI/bge-base-en-v1.5"),
        "cpu",
        cache_dir=meta["settings"]["hf_cache_dir"],
        local_files_only=True,
    )
    started = time.perf_counter()
    truncated = {}
    for field in FIELDS:
        texts = [r["searchable_texts"][field + "_text"] or "none" for r in records]
        truncated[field] = sum(len(encoder.tok.encode(t)) > 128 for t in texts)
        atomic_array(output / f"text/dense_{field}.npy", encoder.encode(texts, batch_size=8))
    for field, dense in (("full", "full"), ("action", "action"), ("relation", "relation"), ("context", "scene")):
        atomic_array(
            output / f"text/caption_{field}.npy",
            encoder.encode([r["searchable_texts"][dense + "_text"] or "none" for r in records], batch_size=8),
        )
    meta["models"]["bge"] = encoder.identity
    meta["dimensions"]["bge"] = int(encoder.model.config.hidden_size)
    meta["dense_provenance"]["bge_identity"] = encoder.identity
    meta["v4_text_timing"] = {
        "seconds": time.perf_counter() - started,
        "truncated_rows_by_field": truncated,
        "max_length": 128,
    }
    atomic_json(output / METADATA_FILES[1], meta["dense_provenance"])
    receipt(output, meta, "metadata", METADATA_FILES)
    receipt(output, meta, "text", TEXT_FILES)
    complete_phase(output, meta, "text-index", meta["v4_text_timing"])


def validate(index, meta, manifest):
    index = Path(index)
    health = json.loads((index / "metadata/health.json").read_text(encoding="utf-8"))
    if (
        not meta.get("v4_native")
        or meta["caption_prompt_sha256"] != contract.PROMPT_SHA256
        or meta["caption_prompt_version"] != contract.VERSION
    ):
        raise ValueError("V4 prompt/schema mismatch")
    if (
        meta["models"]["qwen"] != meta["caption_in_progress"]
        or meta["models"]["qwen"] != meta["dense_provenance"]["qwen"]
    ):
        raise ValueError("V4 provider provenance mismatch")
    if (
        meta["models"]["bge"] != meta["dense_provenance"]["bge_identity"]
        or meta["dense_provenance"]["bge_recipe"] != BGE_RECIPE
    ):
        raise ValueError("V4 BGE recipe/provenance mismatch")
    for phase, files in (("metadata", METADATA_FILES), ("text", TEXT_FILES)):
        if not verify_receipt(index, meta, phase, files):
            raise ValueError("V4 index not ready: " + phase)
    records = rows(index / METADATA_FILES[0])
    if len(records) != len(manifest) or MetadataStore(index / "metadata/v4", contract=contract).status()[
        "SUCCESS"
    ] != len(manifest):
        raise ValueError("V4 row count mismatch")
    for record, w in zip(records, manifest):
        path = index / f"metadata/v4/rows/{w['row']:06d}.json"
        value = contract.parse_metadata(path.read_text(encoding="utf-8"))
        rcp = json.loads((index / f"metadata/v4/receipts/{w['row']:06d}.json").read_text())
        if (
            rcp["sha256"] != digest(path)
            or rcp["state"] != "SUCCESS"
            or rcp["window_id"] != f"{w['row']:06d}"
            or rcp["fingerprint"] != hashlib.sha256(json.dumps(rcp["identity"], sort_keys=True).encode()).hexdigest()
            or rcp["identity"]["prompt_sha256"] != contract.PROMPT_SHA256
            or rcp["identity"]["schema_version"] != contract.VERSION
        ):
            raise ValueError("V4 receipt provenance mismatch")
        sheet = index / f"metadata/contact_sheets/{w['row']:06d}.jpg"
        image_url = "data:image/jpeg;base64," + base64.b64encode(sheet.read_bytes()).decode("ascii")
        source_identity = {
            "sha256": meta["video"]["sha256"],
            "clip_id": w["clip_id"],
            "segment_id": w.get("segment_id"),
        }
        if (
            rcp["identity"]["image_sha256"] != [hashlib.sha256(image_url.encode()).hexdigest()]
            or rcp["identity"]["source_identity"] != source_identity
        ):
            raise ValueError("V4 source/frame provenance mismatch")
        provider = rcp["identity"]["provider"]
        settings = rcp["identity"]["generation_settings"]
        qwen = meta["models"]["qwen"]
        if (
            provider["mode"] != qwen["mode"]
            or provider["base_url"] != qwen["base_url"]
            or provider["model"] != health["requested_model"]
            or settings
            != {
                "qwen_mode": qwen["mode"],
                "caption_frames": 8,
                "jpeg_quality": 85,
                "tile_size": [256, 144],
                "concurrency": 1,
                "temperature": 0,
                "max_tokens": contract.MAX_TOKENS,
            }
            or not rcp.get("timestamp")
        ):
            raise ValueError("V4 generation settings/provider mismatch")
        if record["searchable_texts"] != contract.to_dense(value)["searchable_texts"] or record.get("fallback"):
            raise ValueError("V4 deterministic search text mismatch")
        if (
            any(record[k] != w[k] for k in ("row", "clip_id", "start_sec", "end_sec"))
            or record["caption_sample_frame_ids"]
            != chronological_frame_ids(w["start_frame"], w["end_frame"], 8).tolist()
        ):
            raise ValueError("V4 alignment mismatch")
    for file in TEXT_FILES:
        v = array(index, file)
        if (
            v.dtype != np.float32
            or v.shape != (len(manifest), 768)
            or not np.isfinite(v).all()
            or not np.allclose(np.linalg.norm(v, axis=1), 1, atol=2e-3)
        ):
            raise ValueError("V4 invalid embedding: " + file)
    from .segments import validation_artifact

    if validation_artifact(manifest, meta["segments"])["status"] != "PASS":
        raise ValueError("V4 cross-clip contamination")
    return {"dense_metadata": "PASS", "schema_version": contract.VERSION, "successful_windows": len(records)}
