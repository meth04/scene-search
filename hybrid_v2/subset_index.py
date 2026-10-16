"""Slice verified source rows into a new aligned pilot index.

No re-inference: visual/motion arrays are copied by row from the frozen source
index, which is legitimate because all pilot windows come from the same video.
Text/BGE arrays are NOT copied from the source (they belong to the source
metadata contract); the V4.1 builder regenerates them with the exact BGE recipe.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .dense_metadata import FIELDS
from .local_index import PHASES, complete_phase, read_meta, rows
from .local_runtime import atomic_array, atomic_json, atomic_jsonl, sha256

SLICE_PHASES = PHASES[:5]  # prepare, visual, motion, fuse, frame-project
_VISUAL = {
    "visual": ["visual/raw_frame_features.npy", "visual/base_clip_embeddings.npy"],
    "motion": ["visual/tracks.npy", "visual/track_masks.npy"],
    "fuse": ["visual/fused_embeddings.npy"],
    "frame-project": ["visual/projected_frame_embeddings.npy"],
}
_CAPTION_TEXT = [f"text/caption_{name}.npy" for name in ("full", "action", "relation", "context")]
_DENSE_TEXT = [f"text/dense_{field}.npy" for field in FIELDS]


def _slice_array(source, output, relative, selected):
    data = np.load(source / relative, mmap_mode="r")
    atomic_array(output / relative, np.asarray(data[selected]))


def build_subset(source, output, selected_rows, *, with_caption_text=False, dense_store=None):
    """Create `output` as a new index of `selected_rows` from `source`.

    Always copies the verified visual/motion arrays and the manifest. When
    `dense_store` is given (e.g. "metadata/v3" or "metadata/v4"), also slices the
    dense metadata arrays and copies+renumbers that store's per-row files and
    receipts so the sliced index passes its own validator.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite subset index: {output}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Subset source and output must be independent directories")
    selected = [int(r) for r in selected_rows]
    source_manifest = rows(source / "manifest.jsonl")
    if len(selected) != len(set(selected)) or any(not 0 <= r < len(source_manifest) for r in selected):
        raise ValueError("Invalid subset row selection")

    source_meta = read_meta(source)
    manifest = []
    for new_row, old in enumerate(selected):
        row = dict(source_manifest[old])
        row["row"] = new_row
        manifest.append(row)

    output.mkdir(parents=True)
    # The prepare-phase receipt hashes manifest.jsonl, so write it first.
    atomic_jsonl(output / "manifest.jsonl", manifest)
    for phase in SLICE_PHASES:
        for relative in _VISUAL.get(phase, []):
            _slice_array(source, output, relative, selected)

    captions = rows(source / "captions/captions.jsonl")
    new_captions = []
    for new_row, old in enumerate(selected):
        caption = dict(captions[old])
        caption["row"] = new_row
        new_captions.append(caption)
    atomic_jsonl(output / "captions/captions.jsonl", new_captions)

    meta = copy.deepcopy(source_meta)
    meta["num_windows"] = len(selected)
    meta["created_at"] = datetime.now(UTC).isoformat()
    meta["phases"] = {}
    meta["subset"] = {
        "source_index": str(source),
        "source_meta_sha256": sha256(source / "index_meta.json"),
        "source_video_sha256": source_meta["video"]["sha256"],
        "rows": selected,
    }
    used_segments = {row["segment_id"] for row in manifest}
    meta["segments"] = [s for s in source_meta.get("segments", []) if s["segment_id"] in used_segments]
    for phase in SLICE_PHASES:
        complete_phase(output, meta, phase, {"sliced_from": str(source)})
    complete_phase(output, meta, "caption", {"sliced_captions": len(new_captions)})

    if dense_store:
        for relative in _CAPTION_TEXT:
            _slice_array(source, output, relative, selected)
        complete_phase(output, meta, "text-index", {"sliced_caption_text": True})
        _slice_dense(source, output, selected, dense_store, meta)
    elif with_caption_text:
        for relative in _CAPTION_TEXT:
            _slice_array(source, output, relative, selected)
        complete_phase(output, meta, "text-index", {"sliced_caption_text": True})

    atomic_json(output / "index_meta.json", meta)
    return manifest


def _slice_dense(source, output, selected, dense_store, meta):
    dense = rows(source / "metadata/dense_metadata.jsonl")
    new_dense = []
    for new_row, old in enumerate(selected):
        record = copy.deepcopy(dense[old])
        record["row"] = new_row
        new_dense.append(record)
    atomic_jsonl(output / "metadata/dense_metadata.jsonl", new_dense)
    for relative in _DENSE_TEXT:
        _slice_array(source, output, relative, selected)

    store = Path(dense_store)
    for kind in ("rows", "receipts"):
        for new_row, old in enumerate(selected):
            src = source / store / kind / f"{old:06d}.json"
            if not src.exists():
                continue
            value = json.loads(src.read_text(encoding="utf-8"))
            if isinstance(value, dict) and "window_id" in value:
                value["window_id"] = f"{new_row:06d}"
            atomic_json(output / store / kind / f"{new_row:06d}.json", value)
    sheets = source / "metadata/contact_sheets"
    if sheets.is_dir():
        import shutil

        (output / "metadata/contact_sheets").mkdir(parents=True, exist_ok=True)
        for new_row, old in enumerate(selected):
            src = sheets / f"{old:06d}.jpg"
            if src.exists():
                shutil.copyfile(src, output / "metadata/contact_sheets" / f"{new_row:06d}.jpg")

    fallback = sum(bool(r.get("fallback")) for r in new_dense)
    stats = {
        "total": len(new_dense), "ok": len(new_dense) - fallback, "fallback": fallback,
        "fallback_rate": fallback / len(new_dense),
        "warning_count": sum(len(r.get("metadata_warnings", [])) for r in new_dense),
        "warning_rows": sum(bool(r.get("metadata_warnings")) for r in new_dense),
    }
    meta["caption_stats"] = {k: stats[k] for k in ("total", "ok", "fallback", "fallback_rate")}
    meta["dense_stats"] = stats
    if meta.get("dense_provenance"):
        meta["dense_provenance"] = copy.deepcopy(meta["dense_provenance"])
        meta["dense_provenance"]["stats"] = stats
        meta["dense_provenance"]["subset"] = {"rows": selected, "source_index": str(source)}
        atomic_json(output / "metadata/metadata_meta.json", meta["dense_provenance"])
    # Receipts must be recomputed for the sliced artifacts (source hashes are stale).
    from .dense_index import METADATA_FILES, TEXT_FILES

    meta["dense_receipts"] = {
        "metadata": {p: sha256(output / p) for p in METADATA_FILES},
        "text": {p: sha256(output / p) for p in TEXT_FILES},
    }
