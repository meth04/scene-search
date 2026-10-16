"""Isolated visual indexes and vectorized cosine search, with strict provenance."""
from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from .embeddinggemma2 import (
    DIMENSION,
    FORMAT,
    ROOT,
    EmbeddingGemma2Error,
    EmbeddingGemma2Worker,
    digest,
    identity,
    normalized,
    validate_config,
)

FROZEN_CONFIG = ROOT / "outputs/embeddinggemma2/embeddinggemma2_frozen_config.json"
if not FROZEN_CONFIG.is_file():
    FROZEN_CONFIG = ROOT / "configs/models/embeddinggemma2.json"


def index_identity(index):
    """Cheap warm-cache invalidation; cryptographic validation remains at load."""
    files = [Path(index) / name for name in ("index_meta.json", "manifest.jsonl", "embeddings.npy")]
    if FROZEN_CONFIG.is_file():
        files.append(FROZEN_CONFIG)
    try:
        return tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in files)
    except OSError as exc:
        raise EmbeddingGemma2Error("EG2_INDEX_NOT_READY", repr(exc)) from exc


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=True), encoding="utf-8")
    temp.replace(path)


def source_rows(source, selection=None):
    source = Path(source).resolve()
    meta = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
    manifest = [json.loads(s) for s in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    timeline_path = ROOT / "data/evaluation/roadscapes_hf_200/timeline.json"
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    by_id = {r["sequence_id"]: r for r in timeline}
    selected = selection if selection is not None else list(range(len(manifest)))
    rows = []
    for position, j in enumerate(selected):
        row = dict(manifest[j])
        t = by_id[row["segment_id"]]
        if row["source_clip_id"] != t["hf_path"] or row["start_sec"] != t["global_start_sec"] or row["end_sec"] != t["global_end_sec"]:
            raise EmbeddingGemma2Error("EG2_SOURCE_WINDOW_MISMATCH")
        if abs(row["end_sec"] - row["start_sec"] - 5) > 1e-6 or t["source_start_sec"] != 0 or t["source_end_sec"] != 5:
            raise EmbeddingGemma2Error("EG2_WINDOW_DURATION_MISMATCH")
        row.update(row=position, source_row=j, day_night=t["day_night"], source_path=t["source_path"],
            source_start_sec=t["source_start_sec"], source_end_sec=t["source_end_sec"],
            source_sha256=digest(t["source_path"]))
        row["window_sha256"] = identity(row)
        rows.append(row)
    provenance = {"path": str(source), "meta_sha256": digest(source / "index_meta.json"),
        "manifest_sha256": digest(source / "manifest.jsonl"), "timeline_path": str(timeline_path),
        "timeline_sha256": digest(timeline_path), "video": meta["video"]}
    return rows, provenance


def publish_index(target, rows, matrix, config, provenance, *, metrics=None, copy_privacy=True):
    target = Path(target).resolve()
    if target.exists():
        raise FileExistsError("EmbeddingGemma2 index already exists; no overwrite")
    validate_config(config)
    matrix = normalized(matrix)
    if matrix.shape != (len(rows), DIMENSION) or len({r["clip_id"] for r in rows}) != len(rows):
        raise EmbeddingGemma2Error("EG2_ROW_ALIGNMENT_MISMATCH")
    target.mkdir(parents=True)
    np.save(target / "embeddings.npy", matrix, allow_pickle=False)
    (target / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    meta = {"format": FORMAT, "ranker": "embeddinggemma2", "num_windows": len(rows),
        "window_sec": 5., "stride_sec": 5., "video": provenance["video"],
        "config": config, "config_sha256": identity(config), "source_index": provenance,
        "matrix_sha256": digest(target / "embeddings.npy"), "manifest_sha256": digest(target / "manifest.jsonl"),
        "metrics": metrics or {}, "phases": {"validate": {"finite": len(rows), "dimension": DIMENSION}},
        "decision_provider": "none", "score": "normalized_cosine", "qwen_required": False}
    write_json(target / "index_meta.json", meta)
    validate_index(target, expected_config=config)
    privacy = Path(provenance["path"]) / "privacy"
    if copy_privacy and privacy.is_dir():
        # Product renderers still validate source identity, cache version, boundary resets and coverage.
        shutil.copytree(privacy, target / "privacy")
        # Pilot and full indexes share the same full original timeline/cache. Reuse
        # an already verified private render without touching the source cache.
        peer = ROOT / "outputs/local_hybrid_indexes/roadscapes_hf_50_embeddinggemma2/privacy"
        receipt = peer / "private_media.json"
        if peer != target / "privacy" and receipt.is_file():
            try:
                render = json.loads(receipt.read_text(encoding="utf-8"))
                track_meta = json.loads((privacy / "privacy_meta.json").read_text(encoding="utf-8"))
                name = render["file"]
                if Path(name).name == name and render["tracks_sha256"] == track_meta["tracks_sha256"] and digest(peer / name) == render["sha256"]:
                    shutil.copy2(peer / name, target / "privacy" / name)
                    shutil.copy2(receipt, target / "privacy/private_media.json")
            except (OSError, ValueError, KeyError):
                # A missing render is handled fail-closed by the existing gateway.
                pass
    return target


def validate_index(index, expected_config=None):
    index = Path(index)
    try:
        meta = json.loads((index / "index_meta.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise EmbeddingGemma2Error("EG2_INDEX_FORMAT_MISMATCH")
        config = validate_config(meta["config"])
        if identity(config) != meta["config_sha256"]:
            raise EmbeddingGemma2Error("EG2_CONFIG_HASH_MISMATCH")
        if expected_config is None and FROZEN_CONFIG.is_file():
            expected_config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["config"]
        if expected_config is not None and config != expected_config:
            raise EmbeddingGemma2Error("EG2_FROZEN_CONFIG_MISMATCH")
        if digest(index / "embeddings.npy") != meta["matrix_sha256"] or digest(index / "manifest.jsonl") != meta["manifest_sha256"]:
            raise EmbeddingGemma2Error("EG2_INDEX_HASH_MISMATCH")
        matrix = np.load(index / "embeddings.npy", mmap_mode="r", allow_pickle=False)
        rows = [json.loads(s) for s in (index / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        if matrix.shape != (meta["num_windows"], DIMENSION) or len(rows) != len(matrix) or matrix.dtype != np.float32:
            raise EmbeddingGemma2Error("EG2_DIMENSION_OR_ROW_MISMATCH")
        normalized(matrix)
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-4):
            raise EmbeddingGemma2Error("EG2_UNNORMALIZED_INDEX")
        original = meta["source_index"]
        source = Path(original["path"])
        for name, expected in ((source / "index_meta.json", original["meta_sha256"]),
                               (source / "manifest.jsonl", original["manifest_sha256"]),
                               (Path(original["timeline_path"]), original["timeline_sha256"]),
                               (Path(meta["video"]["path"]), meta["video"]["sha256"])):
            if digest(name) != expected:
                raise EmbeddingGemma2Error("EG2_SOURCE_HASH_MISMATCH", str(name))
        source_manifest = [json.loads(s) for s in (source / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
        seen = set()
        for position, row in enumerate(rows):
            window = {k: v for k, v in row.items() if k != "window_sha256"}
            if identity(window) != row["window_sha256"] or row["row"] != position or row["clip_id"] in seen:
                raise EmbeddingGemma2Error("EG2_WINDOW_HASH_MISMATCH")
            seen.add(row["clip_id"])
            if row["end_sec"] - row["start_sec"] != 5 or digest(row["source_path"]) != row["source_sha256"]:
                raise EmbeddingGemma2Error("EG2_SOURCE_HASH_MISMATCH")
            for key in ("clip_id", "start_sec", "end_sec", "start_frame", "end_frame", "source_clip_id", "segment_id"):
                if row[key] != source_manifest[row["source_row"]][key]:
                    raise EmbeddingGemma2Error("EG2_ROW_ALIGNMENT_MISMATCH", key)
        return {"meta": meta, "matrix": matrix, "rows": rows, "ready": True}
    except EmbeddingGemma2Error:
        raise
    except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
        raise EmbeddingGemma2Error("EG2_INDEX_NOT_READY", repr(exc)) from exc


class EmbeddingGemma2SearchEngine:
    def __init__(self, index_path, *, worker_factory=EmbeddingGemma2Worker):
        self.index_path = Path(index_path).resolve()
        self.worker_factory = worker_factory
        self.lock = threading.RLock()
        self.loaded = self.worker = None
        self.load_count = self.query_count = 0

    def load(self):
        with self.lock:
            if self.loaded is None:
                started = time.perf_counter()
                loaded = validate_index(self.index_path)
                worker = self.worker_factory(loaded["meta"]["config"])
                try:
                    worker.load()
                except Exception:
                    worker.close()
                    raise
                self.loaded, self.worker = loaded, worker
                self.cold_load_seconds = time.perf_counter() - started
                self.load_count += 1
            return self

    def search_table(self, query, *, ranker="embeddinggemma2", decision_provider=None, **unused):
        if ranker != "embeddinggemma2":
            raise EmbeddingGemma2Error("EG2_INDEX_REQUIRES_EMBEDDINGGEMMA2")
        if decision_provider not in (None, "none"):
            raise EmbeddingGemma2Error("EG2_REQUIRES_DECISION_PROVIDER_NONE")
        if not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("INVALID_QUERY")
        with self.lock:
            self.load()
            started = time.perf_counter()
            vector, metrics = self.worker.encode_text(query)
            vector = normalized(vector)
            scoring = time.perf_counter()
            scores = self.loaded["matrix"] @ vector
            order = np.argsort(-scores, kind="stable")
            similarity_seconds = time.perf_counter() - scoring
            rows = self.loaded["rows"]
            candidates = [{"row": int(j), "source_row": rows[j]["source_row"], "clip_id": rows[j]["clip_id"],
                "source_clip_id": rows[j]["source_clip_id"], "segment_id": rows[j]["segment_id"],
                "start_sec": rows[j]["start_sec"], "end_sec": rows[j]["end_sec"], "day_night": rows[j]["day_night"],
                "rank": rank, "ranker": "embeddinggemma2", "final_score": float(scores[j]),
                "score": float(scores[j])} for rank, j in enumerate(order, 1)]
            self.query_count += 1
            config = self.loaded["meta"]["config"]
            return {"format": "local-embeddinggemma2-candidates-v1", "query": query,
                "index": str(self.index_path), "candidates": candidates,
                "decision": {"provider": "none", "state": "UNCALIBRATED", "threshold": None},
                "retrieval": {"index_windows": len(rows), "candidate_count": len(rows),
                    "ranker": "embeddinggemma2", "model_revision": config["revision"],
                    "embedding_dimension": DIMENSION, "representation": config["representation"],
                    "text_encode_seconds": metrics["seconds"], "similarity_seconds": similarity_seconds,
                    "warm_seconds": time.perf_counter() - started, "engine_load_count": self.load_count,
                    "query_count": self.query_count, "peak_VRAM_bytes": metrics["peak_VRAM_bytes"],
                    "Qwen_calls": 0, "external_model_API_calls": 0}}

    def close(self):
        with self.lock:
            if self.worker:
                self.worker.close()
            self.worker = self.loaded = None
