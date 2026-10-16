"""Resumable TRAIN-only V6 evidence build; prior extraction remains read-only."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from hybrid_v2.driveevidence_v6 import (FORMAT, FACTOR_SCHEMA, REPRESENTATION, complexity,
    factor_atoms, sample_positions, temporal_windows, select_tracks, tube_spec)
from hybrid_v2.driveevidence_v6_worker import DriveEvidenceV6Worker
from hybrid_v2.driveevidence_v5_planner import QueryPlanner, REVISION as LAYA_REVISION
from hybrid_v2.embeddinggemma2 import REVISION, DIMENSION
from prepare_driveevidence_v6 import OUT, V3, V5, read, write_json, digest, identity


def specifications(row, tracks, decode, cfg):
    times = decode["timestamps"]
    specs = []
    for position in sample_positions(times, cfg["frame_count"]):
        stamp = times[position]
        specs.append({"family": "FRAME", "indices": [position], "timestamps": [stamp], "start": stamp, "end": stamp})
    for start, end in temporal_windows(5., cfg["window_length"], cfg["window_stride"]):
        indices = sample_positions(times, cfg["window_frames"], start, min(end, times[-1]))
        specs.append({"family": "WINDOW", "indices": indices, "timestamps": [times[i] for i in indices], "start": start, "end": end})
    for track in select_tracks(tracks, cfg):
        specs.append(tube_spec(track, decode["width"], decode["height"], cfg))
    return specs


def vector_cache_identity(row, decode, spec, cfg):
    return identity({"source_sha256": row["source_sha256"], "decoded_frames_sha256": decode["frames_sha256"],
                     "spec": spec, "visual_tokens": cfg["visual_tokens"], "EG2_revision": REVISION,
                     "dimension": DIMENSION, "dtype": "float32", "unit_schema": "v6-evidence-1"})


def build_factors(worker, protocol):
    queries = [q for q in read(OUT / "dev/benchmark/queries.json")["queries"] if q["id"] in protocol["eligible_query_ids"]]
    planner = QueryPlanner(cache=V5 / "query_planner_cache")
    factor_vectors, records = {}, []
    existing_payload = read(V5 / "dev/benchmark/factor_vectors.json")
    existing = existing_payload["vectors"]
    global_config = read(V3 / "dev/index/index_meta.json")["config"]
    assert existing_payload["config_sha256"] == identity(global_config), "V5_FACTOR_CONFIG_MISMATCH"
    query_hashes = {r["id"]: r["text_vector_sha256"] for r in read(OUT / "dev/benchmark/EG2_baseline.json")["records"]}
    for q in queries:
        saved = planner.cached(q["query"])
        if saved is None or saved.get("plan") is None or saved.get("failure"):
            raise ValueError("INVALID_FROZEN_LAYA_PLAN " + q["id"])
        # Copy the pinned query-only raw output into V6-owned cache; never touch V5.
        owned = OUT / "query_planner_cache" / (identity(" ".join(q["query"].casefold().split())) + ".json")
        if not owned.exists():
            write_json(owned, saved)
        atoms_by_template = {}
        for template in ("concise", "bound"):
            atoms = factor_atoms(q["query"], saved["plan"], template)
            atoms_by_template[template] = atoms
            for atom in atoms:
                text = atom["text"]
                if text in factor_vectors:
                    continue
                key = identity({"text": text, "EG2_revision": REVISION, "global_config": global_config})
                path = OUT / "dev/factor_text" / (key + ".json")
                if path.exists():
                    result = read(path)
                    assert result["text"] == text and result["config_sha256"] == identity(global_config)
                else:
                    if text == q["query"]:
                        global_text = V3 / "dev/retrieval/text" / (q["id"] + ".json")
                        assert digest(global_text) == query_hashes[q["id"]], "GLOBAL_TEXT_IDENTITY_CHANGED"
                        old = read(global_text)
                        vector, metrics = old["embedding"], {"reused": "frozen original TRAIN global text"}
                    elif text in existing:
                        vector, metrics = existing[text], {"reused": "frozen V5 EG2 factor text"}
                    else:
                        v, metrics = worker.encode_text(text)
                        vector = v.tolist()
                    result = {"text": text, "embedding": vector, "metrics": metrics, "config_sha256": identity(global_config), "EG2_revision": REVISION}
                    write_json(path, result)
                factor_vectors[text] = result["embedding"]
        records.append({"id": q["id"], "route": complexity(q["query"]), "atoms": atoms_by_template,
                        "plan_identity": identity(saved), "Laya_new_calls": 0, "Laya_source": "validated pinned V5 query-only raw output"})
    write_json(OUT / "dev/benchmark/factors.json", {"schema": FACTOR_SCHEMA, "Laya_revision": LAYA_REVISION,
               "records": records, "vectors": factor_vectors, "new_Laya_calls": 0, "candidate_level_Laya_calls": 0,
               "EG2_revision": REVISION})
    print("V6 factor text complete", len(records), len(factor_vectors), flush=True)


def publish(rows, all_specs, cfg, representation_id):
    target = OUT / "dev/indexes" / representation_id
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(V3 / "dev/index/global_eg2.npy", target / "global_vectors.npy")
    assert digest(target / "global_vectors.npy") == digest(V3 / "dev/index/global_eg2.npy")
    costs, counts = {}, {}
    for family in ("WINDOW", "FRAME", "TUBE"):
        vectors, metadata, offsets = [], [], [0]
        for row, specs in zip(rows, all_specs):
            for spec in specs:
                if spec["family"] != family:
                    continue
                vector_file = OUT / "dev/evidence_vectors" / (spec["cache_key"] + ".npy")
                vector = np.load(vector_file, allow_pickle=False)
                assert vector.shape == (DIMENSION,) and np.isfinite(vector).all() and np.isclose(np.linalg.norm(vector), 1, atol=.003)
                vectors.append(vector)
                metadata.append({"clip_id": row["clip_id"], "source_sha256": row["source_sha256"], **spec})
            offsets.append(len(vectors))
        prefix = family.casefold()
        np.save(target / (prefix + "_vectors.npy"), np.asarray(vectors, np.float32).reshape(-1, DIMENSION))
        np.save(target / (prefix + "_offsets.npy"), np.asarray(offsets, np.int64))
        write_json(target / (prefix + "_metadata.json"), metadata)
        np.savez_compressed(target / (prefix + "_meta.npz"),
            start=np.asarray([m["start"] for m in metadata], np.float32),
            end=np.asarray([m["end"] for m in metadata], np.float32),
            track_id=np.asarray([m.get("track_id", -1) for m in metadata], np.int32),
            detector_class=np.asarray([m.get("detector_class", "") for m in metadata]))
        counts[family] = len(vectors)
        costs[family] = (target / (prefix + "_vectors.npy")).stat().st_size
    write_json(target / "manifest.json", {"clips": rows})
    write_json(target / "provenance.json", {"TRAIN_only": True, "representation": cfg,
               "representation_sha256": identity(cfg), "original_global_sha256": digest(V3 / "dev/index/global_eg2.npy"),
               "source_tracks_sha256": digest(V3 / "dev/index/track_records.json"), "no_training": True,
               "tube_encoding": "One official EG2 chronological video input of expanded per-observation crops; no static-vector averaging",
               "frame_encoding": "official EG2 image encoder", "evidence_vectors": "frozen EG2 normalized float32 768d"})
    hashes = {f.name: digest(f) for f in target.iterdir() if f.is_file() and f.name != "index_meta.json"}
    write_json(target / "index_meta.json", {"format": FORMAT, "schema": 1, "ranker": "driveevidence_v6",
               "config": read(V3 / "dev/index/index_meta.json")["config"], "representation": cfg,
               "representation_sha256": identity(cfg), "hashes": hashes, "num_clips": len(rows),
               "privacy_version": "privacy-v2.1", "policy": {"enabled": False, "state": "ORACLE_ONLY"},
               "model_hashes": read(V5 / "index/serve_index/index_meta.json")["model_hashes"], "time_unix": time.time()})
    total = sum(f.stat().st_size for f in target.iterdir() if f.is_file())
    receipt = {"index": str(target), "clips": len(rows), "vectors": counts, "vectors_per_clip": (1 + sum(counts.values()) / len(rows)),
               "vector_bytes": {"GLOBAL": (target / "global_vectors.npy").stat().st_size, **costs},
               "metadata_bytes": total - sum(costs.values()) - (target / "global_vectors.npy").stat().st_size,
               "serve_index_bytes": total, "bytes_per_clip": total / len(rows),
               "storage_projections": {str(n): int(total / len(rows) * n) for n in (10000, 100000, 1000000)}}
    write_json(OUT / "dev/builds" / (representation_id + ".json"), receipt)
    return receipt


def build(cfg=None, smoke=False, factors_only=False):
    started = time.perf_counter()
    protocol = read(OUT / "dev/protocol_frozen.json")
    assert digest(V3 / "dev/index/track_records.json") == protocol["track_records_sha256"]
    cfg = cfg or dict(REPRESENTATION)
    representation_id = identity(cfg)[:16]
    rows = read(V3 / "dev/corpus/manifest.json")["clips"]
    tracks = read(V3 / "dev/index/track_records.json")
    # Check pinned model weights before entering persistent inference.
    for name, expected in read(V5 / "index/serve_index/index_meta.json")["model_hashes"].items():
        assert digest(ROOT / name) == expected, "MODEL_HASH_MISMATCH " + name
    worker = DriveEvidenceV6Worker(read(V3 / "dev/index/index_meta.json")["config"])
    all_specs, metrics = [], []
    vector_dir = OUT / "dev/evidence_vectors"
    vector_dir.mkdir(parents=True, exist_ok=True)
    try:
        worker.load()
        write_json(OUT / "dev/builds" / (representation_id + "_load.json"), worker.metrics)
        build_factors(worker, protocol)
        if factors_only:
            return {"factors_only": True, "wall_seconds": time.perf_counter() - started}
        for i, row in enumerate(rows[:1] if smoke else rows):
            clip_start = time.perf_counter()
            assert row["split"] == "train", "V6_BUILD_NON_TRAIN_FORBIDDEN"
            cache = V3 / "dev/evidence_cache" / f"{i:06d}"
            decode = read(cache / "decode.json")
            assert decode["source_sha256"] == row["source_sha256"]
            assert digest(cache / "frames.npy") == decode["frames_sha256"], "DECODE_CACHE_MISMATCH"
            specs = specifications(row, tracks[i], decode, cfg)
            reused = 0
            for spec in specs:
                key = vector_cache_identity(row, decode, spec, cfg)
                vector_path, receipt_path = vector_dir / (key + ".npy"), vector_dir / (key + ".json")
                if vector_path.exists() and receipt_path.exists():
                    receipt = read(receipt_path)
                    assert receipt["vector_sha256"] == digest(vector_path) and receipt["identity"] == key
                    reused += 1
                else:
                    vector, measurement = worker.encode_evidence(cache / "frames.npy", spec["indices"],
                        boxes=spec.get("boxes"), visual_tokens=cfg["visual_tokens"])
                    np.save(vector_path, vector)
                    receipt = {"identity": key, "source_sha256": row["source_sha256"], "spec": spec,
                               "metrics": measurement, "vector_sha256": digest(vector_path), "time_unix": time.time()}
                    write_json(receipt_path, receipt)
                spec["cache_key"] = key
                metrics.append(receipt["metrics"])
            all_specs.append(specs)
            progress = {"row": i, "completed": i + 1, "required": 1 if smoke else len(rows), "representation_id": representation_id,
                        "clip_seconds": time.perf_counter() - clip_start, "vector_count": len(specs), "cache_reused": reused,
                        "worker_pid": worker.process.pid, "time_unix": time.time()}
            write_json(OUT / "dev/builds" / (representation_id + "_progress.json"), progress)
            with (OUT / "dev/builds" / (representation_id + "_clips.jsonl")).open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(progress) + "\n")
            print("V6 evidence", i + 1, "/", len(rows), "vectors", len(specs), "reused", reused, "seconds", round(progress["clip_seconds"], 3), flush=True)
        receipt = {"smoke": smoke, "representation": cfg, "representation_id": representation_id,
                   "wall_seconds": time.perf_counter() - started, "encoded_or_reused_vectors": len(metrics),
                   "peak_VRAM_bytes": max(m["peak_VRAM_bytes"] for m in metrics),
                   "peak_RAM_bytes": max(m["RAM_bytes"] for m in metrics), "OOM_count": 0, "model_load_count": worker.metrics["load_count"]}
        if not smoke:
            receipt.update(publish(rows, all_specs, cfg, representation_id))
        write_json(OUT / "dev/builds" / (representation_id + "_complete.json"), receipt)
        print(json.dumps(receipt), flush=True)
        return receipt
    finally:
        worker.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--factors-only", action="store_true")
    args = parser.parse_args()
    build(read(args.config) if args.config else None, args.smoke, args.factors_only)
