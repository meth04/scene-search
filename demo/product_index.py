"""Immutable arbitrary-video schema. Research TRAIN loaders remain unchanged."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from demo.product_assets import runtime_model_assets
from demo.product_privacy import validate as validate_privacy
from hybrid_v2.driveevidence_v6 import REPRESENTATION
from hybrid_v2.driveevidence_v6_selected import POLICY_HASH, REPRESENTATION_HASH, selected_config
from hybrid_v2.embeddinggemma2 import DIMENSION, ROOT, digest, identity
from src.privacy.tracks import probe, write_json

FORMAT = "driveevidence-v6-product-v1"
SCORERS = ("hybrid_v2/driveevidence_v6.py", "hybrid_v2/driveevidence_v6_rerank.py")
FAMILIES = ("WINDOW", "FRAME", "TUBE")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def array(path, value):
    target = Path(path)
    temporary = target.with_name(target.name + ".tmp")
    with temporary.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    os.replace(temporary, target)


def valid_vectors(value, count):
    return (
        value.dtype == np.float32
        and value.shape == (count, DIMENSION)
        and np.isfinite(value).all()
        and np.allclose(np.linalg.norm(value, axis=1), 1, atol=0.003)
    )


class Integrity:
    """Re-hash on stat change; identity verification stays off the warm GPU path."""

    def __init__(self):
        self.checked = {}
        self.privacy_validated = set()
        self.loaded = {}

    def check(self, path, checksum):
        path = Path(path)
        stat = path.stat()
        key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, checksum)
        if key not in self.checked:
            if digest(path) != checksum:
                raise ValueError("PRODUCT_ARTIFACT_HASH_MISMATCH")
            self.checked[key] = True

    def drop(self, folder):
        folder = Path(folder).resolve()
        self.checked = {k: v for k, v in self.checked.items() if not Path(k[0]).is_relative_to(folder)}
        self.loaded = {k: v for k, v in self.loaded.items() if not Path(k[0]).is_relative_to(folder)}
        self.privacy_validated = {k for k in self.privacy_validated if not Path(k[0]).is_relative_to(folder)}


def load_index(folder, *, integrity=None, expected_meta=None):
    folder = Path(folder).resolve()
    checker = integrity or Integrity()
    if expected_meta:
        checker.check(folder / "index_meta.json", expected_meta)
    cache_key = (str(folder), expected_meta or digest(folder / "index_meta.json"))
    saved = checker.loaded.get(cache_key)
    if saved is not None:
        meta = saved["meta"]
        pinned_manifest = read(ROOT / "configs/product/assets.json")
        pinned = runtime_model_assets(meta["model_hashes"], pinned_manifest)
        if read(ROOT / "configs/models/embeddinggemma2.json")["config"] != meta["config"]:
            raise ValueError("PRODUCT_FROZEN_IDENTITIES_CHANGED")
        for name, checksum in pinned.items():
            checker.check(ROOT / name, checksum)
        for name, checksum in meta["core_hashes"].items():
            checker.check(folder / name, checksum)
        for source in meta["sources"]:
            checker.check(Path(meta["source_root"]) / source["relative_path"], source["sha256"])
        try:
            for name, checksum in meta["local_hashes"].items():
                checker.check(folder / name, checksum)
            return saved
        except (OSError, ValueError):
            return {**saved, "families": {}, "evidence_fallback": "PRODUCT_LOCAL_EVIDENCE_INVALID"}
    meta = read(folder / "index_meta.json")
    frozen = read(ROOT / "configs/models/embeddinggemma2.json")["config"]
    if (
        meta["format"] != FORMAT
        or meta["schema"] != 1
        or meta["config"] != frozen
        or meta["representation"] != REPRESENTATION
        or meta["representation_sha256"] != REPRESENTATION_HASH
        or meta["experimental_policy"]
        != {"qualified": False, "config": selected_config(), "config_sha256": POLICY_HASH}
    ):
        raise ValueError("PRODUCT_SCHEMA_OR_FROZEN_CONFIG_INVALID")
    if meta["publication_identity"] != identity(
        {
            "build": meta["build_identity"],
            "sources": meta["sources"],
            "core": meta["core_hashes"],
            "local": meta["local_hashes"],
        }
    ):
        raise ValueError("PRODUCT_PUBLICATION_IDENTITY_INVALID")
    for name, checksum in meta["scorer_hashes"].items():
        if name not in SCORERS:
            raise ValueError("PRODUCT_SCORER_PATH_INVALID")
        checker.check(ROOT / name, checksum)
    pinned = runtime_model_assets(meta["model_hashes"], read(ROOT / "configs/product/assets.json"))
    for name, checksum in pinned.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError("PRODUCT_MODEL_PATH_INVALID")
        checker.check(path, checksum)
    for name, checksum in meta["core_hashes"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder):
            raise ValueError("PRODUCT_PATH_INVALID")
        checker.check(path, checksum)
    rows = read(folder / "manifest.json")["clips"]
    if not rows or len(rows) != meta["num_clips"] or len({r["clip_id"] for r in rows}) != len(rows):
        raise ValueError("PRODUCT_ROW_IDENTITY_INVALID")
    sources = {s["source_id"]: s for s in meta["sources"]}
    core_identity = identity(meta["core_hashes"])
    for source in sources.values():
        checker.check(Path(meta["source_root"]) / source["relative_path"], source["sha256"])
    for j, row in enumerate(rows):
        source = sources[row["source_id"]]
        if (
            row["row"] != j
            or row["source_sha256"] != source["sha256"]
            or not 0 <= row["start_sec"] < row["end_sec"] <= source["duration"] + 1e-6
            or row["end_sec"] - row["start_sec"] > 5.000001
            or row["privacy"] != "privacy-v2.1"
        ):
            raise ValueError("PRODUCT_ROW_SOURCE_INVALID")
        # Receipts and every privacy artifact are core hashes. The scanner's full
        # validation was performed before publication; changed bytes fail closed.
        receipt = read(folder / f"media/{j}/receipt.json")
        if receipt["complete"] is not True or receipt["format"] != "scene-search-privacy-v2.1":
            raise ValueError("PRODUCT_PRIVACY_INVALID")
        privacy_key = (str(folder), j, core_identity)
        if privacy_key not in checker.privacy_validated:
            validate_privacy(folder / f"media/{j}/raw.mp4", folder / f"media/{j}", row["end_sec"] - row["start_sec"])
            checker.privacy_validated.add(privacy_key)
    globals_ = np.load(folder / "global_vectors.npy", allow_pickle=False)
    if not valid_vectors(globals_, len(rows)):
        raise ValueError("PRODUCT_GLOBAL_VECTORS_INVALID")
    families, fallback = {}, None
    try:
        for name, checksum in meta["local_hashes"].items():
            checker.check(folder / name, checksum)
        for family in FAMILIES:
            prefix = family.lower()
            vectors = np.load(folder / f"{prefix}_vectors.npy", allow_pickle=False)
            offsets = np.load(folder / f"{prefix}_offsets.npy", allow_pickle=False)
            metadata = read(folder / f"{prefix}_metadata.json")
            if (
                not valid_vectors(vectors, len(metadata))
                or offsets.dtype != np.int64
                or offsets.shape != (len(rows) + 1,)
                or offsets[0] != 0
                or offsets[-1] != len(metadata)
                or (np.diff(offsets) < 0).any()
            ):
                raise ValueError("PRODUCT_LOCAL_SCHEMA_INVALID")
            for j, row in enumerate(rows):
                local = metadata[int(offsets[j]) : int(offsets[j + 1])]
                if (
                    family == "FRAME"
                    and len(local) != 8
                    or family == "WINDOW"
                    and len(local) != 3
                    or family == "TUBE"
                    and len(local) > 3
                ):
                    raise ValueError("PRODUCT_LOCAL_COUNT_INVALID")
                for unit in local:
                    if (
                        unit["clip_id"] != row["clip_id"]
                        or unit["source_sha256"] != row["source_sha256"]
                        or unit["family"] != family
                        or not 0 <= unit["start"] <= unit["end"] <= 5
                    ):
                        raise ValueError("PRODUCT_LOCAL_MAPPING_INVALID")
            families[family] = {"vectors": vectors, "offsets": offsets, "metadata": metadata}
    except (OSError, ValueError, KeyError, TypeError):
        families, fallback = {}, "PRODUCT_LOCAL_EVIDENCE_INVALID"
    loaded = {
        "meta": meta,
        "rows": rows,
        "global_vectors": globals_,
        "families": families,
        "evidence_fallback": fallback,
        "serving_manifest": meta["publication_identity"],
    }
    if not fallback:
        checker.loaded[cache_key] = loaded
    return loaded


def build(library, upload_id, actor):
    started = time.perf_counter()
    phases, model_metrics = {}, []
    record = library.uploads.get(upload_id, actor)
    root = library.uploads.path(upload_id, "_product")
    root.mkdir(parents=True, exist_ok=True)
    published = root / "published"
    # Crash after atomic rename but before catalog commit: verify and adopt.
    if published.exists():
        try:
            loaded = load_index(published)
            if loaded["evidence_fallback"]:
                raise ValueError("PUBLISHED_LOCAL_EVIDENCE_INVALID")
            return publication(published, len(loaded["rows"]))
        except (OSError, ValueError, KeyError, TypeError):
            # Preserve an invalid generation for diagnosis. It is never served,
            # edited in place or reused as a completed publication.
            quarantine = root / "quarantine"
            quarantine.mkdir(exist_ok=True)
            os.rename(published, quarantine / uuid.uuid4().hex)
    plan, sources = [], []
    scorer_hashes = {name: digest(ROOT / name) for name in SCORERS}
    model_hashes = library.asset_identity()
    build_identity = identity(
        {
            "config": library.config,
            "representation": REPRESENTATION,
            "policy": POLICY_HASH,
            "scorers": scorer_hashes,
            "assets": model_hashes,
            "privacy": library.privacy_identity(),
        }
    )
    source_info = {}
    estimated_cache = 0
    for file in record["files"]:
        info = probe(library.uploads.path(upload_id, file["relative_path"]))
        source_info[file["file_id"]] = info
        duration = info["frame_count"] / info["fps"]
        estimated_cache += int(np.ceil(duration / 5)) * (56 * info["width"] * info["height"] * 3 + 256 * 1024)
        estimated_cache += 5 * info["bytes"]
    existing_cache = (
        sum(p.stat().st_size for p in (root / "cache").rglob("*") if p.is_file()) if (root / "cache").exists() else 0
    )
    required_remaining = max(0, int(estimated_cache * 1.2) - existing_cache)
    if shutil.disk_usage(root).free < required_remaining + 256 * 1024**2:
        raise ValueError("INSUFFICIENT_PRODUCT_STORAGE")
    for file in record["files"]:
        source = library.uploads.path(upload_id, file["relative_path"])
        info = source_info[file["file_id"]]
        if info["sha256"] != file["sha256"]:
            raise ValueError("SOURCE_CHECKSUM_MISMATCH")
        duration = info["frame_count"] / info["fps"]
        if duration > 1800:
            raise ValueError("VIDEO_TOO_LONG")
        sources.append(
            {
                "source_id": file["file_id"],
                "relative_path": file["relative_path"],
                "sha256": file["sha256"],
                "duration": duration,
            }
        )
        for segment, start in enumerate(np.arange(0, duration, 5)):
            j, end = len(plan), min(duration, float(start) + 5)
            work = root / "cache" / f"{j:06d}"
            work.mkdir(parents=True, exist_ok=True)
            row_id = identity(
                {
                    "build": build_identity,
                    "source": file["sha256"],
                    "source_id": file["file_id"],
                    "start": float(start),
                    "end": end,
                }
            )
            raw = work / "raw.mp4"
            clip_receipt = work / "clip.json"
            try:
                saved = read(clip_receipt)
                assert saved["identity"] == row_id and digest(raw) == saved["raw_sha256"]
                assert digest(work / "inference.mp4") == saved["inference_sha256"]
            except (OSError, KeyError, ValueError, AssertionError):
                library.convert(
                    [
                        library.ffmpeg(),
                        "-v",
                        "error",
                        "-y",
                        "-ss",
                        str(start),
                        "-i",
                        str(source),
                        "-t",
                        str(end - start),
                        "-map",
                        "0:v:0",
                        "-an",
                        "-vf",
                        "scale=trunc(iw/2)*2:trunc(ih/2)*2",
                        "-c:v",
                        "libx264",
                        "-preset",
                        "veryfast",
                        "-crf",
                        "23",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        str(raw),
                    ]
                )
                inference = work / "inference.mp4"
                raw_info = probe(raw)
                if abs(raw_info["frame_count"] / raw_info["fps"] - 5) <= 0.15 and raw_info["frame_count"] >= max(
                    8, library.config["frame_count"]
                ):
                    shutil.copy2(raw, inference)
                else:
                    library.convert(
                        [
                            library.ffmpeg(),
                            "-v",
                            "error",
                            "-y",
                            "-i",
                            str(raw),
                            "-vf",
                            "tpad=stop_mode=clone:stop_duration=5,fps=30",
                            "-t",
                            "5",
                            "-an",
                            "-c:v",
                            "libx264",
                            "-preset",
                            "veryfast",
                            "-crf",
                            "23",
                            "-pix_fmt",
                            "yuv420p",
                            str(inference),
                        ]
                    )
                write_json(
                    clip_receipt, {"identity": row_id, "raw_sha256": digest(raw), "inference_sha256": digest(inference)}
                )
            plan.append(
                {
                    "row": j,
                    "clip_id": f"{upload_id}:{file['file_id']}:{segment}",
                    "source_id": file["file_id"],
                    "source_row": j,
                    "source_clip_id": file["file_id"],
                    "filename": file["relative_path"],
                    "source_sha256": file["sha256"],
                    "segment_id": segment,
                    "start_sec": float(start),
                    "end_sec": end,
                    "source_start_sec": float(start),
                    "source_end_sec": end,
                    "day_night": None,
                    "raw": str(raw),
                    "inference": str(work / "inference.mp4"),
                    "cache": str(work / "offline"),
                    "duration": end - float(start),
                    "identity": row_id,
                    "privacy": "privacy-v2.1",
                }
            )
    library.update(upload_id, stage="privacy_and_geometry", total_scenes=len(plan), completed_scenes=0)
    plan_path = root / "plan.json"
    write_json(plan_path, {"rows": plan})
    phases["prepare_seconds"] = time.perf_counter() - started
    phase_started = time.perf_counter()
    library.close()
    library.offline(plan_path)
    phases["privacy_geometry_seconds"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_driveevidence_v6 import specifications

    vectors, all_specs = [], []
    for item in plan:
        j = item["row"]
        library.update(upload_id, stage="embedding", progress=0.35 + 0.6 * j / len(plan), completed_scenes=j)
        work, offline = Path(item["raw"]).parent, Path(item["cache"])
        validate_privacy(item["raw"], offline / "privacy", item["duration"])
        geometry = offline / "geometry/000000"
        decode, objects = read(geometry / "decode.json"), read(geometry / "objects.json")["objects"]
        specs = specifications(item, objects, decode, REPRESENTATION)
        cache_identity = identity(
            {
                "row": item["identity"],
                "offline": digest(offline / "offline_receipt.json"),
                "specs": specs,
                "model": model_hashes,
            }
        )
        receipt = work / "embedding.json"
        try:
            saved = read(receipt)
            assert saved["identity"] == cache_identity
            assert all(digest(work / n) == h for n, h in saved["hashes"].items())
            global_ = np.load(work / "global.npy", allow_pickle=False)
            local = np.load(work / "local.npy", allow_pickle=False)
            assert valid_vectors(global_.reshape(1, -1), 1) and valid_vectors(local, len(specs))
        except (OSError, KeyError, ValueError, AssertionError):
            global_, global_metrics = library.encoder().encode_video(Path(item["inference"]), library.config)
            encoded = [
                library.encoder().encode_evidence(
                    geometry / "frames.npy",
                    spec["indices"],
                    boxes=spec.get("boxes"),
                    visual_tokens=REPRESENTATION["visual_tokens"],
                )
                for spec in specs
            ]
            local = np.asarray([result[0] for result in encoded], np.float32)
            if not valid_vectors(global_.reshape(1, -1), 1) or not valid_vectors(local, len(specs)):
                raise ValueError("PRODUCT_EMBEDDING_INVALID")
            array(work / "global.npy", global_)
            array(work / "local.npy", local)
            write_json(
                receipt,
                {
                    "identity": cache_identity,
                    "metrics": [global_metrics] + [r[1] for r in encoded],
                    "hashes": {n: digest(work / n) for n in ("global.npy", "local.npy")},
                },
            )
        model_metrics.extend(read(receipt).get("metrics", []))
        vectors.append(global_)
        all_specs.append((specs, local))
        library.update(upload_id, progress=0.35 + 0.6 * (j + 1) / len(plan), completed_scenes=j + 1)
    phases["embedding_seconds"] = time.perf_counter() - phase_started
    phase_started = time.perf_counter()
    library.update(upload_id, stage="publish")
    staging = root / (".publish-" + uuid.uuid4().hex)
    staging.mkdir()
    rows = [
        {k: v for k, v in item.items() if k not in {"raw", "inference", "cache", "identity", "duration"}}
        for item in plan
    ]
    array(staging / "global_vectors.npy", np.asarray(vectors, np.float32))
    write_json(staging / "manifest.json", {"clips": rows})
    local_names = []
    for family in FAMILIES:
        vv, mm, offsets = [], [], [0]
        for item, (specs, local) in zip(plan, all_specs):
            for spec, vector in zip(specs, local):
                if spec["family"] == family:
                    vv.append(vector)
                    mm.append({**spec, "clip_id": item["clip_id"], "source_sha256": item["source_sha256"]})
            offsets.append(len(vv))
        prefix = family.lower()
        array(staging / f"{prefix}_vectors.npy", np.asarray(vv, np.float32).reshape(-1, DIMENSION))
        array(staging / f"{prefix}_offsets.npy", np.asarray(offsets, np.int64))
        write_json(staging / f"{prefix}_metadata.json", mm)
        local_names += [f"{prefix}_{name}" for name in ("vectors.npy", "offsets.npy", "metadata.json")]
    for item in plan:
        target = staging / "media" / str(item["row"])
        shutil.copytree(Path(item["cache"]) / "privacy", target)
        shutil.copy2(item["raw"], target / "raw.mp4")
        validate_privacy(target / "raw.mp4", target, item["duration"])
    hashes = {str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()}
    meta = {
        "format": FORMAT,
        "schema": 1,
        "config": library.config,
        "representation": REPRESENTATION,
        "representation_sha256": REPRESENTATION_HASH,
        "experimental_policy": {"qualified": False, "config": selected_config(), "config_sha256": POLICY_HASH},
        "scorer_hashes": scorer_hashes,
        "model_hashes": model_hashes,
        "num_clips": len(rows),
        "source_root": str(library.uploads.path(upload_id, rows[0]["filename"]).parent)
        if len(sources) == 1 and "/" not in sources[0]["relative_path"]
        else str(library.root / "videos" / upload_id),
        "sources": sources,
        "core_hashes": {n: h for n, h in hashes.items() if n not in local_names},
        "local_hashes": {n: hashes[n] for n in local_names},
        "build_identity": build_identity,
        "privacy": "privacy-v2.1",
        "training_performed": False,
        "wall_seconds": time.perf_counter() - started,
    }
    meta["publication_identity"] = identity(
        {"build": build_identity, "sources": sources, "core": meta["core_hashes"], "local": meta["local_hashes"]}
    )
    write_json(staging / "index_meta.json", meta)
    loaded = load_index(staging)
    if loaded["evidence_fallback"]:
        raise ValueError("PRODUCT_PUBLICATION_VALIDATION_FAILED")
    os.rename(staging, published)  # Same volume; absent destination; readers see all or nothing.
    phases["validation_publication_seconds"] = time.perf_counter() - phase_started
    write_json(
        root / "build_metrics.json",
        {
            "phases": phases,
            "wall_seconds": time.perf_counter() - started,
            "clips": len(rows),
            "video_seconds": sum(s["duration"] for s in sources),
            "vectors": 1 * len(rows) + sum(len(specs) for specs, _ in all_specs),
            "peak_EG2_VRAM_bytes": max((m.get("peak_VRAM_bytes", 0) for m in model_metrics), default=0),
            "peak_EG2_RAM_bytes": max((m.get("RAM_bytes", 0) for m in model_metrics), default=0),
            "visual_encode_seconds": sum(m.get("seconds", 0) for m in model_metrics),
            "model_metrics_available": bool(model_metrics),
            "OOM": 0,
            "publication_identity": meta["publication_identity"],
        },
    )
    return publication(published, len(rows))


def publication(folder, count):
    return {
        "status": "ready",
        "stage": "ready",
        "progress": 1,
        "clips": count,
        "total_scenes": count,
        "completed_scenes": count,
        "privacy_state": "READY_PRIVACY_V2_1",
        "v6_state": "READY_V6_EXPERIMENTAL",
        "product_schema": FORMAT,
        "index_meta_sha256": digest(Path(folder) / "index_meta.json"),
        "error": None,
    }
