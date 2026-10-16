"""Frozen best measured V6 serving contract, separate from scientific qualification."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from .driveevidence_v6 import REPRESENTATION
from .driveevidence_v6 import load_index as load_research_index
from .driveevidence_v6_rerank import DEFAULT
from .embeddinggemma2 import DIMENSION, REVISION, digest, identity, validate_config

SELECTED_CONFIG = copy.deepcopy(DEFAULT)
POLICY_HASH = "ecdd3be4493bb6290dc7df3fcaea8fd18c31d9159b71843a4b078cd656e185f9"
REPRESENTATION_HASH = "25929c214b8425e7892fc2ca2c38cefab3f741d9021826b6c5453c489c000735"
STAGE3A_HASH = "9808e3edf81478e0f713d996ce81755b1d4214982e2d70eeb51cf36c88bc8ed7"
EG2_MODEL_HASH = "197a32965d4b1105faf060417baa899e193fb73cd401f42ec9295234d5553d79"
SCORER_FILES = ("driveevidence_v6.py", "driveevidence_v6_rerank.py")
DEPLOYMENT_MODES = ("eg2_only", "shadow", "active")


def selected_config():
    assert identity(SELECTED_CONFIG) == POLICY_HASH
    assert identity(REPRESENTATION) == REPRESENTATION_HASH
    return copy.deepcopy(SELECTED_CONFIG)


def serving_manifest(path):
    return json.loads((Path(path) / "serving_manifest.json").read_text(encoding="utf-8"))


def load_index(path):
    """Core corruption rejects readiness. V6 incompatibility degrades to valid EG2."""
    path = Path(path)
    meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
    validate_config(meta["config"])
    model_hashes = {n: h for n, h in meta.get("model_hashes", {}).items() if "laya" not in n.casefold()}
    if not model_hashes or set(model_hashes.values()) != {EG2_MODEL_HASH}:
        raise ValueError("V6_MODEL_HASH_IDENTITY_MISMATCH")
    for name in ("global_vectors.npy", "manifest.json"):
        if digest(path / name) != meta.get("hashes", {}).get(name):
            raise ValueError("V6_CORE_SOURCE_IDENTITY_MISMATCH " + name)
    rows = json.loads((path / "manifest.json").read_text(encoding="utf-8"))["clips"]
    if (not rows or len({r["clip_id"] for r in rows}) != len(rows) or
            any(r.get("split") != "train" or len(r.get("source_sha256", "")) != 64 for r in rows)):
        raise ValueError("V6_SOURCE_IDENTITY_MISMATCH")
    matrix = np.load(path / "global_vectors.npy", mmap_mode="r", allow_pickle=False)
    if (matrix.shape != (len(rows), DIMENSION) or matrix.dtype != np.float32 or
            not np.isfinite(matrix).all() or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=.003)):
        raise ValueError("V6_INVALID_GLOBAL_CORE")
    out = {"meta": meta, "rows": rows, "global_vectors": matrix, "families": {},
           "evidence_fallback": [], "serving_manifest": {}}
    try:
        # V6 sidecar failures must never invalidate an independently valid EG2 core.
        manifest = serving_manifest(path)
        if not isinstance(manifest, dict):
            raise ValueError("V6_SERVING_MANIFEST_SCHEMA")
        out["serving_manifest"] = manifest
        if (identity(meta["config"]) != manifest["EG2_config_sha256"] or manifest["EG2_revision"] != REVISION or
                manifest.get("EG2_model_hashes") != model_hashes):
            raise ValueError("V6_MANIFEST_MODEL_IDENTITY_MISMATCH")
        for name in ("global_vectors.npy", "manifest.json"):
            if meta["hashes"][name] != manifest["artifact_hashes"][name]:
                raise ValueError("V6_MANIFEST_SOURCE_IDENTITY_MISMATCH")
        experimental = meta["experimental_policy"]
        if (manifest["schema"] != "driveevidence-v6" or manifest["version"] != 1 or
                meta.get("format") != "driveevidence-v6" or meta.get("schema") != 1):
            raise ValueError("V6_SCHEMA_MISMATCH")
        if (meta["representation"] != REPRESENTATION or meta["representation_sha256"] != REPRESENTATION_HASH or
                manifest["representation_config_sha256"] != REPRESENTATION_HASH):
            raise ValueError("V6_REPRESENTATION_HASH_MISMATCH")
        if (experimental["qualified"] is not False or experimental["config"] != selected_config() or
                experimental["config_sha256"] != POLICY_HASH or manifest["reranker_config_sha256"] != POLICY_HASH or
                identity(meta["policy"]) != meta["policy_sha256"]):
            raise ValueError("V6_SELECTED_POLICY_HASH_MISMATCH")
        if manifest["Stage3A_checkpoint_sha256"] != STAGE3A_HASH:
            raise ValueError("V6_STAGE3A_IDENTITY_MISMATCH")
        provenance = json.loads((path / "provenance.json").read_text(encoding="utf-8"))
        if provenance["source_tracks_sha256"] != manifest["source_tracks_sha256"]:
            raise ValueError("V6_SOURCE_TRACK_IDENTITY_MISMATCH")
        if manifest["vector_dimension"] != DIMENSION or manifest["dtype"] != "float32":
            raise ValueError("V6_DIMENSION_MISMATCH")
        if manifest["source_manifest_sha256"] != digest(path / "manifest.json"):
            raise ValueError("V6_SOURCE_MANIFEST_HASH_MISMATCH")
        for name, expected in manifest["scorer_hashes"].items():
            if name not in SCORER_FILES or digest(Path(__file__).parent / name) != expected:
                raise ValueError("V6_SCORER_HASH_MISMATCH")
        if set(manifest["scorer_hashes"]) != set(SCORER_FILES):
            raise ValueError("V6_SCORER_HASH_MISSING")
        for name, expected in manifest["artifact_hashes"].items():
            if Path(name).name != name or digest(path / name) != expected:
                raise ValueError("V6_EVIDENCE_HASH_MISMATCH")
        research = load_research_index(path)
        if research["evidence_fallback"]:
            raise ValueError("V6_INVALID_EVIDENCE " + str(research["evidence_fallback"]))
        counts = {"GLOBAL": len(matrix), **{k: len(v["vectors"]) for k, v in research["families"].items()}}
        if counts != manifest["family_counts"]:
            raise ValueError("V6_VECTOR_COUNTS_MISMATCH")
        out.update(research)
    except Exception as exc:
        out["evidence_fallback"] = [str(exc)]
        out["families"] = {}
    return out
