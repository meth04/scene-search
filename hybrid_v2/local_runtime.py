"""Offline model resolution and immutable-artifact helpers for laptop inference."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HF_CACHE = ROOT / "data/model-cache/huggingface"
CHECKPOINTS = {
    "stage2b": ROOT / "outputs/driveretriever_stage2b_vision/best_composite.pt",
    "stage3a": ROOT / "outputs/driveretriever_stage3a/best_composite.pt",
    "stage4b": ROOT / "outputs/driveretriever_stage4b/best_reranker.pt",
    "stage5": ROOT / "outputs/driveretriever_hybrid_v2/stage5/best_stage5.pt",
}
SIGLIP = "google/siglip2-base-patch16-224"
BGE = "BAAI/bge-base-en-v1.5"
FORMAT = "driveretriever-local-hybrid-v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_checkpoint(path, expected_format=None):
    """CPU-only, restricted loading; Stage5 stores ordinary NumPy float32 arrays."""
    with torch.serialization.safe_globals([
        np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.Float32DType,
    ]):
        ck = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        raise ValueError(f"Checkpoint root is not a dictionary: {path}")
    if expected_format and ck.get("format") != expected_format:
        raise ValueError(f"Unexpected checkpoint format at {path}: {ck.get('format')!r}")
    return ck


def resolve_model(name, cache_dir=None, revision=None, local_files_only=True):
    from huggingface_hub import snapshot_download, try_to_load_from_cache

    local = Path(name).expanduser()
    if local.is_dir():
        root = local.absolute()
    else:
        try:
            # New Hub versions can require every remote repo file when resolving
            # a whole offline snapshot. Runtime-only prefetch intentionally omits
            # ONNX/export/training files: resolve its cached config first instead.
            config = try_to_load_from_cache(str(name), "config.json", cache_dir=cache_dir, revision=revision) if local_files_only else None
            if isinstance(config, str):
                root = Path(config).parent
            else:
                root = Path(snapshot_download(
                    repo_id=str(name), cache_dir=str(cache_dir) if cache_dir else None,
                    revision=revision, local_files_only=local_files_only,
                ))
        except Exception as exc:
            command = (
                f'.\\.venv\\Scripts\\python.exe scripts\\prefetch_hybrid_assets.py '
                f'--model "{name}" --hf-cache-dir "{cache_dir or HF_CACHE}"'
            )
            raise RuntimeError(f"Model assets unavailable: {name}. Explicit setup: {command}") from exc
    if not (root / "config.json").is_file():
        raise RuntimeError(f"Model config missing: {root}")
    if not any(root.glob("*.safetensors")) and not any(root.glob("pytorch_model*.bin")):
        raise RuntimeError(f"Model weights missing/incomplete: {root}")
    return root


def model_identity(name, root):
    root = Path(root).absolute()
    revision = root.name if root.parent.name == "snapshots" else None
    return {
        "name": str(name), "path": str(root), "revision": revision,
        "config_sha256": sha256(root / "config.json"),
        "tokenizer_sha256": {p.name: sha256(p) for p in root.glob("*token*json")},
    }


def check_model_identity(identity):
    if model_identity(identity["name"], identity["path"]) != identity:
        raise ValueError(f"Foundation model/tokenizer provenance mismatch: {identity['path']}")


def device_for(name):
    if name not in {"cpu", "cuda"}:
        raise ValueError("Device must be cpu or cuda")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; rerun with --device cpu / --yolo-device cpu")
    return torch.device(name)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def atomic_array(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
    tmp.replace(path)
