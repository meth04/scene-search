"""V4.2 reuses the verified index assembler and exact BGE recipe, not old embeddings."""
from pathlib import Path

from . import metadata_v4_2 as contract, v4_1_index as shared
from .metadata_v4_2_pipeline import MultiPassStore
from .local_index import read_meta, rows
from .qwen_store import atomic_json

OPTIONS = dict(contract_module=contract, store_cls=MultiPassStore,
               store_relative="metadata/v4_2", native_flag="v4_2_native")
embed = shared.embed


def publication_gate(success, expected):
    return expected in (50, 200) and success >= (49 if expected == 50 else 198)


def finalize(index, **kwargs):
    index = Path(index)
    windows = rows(index / "manifest.jsonl")
    store = MultiPassStore(index / "metadata/v4_2")
    success = sum(store.final_success(f"{w['row']:06d}") for w in windows)
    if not publication_gate(success, len(windows)):
        atomic_json(index / "metadata/publication.json", {"status": "UNPUBLISHED", "success": success, "expected": len(windows)})
        raise RuntimeError("V4.2 publication gate failed")
    meta = read_meta(index)
    meta["v4_1_native"] = False
    atomic_json(index / "index_meta.json", meta)
    # Declared missing rows remain nonsearchable and API-unready even if >=198 succeeds.
    return shared.finalize(index, allow_partial=success < len(windows), **OPTIONS, **kwargs)


def validate(index, meta, manifest):
    result = shared.validate(index, meta, manifest, **OPTIONS)
    if meta.get("caption_prompt_sha256") != contract.COMBINED_PROMPT_SHA256:
        raise ValueError("V4.2 prompt hash mismatch")
    return result
