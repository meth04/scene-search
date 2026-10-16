"""Prepare an independent caption-only derivative with verified visual reuse."""
from __future__ import annotations

import hashlib
from pathlib import Path

from .caption_quality import prompt_for
from .local_index import ARTIFACTS, PHASES, read_meta, reuse_visual_index, validate_index
from .local_runtime import atomic_json, sha256


def prepare_recaption(source, target, overrides=None, resume=False, *, metadata_contract="scene-search-metadata-v3"):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Recap source and target must be independent directories")
    if target.exists() and not resume:
        raise FileExistsError("Recap refuses to overwrite an index; use a new output or --resume")
    validate_index(source)
    original = read_meta(source)
    if sha256(original["video"]["path"]) != original["video"]["sha256"]:
        raise ValueError("Source video changed; refusing recap")
    options = {"caption_mode":"temporal", "caption_frames":8, "caption_jpeg_quality":86,
               "caption_tile_width":256, "caption_tile_height":144, "bge_device":"cpu"}
    options.update({k:v for k,v in (overrides or {}).items() if v is not None})
    if any(k not in {"caption_mode", "caption_frames", "caption_jpeg_quality", "caption_tile_width",
                      "caption_tile_height", "qwen_base_url", "vlm_model", "qwen_runtime_description",
                      "bge_device", "text_batch_size", "qwen_mode", "qwen_config", "qwen_cli"} for k in options):
        raise ValueError("Recap cannot alter visual or BGE model configuration")
    prompt, version, _ = prompt_for(options["caption_mode"], options["caption_frames"])
    if not 1 <= options["caption_jpeg_quality"] <= 100:
        raise ValueError("Caption JPEG quality must be between 1 and 100")
    if min(options["caption_tile_width"], options["caption_tile_height"], options.get("text_batch_size", 8)) <= 0:
        raise ValueError("Caption dimensions and text batch size must be positive")
    config = {**original["settings"], **options}
    from .qwen_metadata import VERSION
    if metadata_contract not in (VERSION, "legacy"):
        raise ValueError("Unsupported metadata contract")
    config["qwen_contract"] = metadata_contract
    config["bge_model"] = original["models"]["bge"]["name"]
    config["bge_revision"] = original["models"]["bge"].get("revision")
    if metadata_contract != "legacy":
        from .qwen_metadata import PROMPT_SHA256 as v3_prompt_sha256
        version, prompt_digest = VERSION, v3_prompt_sha256
    else:
        prompt_digest = hashlib.sha256(prompt.encode()).hexdigest()
    identity = {"source_index":str(source), "source_meta_sha256":sha256(source/"index_meta.json"),
                "source_caption_prompt":{k:original.get(k, original["models"].get("qwen",{}).get(
                    {"caption_prompt_version":"prompt_version", "caption_prompt_sha256":"prompt_sha256"}[k]))
                    for k in ("caption_prompt_version", "caption_prompt_sha256")},
                "source_bge_identity":original["models"]["bge"],
                "prompt_version":version, "prompt_sha256":prompt_digest,
                "caption_config":options}
    if target.exists():
        meta = read_meta(target)
        if meta.get("recaption") != identity or meta["settings"] != config:
            raise ValueError("Recap resume provenance/configuration mismatch")
        for phase in PHASES[:5]:
            for relative in ARTIFACTS[phase]:
                if sha256(source/relative) != sha256(target/relative):
                    raise ValueError(f"Recap reused artifact changed: {relative}")
        return target
    # Check the endpoint before creating the derivative. Never start/stop it.
    if metadata_contract == "legacy":
        from .vlm_client import VLMClient, require_caption_endpoint
        require_caption_endpoint(VLMClient(config["qwen_base_url"], config.get("vlm_model"), timeout=5, retries=1))
    else:
        from .qwen_provider import create_provider
        provider = create_provider(config)
        try:
            provider.healthcheck()
        finally:
            provider.close()
    reuse_visual_index(source, target)
    meta = read_meta(target)
    meta["settings"] = config
    meta["recaption"] = identity
    atomic_json(target/"index_meta.json", meta)
    return target


def verify_recaption(source, target):
    validation = validate_index(target)
    original, derivative = read_meta(source), read_meta(target)
    if derivative["models"]["bge"] != original["models"]["bge"]:
        raise ValueError("Recap BGE model identity changed")
    for phase in PHASES[:5]:
        for relative in ARTIFACTS[phase]:
            if sha256(Path(source)/relative) != sha256(Path(target)/relative):
                raise ValueError(f"Recap reused artifact changed: {relative}")
    if original["checkpoints"] != derivative["checkpoints"]:
        raise ValueError("Recap checkpoint identity changed")
    return validation
