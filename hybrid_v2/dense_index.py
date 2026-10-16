"""Independent dense metadata derivatives; no visual inference or training."""
from __future__ import annotations

import base64
import gc
import json
import math
import time
from pathlib import Path

import numpy as np

from .common import chronological_frame_ids, jpeg_data_url
from .dense_metadata import (BGE_RECIPE, DENSE_FORMAT, DENSE_PROMPTS, FIELDS, METADATA_VERSION, NORMALIZATION_VERSION, PASS_KEYS,
                             PROMPT_SHA256, PROMPT_VERSION, empty_metadata, legacy_caption,
                             normalize_dense, searchable_fields, validate_record)
from .local_index import (ARTIFACTS, PHASES, array, complete_phase, read_meta,
                          reuse_visual_index, rows, validate_index, verify_checkpoints)
from .local_runtime import atomic_array, atomic_json, atomic_jsonl, sha256

METADATA_FILES = ("metadata/dense_metadata.jsonl", "metadata/metadata_meta.json")
TEXT_FILES = tuple(f"text/dense_{field}.npy" for field in FIELDS)


def prepare_dense(source, target, options=None, resume=False, *, metadata_contract="scene-search-metadata-v3"):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError("Dense source and target must be independent directories")
    if target.exists() and not resume:
        raise FileExistsError("Dense recap refuses to overwrite; choose a new output or --resume")
    validate_index(source)
    original = read_meta(source)
    options = {**{"caption_frames":10, "caption_tile_width":256, "caption_tile_height":144,
                  "caption_jpeg_quality":86, "bge_device":"cpu", "text_batch_size":8},
               **{k:v for k,v in (options or {}).items() if v is not None}}
    allowed = {"caption_frames", "caption_tile_width", "caption_tile_height", "caption_jpeg_quality",
               "qwen_base_url", "vlm_model", "qwen_runtime_description", "bge_device", "text_batch_size", "qwen_mode", "qwen_config", "qwen_cli"}
    if set(options)-allowed or options["bge_device"] != "cpu":
        raise ValueError("Dense recap only changes metadata settings and uses BGE CPU")
    if options["caption_frames"] < 2 or min(options["caption_tile_width"],options["caption_tile_height"],options["text_batch_size"]) <= 0 or not 1 <= options["caption_jpeg_quality"] <= 100:
        raise ValueError("Invalid dense sampling/embedding settings")
    config = {**original["settings"], **options, "bge_model":original["models"]["bge"]["name"],
              "bge_revision":original["models"]["bge"].get("revision")}
    from .qwen_metadata import VERSION
    if metadata_contract not in (VERSION, "legacy"):
        raise ValueError("Unsupported metadata contract")
    config["qwen_contract"] = metadata_contract
    from .qwen_metadata import PROMPT_SHA256 as v3_prompt_sha256
    identity_prompt = PROMPT_SHA256 if metadata_contract == "legacy" else v3_prompt_sha256
    identity_version = PROMPT_VERSION if metadata_contract == "legacy" else VERSION
    identity = {"source_index":str(source), "source_meta_sha256":sha256(source/"index_meta.json"),
                "source_bge_identity":original["models"]["bge"], "prompt_sha256":identity_prompt,
                "prompt_version":identity_version, "options":options}
    if target.exists():
        meta = read_meta(target)
        if meta.get("dense_recap") != identity or meta["settings"] != config:
            raise ValueError("Dense resume provenance/configuration mismatch")
        verify_reuse(source, target, validate=False)
        return target
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
    meta.update(format=DENSE_FORMAT, index_format=DENSE_FORMAT, index_version=1,
                dense_metadata_format=DENSE_FORMAT, dense_metadata_version=METADATA_VERSION,
                dense_recap=identity, settings=config)
    atomic_json(target/"index_meta.json", meta)
    return target


def receipt(index, meta, phase, files):
    meta.setdefault("dense_receipts", {})[phase] = {path:sha256(index/path) for path in files}
    atomic_json(index/"index_meta.json", meta)


def verify_receipt(index, meta, phase, files):
    saved = meta.get("dense_receipts", {}).get(phase)
    if not saved:
        return False
    if set(saved) != set(files):
        raise ValueError("Dense incomplete artifact receipt: "+phase)
    for path, digest in saved.items():
        if not (index/path).is_file() or sha256(index/path) != digest:
            raise ValueError("Dense artifact changed/incomplete: "+path)
    return True


def extract_metadata(index, meta):
    if meta["settings"].get("qwen_contract") == "scene-search-metadata-v3":
        from .product_index import extract
        return extract(index, meta, meta["settings"])
    from .local_video_index import contact_sheet
    from .vlm_client import VLMClient, require_caption_endpoint
    settings = meta["settings"]
    client = VLMClient(settings["qwen_base_url"], settings.get("vlm_model"), timeout=180, retries=2)
    client.sampling = {"repeat_penalty":1.15,"frequency_penalty":.2,"top_p":.9}
    health = require_caption_endpoint(client)
    served = next(item for item in health["data"] if item["id"] == client.model)
    count = settings["caption_frames"]
    cols = math.ceil(count/2)
    info = {"served_model":client.model, "base_url":client.base_url,
            "quantization":served.get("meta", {}).get("ftype", "unknown"),
            "runtime_description":settings.get("qwen_runtime_description") or "existing external llama-server; no configuration changes",
            "prompt_version":PROMPT_VERSION, "prompt_sha256":PROMPT_SHA256, "frame_count":count,
            "tile_size":[settings["caption_tile_width"],settings["caption_tile_height"]],
            "sheet_layout":[math.ceil(count/cols),cols],
            "sheet_size":[cols*settings["caption_tile_width"],math.ceil(count/cols)*settings["caption_tile_height"]],
            "jpeg_quality":settings["caption_jpeg_quality"]}
    info.update(extraction_passes=3,temperature=.2,max_tokens_per_pass=1400,json_mode_requested=True,
                sampling=client.sampling,transport_retries=2,json_schema_attempts=2)
    if meta.get("caption_in_progress") and meta["caption_in_progress"] != info:
        raise ValueError("Dense cached model/prompt/sampling identity changed")
    meta.update(caption_in_progress=info, caption_prompt_version=PROMPT_VERSION,
                caption_prompt_sha256=PROMPT_SHA256, caption_frame_count=count, contact_sheet_layout=info["sheet_layout"])
    atomic_json(index/"index_meta.json",meta)
    records, captions = [], []
    for window in rows(index/"manifest.jsonl"):
        cache = index/f"metadata/rows/{window['row']:06d}.json"
        ids = chronological_frame_ids(window["start_frame"],window["end_frame"],count).tolist()
        if cache.exists():
            record = json.loads(cache.read_text(encoding="utf-8"))
            validate_record(record)
            if record.get("row") != window["row"] or record.get("clip_id") != window["clip_id"] or record.get("caption_sample_frame_ids") != ids:
                raise ValueError("Dense cache alignment mismatch")
        else:
            started = time.perf_counter()
            try:
                sheet = contact_sheet(window["video_path"],window,settings["caption_tile_width"],settings["caption_tile_height"],count)
                image = jpeg_data_url(sheet,quality=settings["caption_jpeg_quality"])
                sheet_path = index/f"metadata/contact_sheets/{window['row']:06d}.jpg"
                sheet_path.parent.mkdir(parents=True,exist_ok=True)
                sheet_path.write_bytes(base64.b64decode(image.split(",",1)[1]))
                started = time.perf_counter()
                raw,pass_results,pass_warnings = {},[],[]
                for pass_id,(prompt,keys) in enumerate(zip(DENSE_PROMPTS,PASS_KEYS)):
                    pass_started = time.perf_counter()
                    last = None
                    for attempt in range(2):
                        try:
                            response = client.chat(prompt,[image],max_tokens=1400,temperature=.2)
                            raw_path = index/f"metadata/raw_responses/{window['row']:06d}_{pass_id}_{attempt}.txt"
                            raw_path.parent.mkdir(parents=True,exist_ok=True)
                            raw_path.write_text(response,encoding="utf-8")
                            value = parse_dense_response(response)
                            supplied = {k:v for k,v in value.items() if k in keys}
                            if not supplied:
                                raise ValueError("Dense response missing requested fields")
                            if pass_id == 0 and not (supplied.get("description") or supplied.get("scene_summary")):
                                raise ValueError("Dense scene pass missing literal description")
                            raw.update(supplied)
                            break
                        except Exception as exc:
                            last = f"{type(exc).__name__}: {exc}"
                    else:
                        pass_warnings.append(f"extraction pass {pass_id} failed: {last}")
                    pass_results.append({"pass":pass_id,"seconds":time.perf_counter()-pass_started,
                                         "attempts":attempt+1,"ok":not any(w.startswith(f'extraction pass {pass_id} failed') for w in pass_warnings)})
                record = normalize_dense(raw)
                record["metadata_warnings"].extend(pass_warnings)
                record["extraction_passes"] = pass_results
                atomic_json(index/f"metadata/raw_rows/{window['row']:06d}.json",raw)
                record["fallback"] = False
            except Exception as exc:
                record = empty_metadata(f"{type(exc).__name__}: {exc}")
            record.update({k:window[k] for k in ("row","clip_id","start_sec","end_sec")})
            record.update(caption_sample_frame_ids=ids, caption_request_seconds=time.perf_counter()-started)
            record["searchable_texts"] = searchable_fields(record)
            validate_record(record)
            atomic_json(cache,record)
        records.append(record)
        caption = legacy_caption(record)
        caption.update({k:record[k] for k in ("row","clip_id","start_sec","end_sec","caption_sample_frame_ids","caption_request_seconds")})
        captions.append(caption)
        print(f"[dense metadata] {len(records)}/{meta['num_windows']} fallback={record['fallback']} seconds={record['caption_request_seconds']:.2f}",flush=True)
    atomic_jsonl(index/METADATA_FILES[0],records)
    atomic_jsonl(index/"captions/captions.jsonl",captions)
    fallback = sum(r["fallback"] for r in records)
    meta["caption_stats"] = {"total":len(records),"ok":len(records)-fallback,"fallback":fallback,"fallback_rate":fallback/len(records)}
    meta["dense_stats"] = {**meta["caption_stats"], "warning_count":sum(len(r["metadata_warnings"]) for r in records),
                           "warning_rows":sum(bool(r["metadata_warnings"]) for r in records)}
    meta["models"]["qwen"] = info
    meta["dense_provenance"] = {"format":DENSE_FORMAT,"version":METADATA_VERSION,"qwen":info,
                                "bge_identity":meta["dense_recap"]["source_bge_identity"],"bge_recipe":BGE_RECIPE,
                                "stats":meta["dense_stats"]}
    atomic_json(index/METADATA_FILES[1],meta["dense_provenance"])
    complete_phase(index,meta,"caption",{"dense_metadata":True})
    receipt(index,meta,"metadata",METADATA_FILES)


def parse_dense_response(response):
    """Tolerate surrounding prose/fences, reject truncated or extra JSON roots."""
    start = response.find("{")
    if start < 0:
        raise ValueError("Dense response missing root object")
    value,end = json.JSONDecoder().raw_decode(response[start:])
    if not isinstance(value,dict) or "{" in response[start+end:] or "[" in response[start+end:]:
        raise ValueError("Dense response must contain exactly one complete root object")
    return value


def embed_metadata(index, meta):
    if meta.get("v3_native"):
        from .product_index import embed
        embed(index, meta, meta["settings"])
        complete_phase(index, meta, "text-index", {"schema_version": "scene-search-metadata-v3"})
        return
    from .embed_captions import fields
    from .text_encoder import BGEEncoder
    started = time.perf_counter()
    renormalize_metadata(index,meta)
    settings = meta["settings"]
    encoder = BGEEncoder(meta["dense_recap"]["source_bge_identity"]["path"],device="cpu",local_files_only=True)
    # Keep the source name/revision/config/tokenizer identity, even when loading by snapshot path.
    expected = meta["dense_recap"]["source_bge_identity"]
    if any(encoder.identity.get(k) != expected.get(k) for k in ("path","revision","config_sha256","tokenizer_sha256")):
        raise ValueError("Dense BGE snapshot identity changed")
    meta["models"]["bge"] = expected
    meta["dimensions"]["bge"] = int(encoder.model.config.hidden_size)
    records = rows(index/METADATA_FILES[0])
    captions = rows(index/"captions/captions.jsonl")
    for field,texts in zip(("full","action","relation","context"),zip(*[fields(r) for r in captions])):
        atomic_array(index/f"text/caption_{field}.npy",encoder.encode(texts,batch_size=settings["text_batch_size"]))
    for field in FIELDS:
        texts = [r["searchable_texts"][field+"_text"] or "none" for r in records]
        atomic_array(index/f"text/dense_{field}.npy",encoder.encode(texts,batch_size=settings["text_batch_size"]))
    # BGE's existing 128-token limit is unchanged; quantify actual truncation.
    meta["dense_text_timing"] = {"seconds":time.perf_counter()-started,"device":"cpu","arrays":12,
                                "truncated_rows_by_field":{f:sum(len(encoder.tok.encode(r["searchable_texts"][f+"_text"]))>128 for r in records) for f in FIELDS}}
    del encoder
    gc.collect()
    complete_phase(index,meta,"text-index",meta["dense_text_timing"])
    receipt(index,meta,"text",TEXT_FILES)


def renormalize_metadata(index, meta):
    """Recover schema-shaped model details from saved raw JSON, without Qwen."""
    records,captions = [],[]
    for saved in rows(index/METADATA_FILES[0]):
        raw_path = index/f"metadata/raw_rows/{saved['row']:06d}.json"
        if saved["fallback"]:
            record = saved
        else:
            if not raw_path.is_file():raise ValueError("Dense raw metadata unavailable for normalization")
            record = normalize_dense(json.loads(raw_path.read_text(encoding="utf-8")))
            record["metadata_warnings"].extend(w for w in saved["metadata_warnings"] if w.startswith("extraction pass "))
            record.update({k:saved[k] for k in ("row","clip_id","start_sec","end_sec","fallback","caption_sample_frame_ids","caption_request_seconds")})
            if "extraction_passes" in saved:record["extraction_passes"] = saved["extraction_passes"]
            record["searchable_texts"] = searchable_fields(record)
        validate_record(record)
        records.append(record)
        atomic_json(index/f"metadata/rows/{record['row']:06d}.json",record)
        caption = legacy_caption(record)
        caption.update({k:record[k] for k in ("row","clip_id","start_sec","end_sec","caption_sample_frame_ids","caption_request_seconds")})
        captions.append(caption)
    meta["dense_stats"].update(warning_count=sum(len(r["metadata_warnings"]) for r in records),warning_rows=sum(bool(r["metadata_warnings"]) for r in records))
    meta["dense_provenance"].update(normalization_version=NORMALIZATION_VERSION,stats=meta["dense_stats"])
    atomic_jsonl(index/METADATA_FILES[0],records)
    atomic_jsonl(index/"captions/captions.jsonl",captions)
    atomic_json(index/METADATA_FILES[1],meta["dense_provenance"])
    complete_phase(index,meta,"caption",{"dense_metadata":True,"normalization_version":NORMALIZATION_VERSION})
    receipt(index,meta,"metadata",METADATA_FILES)


def run_dense_phase(index, phase):
    index = Path(index).resolve()
    meta = read_meta(index)
    verify_checkpoints(meta)
    if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:
        raise ValueError("Dense source video changed")
    verify_reuse(meta["dense_recap"]["source_index"],index,validate=False)
    if phase == "metadata":
        if not verify_receipt(index,meta,"metadata",METADATA_FILES):
            extract_metadata(index,meta)
    elif phase == "text":
        if not verify_receipt(index,meta,"metadata",METADATA_FILES):
            raise ValueError("Dense metadata phase incomplete")
        if not verify_receipt(index,meta,"text",TEXT_FILES):
            embed_metadata(index,meta)
    elif phase == "validate":
        print(json.dumps(verify_reuse(meta["dense_recap"]["source_index"],index),indent=2))
    else:
        raise ValueError("Unknown dense phase")


def verify_reuse(source, target, validate=True):
    source,target = Path(source),Path(target)
    original, derivative = read_meta(source),read_meta(target)
    if derivative["dense_recap"]["source_meta_sha256"] != sha256(source/"index_meta.json"):
        raise ValueError("Dense source metadata changed")
    if original["checkpoints"] != derivative["checkpoints"]:
        raise ValueError("Dense checkpoint identity changed")
    for phase in PHASES[:5]:
        for path in ARTIFACTS[phase]:
            if sha256(source/path) != sha256(target/path):
                raise ValueError("Dense reused artifact changed: "+path)
    return validate_index(target) if validate else {"visual_reuse":"PASS"}


def validate_dense(index, meta=None, manifest=None):
    index = Path(index)
    meta = meta or read_meta(index)
    manifest = manifest or rows(index/"manifest.jsonl")
    if meta.get("v4_2_native"):
        from .v4_2_index import validate as v42_validate
        return v42_validate(index, meta, manifest)
    if meta.get("v4_1_native"):
        from .v4_1_index import validate as v41_validate
        return v41_validate(index, meta, manifest)
    if meta.get("v4_native"):
        from .v4_index import validate
        return validate(index, meta, manifest)
    if meta.get("v3_native"):
        from .product_index import validate
        return validate(index, meta, manifest)
    if (meta["format"] != DENSE_FORMAT or meta.get("index_format") != DENSE_FORMAT or meta.get("index_version") != 1
            or meta.get("dense_metadata_version") != METADATA_VERSION or meta.get("dense_metadata_format") != DENSE_FORMAT):
        raise ValueError("Dense format/version incompatible")
    for phase,files in (("metadata",METADATA_FILES),("text",TEXT_FILES)):
        if not verify_receipt(index,meta,phase,files):
            raise ValueError("Dense phase incomplete: "+phase)
    provenance = json.loads((index/METADATA_FILES[1]).read_text(encoding="utf-8"))
    if (provenance != meta.get("dense_provenance") or provenance["bge_recipe"] != BGE_RECIPE
            or provenance["bge_identity"] != meta["models"]["bge"] or provenance.get("normalization_version") != NORMALIZATION_VERSION):
        raise ValueError("Dense model/recipe provenance mismatch")
    qwen = provenance["qwen"]
    if qwen != meta["models"]["qwen"] or qwen["prompt_version"] != PROMPT_VERSION or qwen["prompt_sha256"] != PROMPT_SHA256 or not qwen["served_model"]:
        raise ValueError("Dense prompt/model identity mismatch")
    if qwen["frame_count"] != meta["caption_frame_count"] or meta["contact_sheet_layout"] != qwen["sheet_layout"]:
        raise ValueError("Dense sampling provenance mismatch")
    config = meta["settings"]
    count = config["caption_frames"]
    cols = math.ceil(count/2)
    if (qwen["frame_count"] != count or qwen["tile_size"] != [config["caption_tile_width"],config["caption_tile_height"]]
            or qwen["sheet_layout"] != [math.ceil(count/cols),cols]
            or qwen["sheet_size"] != [cols*config["caption_tile_width"],math.ceil(count/cols)*config["caption_tile_height"]]
            or qwen["jpeg_quality"] != config["caption_jpeg_quality"] or qwen.get("extraction_passes") != len(DENSE_PROMPTS)):
        raise ValueError("Dense contact sheet configuration mismatch")
    if meta["models"]["bge"] != meta["dense_recap"]["source_bge_identity"]:
        raise ValueError("Dense source BGE identity changed")
    records = rows(index/METADATA_FILES[0])
    if len(records) != len(manifest):
        raise ValueError("Dense metadata row count mismatch")
    for i,(record,window) in enumerate(zip(records,manifest)):
        validate_record(record)
        if any(record.get(k) != window[k] for k in ("row","clip_id","start_sec","end_sec")) or record["row"] != i:
            raise ValueError("Dense row alignment mismatch")
        if type(record.get("fallback")) is not bool:
            raise ValueError("Dense fallback must be bool")
        if record.get("caption_sample_frame_ids") != chronological_frame_ids(window["start_frame"],window["end_frame"],qwen["frame_count"]).tolist():
            raise ValueError("Dense sampled frame alignment mismatch")
        if not math.isfinite(record.get("caption_request_seconds",float("nan"))) or record["caption_request_seconds"] < 0:
            raise ValueError("Dense invalid request timing")
    fallback = sum(r["fallback"] for r in records)
    stats = {"total":len(records),"ok":len(records)-fallback,"fallback":fallback,"fallback_rate":fallback/len(records),
             "warning_count":sum(len(r["metadata_warnings"]) for r in records),"warning_rows":sum(bool(r["metadata_warnings"]) for r in records)}
    if meta.get("dense_stats") != stats or provenance["stats"] != stats:
        raise ValueError("Dense fallback/warning statistics mismatch")
    for path in TEXT_FILES:
        data = array(index,path)
        if data.shape != (len(manifest),meta["dimensions"]["bge"]) or data.dtype != np.float32:
            raise ValueError("Dense array row alignment/dtype mismatch: "+path)
        for start in range(0,len(data),128):
            chunk = data[start:start+128]
            if not np.isfinite(chunk).all() or not np.allclose(np.linalg.norm(chunk,axis=-1),1,atol=2e-3):
                raise ValueError("Dense array non-finite/non-normalized: "+path)
    return {"dense_metadata":"PASS","dense_warning_count":stats["warning_count"]}
