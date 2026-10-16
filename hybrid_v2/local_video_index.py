"""Resumable single-video phases. Each expensive phase runs in its own process."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import time

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .build_features import FEATURE_NAMES
from .common import chronological_frame_ids, jpeg_data_url, sheet_from_frames
from .local_index import PHASES, array, complete_phase, phase_valid, read_meta, rows, validate_index, verify_checkpoints
from .local_runtime import FORMAT, ROOT, atomic_array, atomic_json, atomic_jsonl, device_for, load_checkpoint, model_identity, resolve_model, sha256


def probe(video):
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open local video: {video}")
        ok, frame = cap.read()
        fps, count = float(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid/undecodable video metadata: {video}")
        return {"fps": fps, "frame_count": count, "duration_sec": count/fps,
                "width": frame.shape[1], "height": frame.shape[0]}
    finally:
        cap.release()


def make_manifest(video, info, video_id, window=10., stride=5., max_windows=None, segment_manifest=None):
    if not all(math.isfinite(x) and x > 0 for x in (window, stride)):
        raise ValueError("Window and stride must be positive finite seconds")
    if max_windows is not None and max_windows <= 0:
        raise ValueError("max-windows must be positive")
    duration, fps = info["duration_sec"], info["fps"]
    if segment_manifest is not None:
        from .segments import load_segments
        result = []
        for segment in load_segments(segment_manifest, duration, fps):
            start, end = segment["global_start_sec"], segment["global_end_sec"]
            count = int(round((end-start)*fps))
            local = make_manifest(video, {**info, "duration_sec": end-start, "frame_count": count}, video_id, window, stride)
            for row in local:
                row.update(row=len(result), segment_id=segment["segment_id"], source_clip_id=segment["source_clip_id"],
                           local_start_sec=row["start_sec"], local_end_sec=row["end_sec"],
                           global_start_sec=start+row["start_sec"], global_end_sec=start+row["end_sec"])
                row["start_sec"], row["end_sec"] = row["global_start_sec"], row["global_end_sec"]
                row["start_frame"] += int(round(start*fps))
                row["end_frame"] += int(round(start*fps))
                row["clip_id"] = f"{video_id}:{row['start_frame']}:{row['end_frame']}"
                result.append(row)
        return result[:max_windows]
    if duration <= window:
        starts = [0.]
    else:
        starts = [i*stride for i in range(int(math.floor((duration-window)/stride))+1)]
        tail = max(0., duration-window)
        if abs(starts[-1]-tail) > 1.:
            starts.append(tail)
    # Exact server window construction including its tail policy.
    result = []
    for i, start in enumerate(starts[:max_windows]):
        end = min(duration, start+window)
        sf = max(0, int(round(start*fps)))
        ef = min(info["frame_count"], max(sf+1, int(round(end*fps))))
        result.append({"row": i, "clip_id": f"{video_id}:{sf}:{ef}", "video_id": video_id,
                       "video_path": str(Path(video).absolute()), "start_sec": start, "end_sec": end,
                       "start_frame": sf, "end_frame": ef, "fps": fps})
    return result


def prepare(index, video, settings, checkpoints, window=10., stride=5., max_windows=None, segment_manifest=None):
    index, video = Path(index).absolute(), Path(video).absolute()
    if (index / "index_meta.json").exists():
        meta = read_meta(index)
        if meta.get("segment_manifest_sha256") != (sha256(segment_manifest) if segment_manifest else None):
            raise ValueError("Segment manifest changed; choose a new output")
        if (meta["video"]["path"] != str(video) or meta["window_sec"] != window
                or meta["stride_sec"] != stride or meta.get("max_windows") != max_windows):
            raise ValueError("Existing index belongs to a different video/window configuration; choose a new output")
        if sha256(video) != meta["video"]["sha256"]:
            raise ValueError("Source video changed; choose a new output")
        if any(str(Path(p).absolute()) != meta["checkpoints"][n]["path"] for n, p in checkpoints.items()):
            raise ValueError("Checkpoint paths differ from existing index; use phase/resume or a new output")
        verify_checkpoints(meta)
        if not phase_valid(index, meta, "prepare"):
            raise ValueError("Existing index has an incomplete prepare receipt; choose a new output")
        print("[prepare] verified existing manifest; resume")
        return
    if index.exists() and any(index.iterdir()):
        raise ValueError("Refusing to overwrite a nonempty directory without valid index metadata")
    info = probe(video)
    ck = load_checkpoint(checkpoints["stage2b"], "driveretriever-stage2b-vision-v1")
    from .local_index import load_stage5
    load_stage5(checkpoints["stage5"])
    geom = (int(ck["cache_meta"]["num_frames"]), int(ck["cache_meta"]["hidden_dim"]),
            int(ck["head_state_dict"]["clip_proj.1.weight"].shape[0]))
    if geom != (12, 768, 512):
        raise ValueError(f"Unsupported Stage2B geometry: {geom}")
    identifier = re.sub(r"[^A-Za-z0-9_-]", "_", video.stem) + "_" + sha256(video)[:10]
    manifest = make_manifest(video, info, identifier, window, stride, max_windows, segment_manifest)
    from .qwen_metadata import VERSION
    settings = {"qwen_contract": VERSION, **settings}
    meta = {
        "format": FORMAT, "created_at": datetime.now(timezone.utc).isoformat(), "video_id": identifier,
        "video": {**info, "path": str(video), "bytes": video.stat().st_size, "sha256": sha256(video)},
        "window_sec": window, "stride_sec": stride, "num_windows": len(manifest), "max_windows": max_windows,
        "image_size": int(ck["cache_meta"]["image_size"]),
        "dimensions": {"frames": geom[0], "hidden": geom[1], "embedding": geom[2], "tracks": 24, "motion": 29},
        "feature_names": FEATURE_NAMES,
        "checkpoints": {n: {"path": str(Path(p).absolute()), "sha256": sha256(p)} for n, p in checkpoints.items()},
        "settings": settings, "models": {"siglip_name": ck["cache_meta"]["model_name"]}, "phases": {},
        "preprocessing": "stage2a-opencv-resize-linear-rgb-minus1-to1-endpoint-sampling-v1",
    }
    atomic_jsonl(index / "manifest.jsonl", manifest)
    if segment_manifest is not None:
        from .segments import load_segments, validation_artifact
        segments = load_segments(segment_manifest, info["duration_sec"], info["fps"])
        meta.update(segment_manifest=str(Path(segment_manifest).absolute()), segment_manifest_sha256=sha256(segment_manifest), segments=segments)
        atomic_json(index / "segment_boundaries.json", validation_artifact(manifest, segments))
    complete_phase(index, meta, "prepare")
    print(f"[prepare] windows={len(manifest)} output={index}")


def build_head(ck):
    from stage2a.stage2a_common import Stage2AHead
    state, info = ck["head_state_dict"], ck["cache_meta"]
    layers = {int(k.split(".")[2]) for k in state if k.startswith("temporal.layers.")}
    head = Stage2AHead(int(info["hidden_dim"]), int(info["num_frames"]),
                      int(state["clip_proj.1.weight"].shape[0]), len(layers), 8, .1)
    head.load_state_dict(state, strict=True)
    return head.eval()


def amp(device, precision):
    return torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" and precision == "fp16" else nullcontext()


def pooled(value):
    if torch.is_tensor(value):
        return value
    output = getattr(value, "pooler_output", None)
    if output is None:
        raise TypeError(f"Unexpected backbone feature result: {type(value)}")
    return output


def guard_gpu_service(settings):
    """An externally running GPU VLM also occupies memory outside this process."""
    from .qwen_provider import QwenConfig
    config = QwenConfig.resolve(config=settings)
    from urllib.parse import urlsplit
    if config.mode == "server" or urlsplit(config.base_url).hostname not in ("localhost", "127.0.0.1", "::1"):
        return
    from .vlm_client import VLMClient, VLMError
    try:
        VLMClient(config.base_url, timeout=2, retries=1).health()
    except VLMError:
        return
    raise RuntimeError("Qwen endpoint is running. Stop its process before CUDA visual/motion phases, "
                       "or use --device cpu --yolo-device cpu. Start Qwen after frame-project.")


@torch.inference_mode()
def visual(index, meta, settings):
    from transformers import AutoModel
    from stage2a.stage2a_common import decode_frames
    device = device_for(settings["device"])
    if device.type == "cuda":
        guard_gpu_service(settings)
    ck = load_checkpoint(meta["checkpoints"]["stage2b"]["path"], "driveretriever-stage2b-vision-v1")
    model_name = ck["cache_meta"]["model_name"]
    root = resolve_model(settings.get("siglip_model") or model_name, settings["hf_cache_dir"],
                         settings.get("siglip_revision"), True)
    dtype = torch.float16 if device.type == "cuda" and settings["precision"] == "fp16" else torch.float32
    backbone = AutoModel.from_pretrained(root, local_files_only=True, torch_dtype=dtype).to(device).eval()
    backbone.requires_grad_(False)
    meta["models"]["siglip"] = model_identity(settings.get("siglip_model") or model_name, root)
    windows = rows(index / "manifest.jsonl")
    n, t, h = len(windows), meta["dimensions"]["frames"], meta["dimensions"]["hidden"]
    rawpath = index / "visual/raw_frame_features.npy"
    basepath = index / "visual/base_clip_embeddings.npy"
    rawpath.parent.mkdir(parents=True, exist_ok=True)
    rawtmp, basetmp = rawpath.with_suffix(".npy.tmp"), basepath.with_suffix(".npy.tmp")
    raw = np.lib.format.open_memmap(rawtmp, mode="w+", dtype=np.float32, shape=(n, t, h))
    base = np.lib.format.open_memmap(basetmp, mode="w+", dtype=np.float32, shape=(n, 512))
    def batches():
        for s in range(0, n, settings["visual_batch_size"]):
            items = windows[s:s+settings["visual_batch_size"]]
            frames = torch.stack([decode_frames(Path(w["video_path"]), w["start_frame"], w["end_frame"], t, meta["image_size"]) for w in items])
            flat = frames.reshape(-1, 3, meta["image_size"], meta["image_size"]).to(device, dtype=dtype)
            features = []
            with amp(device, settings["precision"]):
                for j in range(0, len(flat), settings["vision_chunk_size"]):
                    features.append(pooled(backbone.get_image_features(pixel_values=flat[j:j+settings["vision_chunk_size"]])).float())
            yield s, torch.cat(features).reshape(len(items), t, h)
    # Unadapted foundation frames must be saved BEFORE applying Stage2B vision state.
    for s, ff in batches():
        raw[s:s+len(ff)] = ff.cpu().numpy()
        print(f"[raw frames] {s+len(ff)}/{n}", flush=True)
    params = dict(backbone.named_parameters())
    for key, value in ck["backbone_trainable_state_dict"].items():
        if key not in params or params[key].shape != value.shape:
            raise ValueError(f"Stage2B backbone key/shape mismatch: {key}")
        params[key].copy_(value.to(device=device, dtype=params[key].dtype))
    head = build_head(ck).to(device)
    for s, ff in batches():
        with amp(device, settings["precision"]):
            out = head.encode_video(ff)["clip_emb"].float()
        base[s:s+len(ff)] = out.cpu().numpy()
        print(f"[adapted base] {s+len(ff)}/{n}", flush=True)
    raw.flush()
    base.flush()
    del raw, base, head, params
    backbone = None
    gc.collect()
    rawtmp.replace(rawpath)
    basetmp.replace(basepath)


def motion(index, meta, settings):
    from stage3a.stage3a_motion import Obs, aggregate_window_tracks
    if importlib.util.find_spec("ultralytics") is None or importlib.util.find_spec("lap") is None:
        raise RuntimeError("Motion requires ultralytics and lap. Run: .\\.venv\\Scripts\\python.exe "
                           "-m pip install -r requirements_laptop_hybrid.txt")
    detector = Path(settings["detector"])
    if not detector.is_file():
        raise RuntimeError(f"YOLO weights missing: {detector}. Run: .\\.venv\\Scripts\\python.exe "
                           "scripts\\prefetch_hybrid_assets.py --yolo")
    if settings["yolo_device"] == "cuda":
        device_for("cuda")
        guard_gpu_service(settings)
    # Recent Ultralytics appends its own subdirectory and otherwise falls back to
    # /tmp. Create the Windows workspace parent before import, including in a sandbox.
    config_dir = Path(os.environ.get("YOLO_CONFIG_DIR", ROOT / "data/model-cache/ultralytics-config"))
    (config_dir / "Ultralytics").mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(config_dir.absolute())
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_OFFLINE"] = "true"
    from ultralytics import YOLO
    yolo = YOLO(str(detector))
    windows = rows(index / "manifest.jsonl")
    fps = meta["video"]["fps"]
    stride = max(1, int(round(fps / settings["track_fps"])))
    observations = [[] for _ in windows]
    end = max(w["end_sec"] for w in windows)
    def isolated_stream():
        sources = meta.get("segments") or [{"source_path": meta["video"]["path"], "global_start_sec": 0., "global_end_sec": end}]
        for segment in sources:
            source = segment.get("source_path")
            for tracker in getattr(getattr(yolo, "predictor", None), "trackers", []):
                tracker.reset()
            # Manifests may supply only timeline ranges, or refer to trimmed/time-scaled
            # originals. Decode exactly that segment from the concatenated source then.
            if not source or segment.get("source_start_sec", 0) != 0 or segment.get("temporal_scale", 1) != 1:
                capture = cv2.VideoCapture(meta["video"]["path"])
                first = int(round(segment["global_start_sec"]*fps))
                stop = int(round(segment["global_end_sec"]*fps))
                try:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, first)
                    for frame_index in range(first, stop):
                        ok, image = capture.read()
                        if not ok:
                            raise ValueError("Cannot decode segment motion frame")
                        if (frame_index-first) % stride:
                            continue
                        predictions = yolo.track(source=image, persist=True, tracker="bytetrack.yaml",
                            imgsz=settings["yolo_imgsz"], conf=.25, iou=.5,
                            device="0" if settings["yolo_device"] == "cuda" else "cpu", verbose=False)
                        yield frame_index/fps, predictions[0]
                finally:
                    capture.release()
                continue
            if not Path(source).is_file():
                raise ValueError("Segment source_path does not exist")
            child = yolo.track(source=source, stream=True, persist=True, tracker="bytetrack.yaml",
                               vid_stride=stride, imgsz=settings["yolo_imgsz"], conf=.25, iou=.5,
                               device="0" if settings["yolo_device"] == "cuda" else "cpu", verbose=False)
            try:
                for j, result in enumerate(child):
                    sec = segment["global_start_sec"] + j * stride / fps
                    if sec >= segment["global_end_sec"]:
                        break
                    yield sec, result
            finally:
                child.close()
    stream = isolated_stream()
    try:
        for sec, result in stream:
            if sec >= end:
                break
            boxes = getattr(result, "boxes", None)
            if boxes is None or boxes.id is None:
                continue
            for tid, cid, conf, box in zip(boxes.id.cpu().numpy(), boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy(), boxes.xywhn.cpu().numpy()):
                name = result.names[int(cid)].lower().strip()
                name = {"motorbike": "motorcycle", "bike": "bicycle"}.get(name, name)
                for i, w in enumerate(windows):
                    if w["start_sec"] <= sec < w["end_sec"]:
                        frame_idx = int(round((sec-w["start_sec"])*fps/stride))
                        observations[i].append(Obs(frame_idx, int(tid), name, float(conf), *map(float, box)))
    finally:
        stream.close()
    features = np.zeros((len(windows), 24, 29), np.float32)
    masks = np.zeros((len(windows), 24), np.bool_)
    for i, w in enumerate(windows):
        count = max(1, int(math.ceil((w["end_sec"]-w["start_sec"])*fps/stride)))
        features[i], masks[i] = aggregate_window_tracks(observations[i], count, max_tracks=24, min_track_obs=2)
    atomic_array(index / "visual/tracks.npy", features)
    atomic_array(index / "visual/track_masks.npy", masks)
    meta["motion"] = {"detector_path": str(detector.absolute()), "detector_sha256": sha256(detector),
                      "tracker": "bytetrack.yaml", "track_fps": settings["track_fps"],
                      "imgsz": settings["yolo_imgsz"], "track_coverage": float(masks.any(1).mean())}


def fuse(index, meta, settings):
    from stage3a.stage3a_infer import export_fused
    export_fused(array(index, "visual/base_clip_embeddings.npy"), array(index, "visual/tracks.npy"),
                 array(index, "visual/track_masks.npy"), meta["checkpoints"]["stage3a"]["path"],
                 index / "visual/fused_embeddings.npy", device="cpu", batch_size=64)


@torch.inference_mode()
def frame_project(index, meta, settings):
    from .query_encoder import make_projection
    ck = load_checkpoint(meta["checkpoints"]["stage2b"]["path"], "driveretriever-stage2b-vision-v1")
    projection = make_projection(ck["head_state_dict"], "frame_proj")
    raw = array(index, "visual/raw_frame_features.npy")
    path = index / "visual/projected_frame_embeddings.npy"
    tmp = path.with_suffix(".npy.tmp")
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=(*raw.shape[:2], 512))
    for s in range(0, len(raw), 64):
        x = torch.tensor(np.asarray(raw[s:s+64]), dtype=torch.float32)
        out[s:s+len(x)] = F.normalize(projection(x), dim=-1).numpy()
    out.flush()
    del out
    tmp.replace(path)


def contact_sheet(video, window, tile_w=256, tile_h=144, n=6):
    # Historical six-frame 2x3 layout; temporal mode uses eight frames in 2x4.
    ids = chronological_frame_ids(window["start_frame"], window["end_frame"], n)
    capture = cv2.VideoCapture(str(video))
    frames = []
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video: {video}")
        for idx in ids:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, image = capture.read()
            if not ok or image is None:
                raise ValueError(f"Cannot decode frame {idx}")
            frames.append(image)
    finally:
        capture.release()
    return sheet_from_frames(frames, tile_w, tile_h)


def fallback_caption(row, reason):
    return {"row": row["row"], "clip_id": row["clip_id"], "description": "Dashcam driving scene.",
            "objects": [], "ego_action": "", "other_actions": [], "relations": [], "road_context": [],
            "traffic_control": [], "weather": "", "events": [], "queries": [], "fallback": True,
            "fallback_reason": str(reason), "start_sec": row["start_sec"], "end_sec": row["end_sec"]}


def caption(index, meta, settings):
    if settings.get("qwen_contract") == "scene-search-metadata-v3":
        from .product_index import extract
        return extract(index, meta, settings)
    from .caption_quality import checked_caption, prompt_for
    from .vlm_client import VLMClient, require_caption_endpoint
    client = VLMClient(settings["qwen_base_url"], settings.get("vlm_model"),
                       timeout=120 if settings.get("caption_mode") == "temporal" else 60, retries=2)
    require_caption_endpoint(client)  # Fail before expensive work if none is served.
    frame_count = settings.get("caption_frames", 6)
    mode = settings.get("caption_mode", "literal")
    prompt, version, validation_version = prompt_for(mode, frame_count)
    quality = settings.get("caption_jpeg_quality", 86)
    if not isinstance(quality, int) or not 1 <= quality <= 100:
        raise ValueError("Caption JPEG quality must be between 1 and 100")
    columns = 3 if frame_count == 6 else 4 if frame_count == 8 else math.ceil(frame_count/2)
    caption_info = {"served_model": client.model, "base_url": client.base_url,
                    "runtime_description": settings.get("qwen_runtime_description") or "external HTTP service; build/quantization unverified",
                    "prompt_version": version, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
                    "caption_validation_version": validation_version,
                    "tile_size": [settings["caption_tile_width"], settings["caption_tile_height"]]}
    if mode == "temporal" or frame_count != 6 or quality != 86:
        caption_info.update(frame_count=frame_count, sheet_layout=[math.ceil(frame_count/columns), columns],
                            sheet_size=[columns*settings["caption_tile_width"], math.ceil(frame_count/columns)*settings["caption_tile_height"]],
                            jpeg_quality=quality)
    cached_info = meta.get("caption_in_progress")
    if cached_info and cached_info != caption_info:
        raise ValueError("Caption model/prompt changed during resume; choose a new output")
    meta["caption_in_progress"] = caption_info
    meta["caption_prompt_version"] = caption_info["prompt_version"]
    meta["caption_prompt_sha256"] = caption_info["prompt_sha256"]
    meta["caption_frame_count"] = frame_count
    atomic_json(index / "index_meta.json", meta)
    final = []
    for row in rows(index / "manifest.jsonl"):
        cache = index / f"captions/rows/{row['row']:06d}.json"
        if cache.exists():
            value = json.loads(cache.read_text(encoding="utf-8"))
            if value.get("row") != row["row"] or value.get("clip_id") != row["clip_id"]:
                raise ValueError(f"Cached caption alignment mismatch: {cache}")
        else:
            started = None
            try:
                sheet = contact_sheet(Path(row["video_path"]), row, settings["caption_tile_width"], settings["caption_tile_height"], n=frame_count)
                image_url = jpeg_data_url(sheet, quality=quality)
                # Persist the exact JPEG seen by Qwen for independent review.
                import base64
                sheet_path = index / f"captions/contact_sheets/{row['row']:06d}.jpg"
                sheet_path.parent.mkdir(parents=True, exist_ok=True)
                sheet_path.write_bytes(base64.b64decode(image_url.split(",", 1)[1]))
                started = time.perf_counter()
                raw_value = client.chat_json(prompt, [image_url], max_tokens=900, temperature=.05)
                atomic_json(index / f"captions/raw_rows/{row['row']:06d}.json", raw_value)
                value = checked_caption(raw_value, temporal=mode == "temporal")
                value.update(row=row["row"], clip_id=row["clip_id"], start_sec=row["start_sec"], end_sec=row["end_sec"], fallback=False)
                value["caption_sample_frame_ids"] = chronological_frame_ids(row["start_frame"], row["end_frame"], frame_count).tolist()
            except Exception as exc:
                value = fallback_caption(row, f"{type(exc).__name__}: {exc}")
            if started is not None:
                value["caption_request_seconds"] = time.perf_counter() - started
            atomic_json(cache, value)
        final.append(value)
        print(f"[caption] {len(final)}/{meta['num_windows']} fallback={value.get('fallback', False)}", flush=True)
    atomic_jsonl(index / "captions/captions.jsonl", final)
    fallback = sum(bool(r.get("fallback")) for r in final)
    meta["caption_stats"] = {"total": len(final), "ok": len(final)-fallback,
                             "fallback": fallback, "fallback_rate": fallback/len(final)}
    meta["models"]["qwen"] = caption_info
    request_times = [r["caption_request_seconds"] for r in final if "caption_request_seconds" in r]
    meta["caption_timing"] = {"timed_rows": len(request_times),
                              "average_request_seconds": sum(request_times)/len(request_times) if request_times else None,
                              "max_tokens": 900, "temperature": .05,
                              "includes_transport_and_json_retries": True}
    print(json.dumps(meta["caption_stats"], indent=2))


def text_index(index, meta, settings):
    if meta.get("v3_native"):
        from .product_index import embed
        return embed(index, meta, settings)
    from .embed_captions import fields
    from .text_encoder import BGEEncoder
    if settings["bge_device"] == "cuda":
        guard_gpu_service(settings)
    encoder = BGEEncoder(settings["bge_model"], device=settings["bge_device"], cache_dir=settings["hf_cache_dir"],
                         revision=settings.get("bge_revision"), local_files_only=True)
    captions = rows(index / "captions/captions.jsonl")
    meta["models"]["bge"] = encoder.identity
    meta["dimensions"]["bge"] = int(encoder.model.config.hidden_size)
    for name, texts in zip(("full", "action", "relation", "context"), zip(*[fields(r) for r in captions])):
        value = encoder.encode(texts, batch_size=settings["text_batch_size"])
        atomic_array(index / f"text/caption_{name}.npy", value)


def run_phase(index, phase, overrides=None):
    index = Path(index).absolute()
    meta = read_meta(index)
    from .dense_metadata import DENSE_FORMAT
    if meta["format"] == DENSE_FORMAT and not meta.get("v3_native") and phase != "validate":
        raise ValueError("Dense derivatives only support dense-phase metadata/text or validate; visual reuse is immutable")
    verify_checkpoints(meta)
    if sha256(meta["video"]["path"]) != meta["video"]["sha256"]:
        raise ValueError("Source video provenance mismatch")
    if phase not in PHASES[1:]:
        raise ValueError(f"Invalid indexing phase: {phase}")
    for previous in PHASES[:PHASES.index(phase)]:
        if not phase_valid(index, meta, previous):
            raise ValueError(f"Missing prerequisite phase: {previous}")
    if phase != "validate" and phase_valid(index, meta, phase):
        print(f"[{phase}] verified complete; skipped")
        return
    settings = {**meta["settings"], **{k: v for k, v in (overrides or {}).items() if v is not None}}
    if settings["precision"] not in ("fp16", "fp32"):
        raise ValueError("Only fp16/fp32 supported; BF16 is not selected on GTX1650")
    if phase == "validate":
        result = validate_index(index)
        complete_phase(index, meta, phase, result)
        print(json.dumps(result, indent=2))
        return
    functions = {"visual": visual, "motion": motion, "fuse": fuse, "frame-project": frame_project,
                 "caption": caption, "text-index": text_index}
    try:
        functions[phase](index, meta, settings)
    except torch.OutOfMemoryError as exc:
        raise RuntimeError(f"CUDA OOM in {phase}. Rerun this phase with --device cpu --yolo-device cpu "
                           "--bge-device cpu, or reduce --visual-batch-size/--vision-chunk-size; "
                           "stop Qwen during GPU visual/motion work.") from exc
    complete_phase(index, meta, phase, {k: settings.get(k, "cpu") for k in ("device", "precision", "yolo_device", "bge_device")})
    print(f"[{phase}] PASS", flush=True)
