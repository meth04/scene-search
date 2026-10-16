"""Product geometry: same detector/tracker recipe on a selected CUDA device or CPU."""

from __future__ import annotations

import gc
import time
from collections import defaultdict

import numpy as np

from demo.product_runtime import execution_device
from hybrid_v2.driveevidence_v3 import CLASSES
from hybrid_v2.embeddinggemma2 import ROOT, digest, identity
from hybrid_v2.embeddinggemma2_index import write_json


def detector(rows, base):
    import sys

    import torch

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_driveevidence_v3 as frozen

    device = execution_device(torch)
    if device == "cuda" and torch.cuda.device_count() == 1 and "GTX 1650" in torch.cuda.get_device_name(0):
        return frozen.detector(rows, base)
    return _portable_detector(rows, base)


def _portable_detector(rows, base):
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import torch
    from build_driveevidence_v3 import PROTOCOL, decode, read
    from ultralytics import YOLO

    from stage3a.stage3a_motion import Obs, aggregate_window_tracks

    device = execution_device(torch)
    torch.set_num_threads(4)
    checkpoint = ROOT / "data/model-cache/yolo11n.pt"
    h = digest(checkpoint)
    model = YOLO(str(checkpoint))
    model.model.requires_grad_(False)
    for j, row in enumerate(rows):
        c = base / f"{j:06d}"
        d = decode(row, c)
        if (c / "objects.json").exists():
            o = read(c / "objects.json")
            if (
                o["source_sha256"] != row["source_sha256"]
                or o["detector_sha256"] != h
                or o["protocol_sha256"] != identity(PROTOCOL)
            ):
                raise ValueError("DENSE_TRACK_IDENTITY")
            continue
        start = time.perf_counter()
        frames = np.load(c / "frames.npy", mmap_mode="r")
        byid = defaultdict(list)
        obs = []
        if getattr(model, "predictor", None) and getattr(model.predictor, "trackers", None):
            for tracker in model.predictor.trackers:
                tracker.reset()
        with torch.inference_mode():
            for sample, pos in enumerate(d["track_positions"]):
                result = model.track(
                    np.ascontiguousarray(frames[pos, :, :, ::-1]),
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=640,
                    conf=0.25,
                    iou=0.5,
                    device=0 if device == "cuda" else "cpu",
                    half=False,
                    verbose=False,
                )[0]
                boxes = result.boxes
                if boxes is None or boxes.id is None:
                    continue
                for k in range(len(boxes)):
                    cls = result.names[int(boxes.cls[k])]
                    if cls not in CLASSES:
                        continue
                    tid = int(boxes.id[k])
                    cf = float(boxes.conf[k])
                    xywh = boxes.xywhn[k].cpu().numpy().tolist()
                    box = boxes.xyxy[k].cpu().numpy().astype(int).tolist()
                    byid[tid].append(
                        {
                            "class": cls,
                            "confidence": cf,
                            "frame_position": pos,
                            "sample": sample,
                            "timestamp": d["timestamps"][pos],
                            "bbox": box,
                            "xywh": xywh,
                        }
                    )
                    if cls in CLASSES[:6]:
                        obs.append(Obs(sample, tid, cls, cf, *xywh))
        features, mask = aggregate_window_tracks(obs, len(d["track_positions"]))
        np.save(c / "stage3a_geometry.npy", features, allow_pickle=False)
        np.save(c / "stage3a_mask.npy", mask, allow_pickle=False)
        objects = []
        for tid, observations in sorted(byid.items()):
            cls = max(CLASSES, key=lambda x: sum(o["class"] == x for o in observations))
            objects.append(
                {
                    "slot": tid,
                    "class": cls,
                    "visible_ratio": len(observations) / len(d["track_positions"]),
                    "mean_area": float(np.mean([o["xywh"][2] * o["xywh"][3] for o in observations])),
                    "duration": observations[-1]["timestamp"] - observations[0]["timestamp"],
                    "confidence": float(np.mean([o["confidence"] for o in observations])),
                    "observations": observations,
                    "physical_id_certified": False,
                }
            )
        write_json(
            c / "objects.json",
            {
                "objects": objects,
                "source_sha256": row["source_sha256"],
                "detector_sha256": h,
                "protocol_sha256": identity(PROTOCOL),
                "seconds": time.perf_counter() - start,
                "peak_VRAM_bytes": (torch.cuda.max_memory_allocated() if device == "cuda" else 0),
            },
        )
        print(
            "V3 dense track", j + 1, len(rows), "observations", sum(len(x["observations"]) for x in objects), flush=True
        )
    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
