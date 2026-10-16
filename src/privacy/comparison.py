"""Versioned cache diagnostics and exact-frame anonymized engineering comparisons."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np

from .render import apply_privacy_to_frame, render_private_clip
from .tracks import SOURCES, STATES, load_privacy_tracks, write_json


def cache_metrics(cache, video):
    tracks = load_privacy_tracks(cache, video)
    data, meta = tracks.tracks, tracks.meta
    area = (data["box"][:, 2] - data["box"][:, 0]) * (data["box"][:, 3] - data["box"][:, 1])
    lifetimes = []
    for kind in range(2):
        for track_id in np.unique(data["track_id"][data["kind"] == kind]):
            frames = data["frame_index"][(data["kind"] == kind) & (data["track_id"] == track_id)]
            lifetimes.append(int(frames.max() - frames.min() + 1))
    return {
        "format": meta["format"], "config": meta.get("config"), "frames": meta["covered_frames"],
        "total_masks": len(data),
        "face_masks": int((data["kind"] == 0).sum()), "plate_masks": int((data["kind"] == 1).sum()),
        "average_mask_area_px": float(np.mean(area)) if len(area) else 0,
        "p95_mask_area_px": float(np.percentile(area, 95)) if len(area) else 0,
        "max_mask_area_px": float(np.max(area)) if len(area) else 0,
        "summed_mask_area_px": float(area.sum()),
        "tracker_only_masks": int((data["source"] == SOURCES.index("tracker")).sum()),
        "provisional_masks": int((data["state"] == STATES.index("PROVISIONAL")).sum()) if "state" in data.dtype.names else None,
        "legacy_direct_masks": int((data["track_id"] < 0).sum()) if "state" not in data.dtype.names else None,
        "state_counts": {state: int((data["state"] == index).sum()) for index, state in enumerate(STATES)} if "state" in data.dtype.names else {},
        "average_track_lifetime_frames": float(np.mean(lifetimes)) if lifetimes else 0,
        "max_track_lifetime_frames": max(lifetimes, default=0), "track_count": len(lifetimes),
        "risk_frame_count": meta["risk_frames"], "metrics": meta["metrics"],
        "note": "Counts and area are mask diagnostics, not labeled false-positive rates or privacy recall. v1 direct masks have no confirmed state.",
    }


def selected_frames(old, new, samples=12):
    count = old.meta["covered_frames"]
    ids = set(np.linspace(0, count - 1, samples, dtype=int).tolist())
    for cache in (old, new):
        for kind in (0, 1):
            rows = cache.tracks[cache.tracks["kind"] == kind]
            if len(rows):
                areas = (rows["box"][:, 2] - rows["box"][:, 0]) * (rows["box"][:, 3] - rows["box"][:, 1])
                ids.add(int(rows[np.argmax(areas)]["frame_index"]))
        tracker = cache.tracks[cache.tracks["source"] == 1]
        if len(tracker):
            ids.update(int(tracker[p]["frame_index"]) for p in np.linspace(0, len(tracker) - 1, 4, dtype=int))
    return sorted(ids)


def comparison_sheets(video, baseline, candidate, output, frame_ids=None):
    old, new = load_privacy_tracks(baseline, video), load_privacy_tracks(candidate, video)
    ids = selected_frames(old, new) if frame_ids is None else sorted(set(frame_ids))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    needed = set(ids)
    cap = cv2.VideoCapture(str(video))
    rows = {}
    try:
        for i in range(max(ids) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Comparison frame decode failed")
            if i not in needed:
                continue
            tiles = []
            for label, tracks, debug in (("baseline v1", old, False), ("candidate v2", new, False), ("v2 engineering", new, True)):
                private = apply_privacy_to_frame(frame, tracks.boxes(i), "pixelate", debug_overlay=debug)
                tile = cv2.resize(private, (512, 288))
                cv2.rectangle(tile, (0, 0), (512, 25), (0, 0, 0), -1)
                cv2.putText(tile, f"{label} | frame {i}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                tiles.append(tile)
            rows[i] = np.concatenate(tiles, axis=1)
    finally:
        cap.release()
    # One sheet of representative exact frames; additional pages preserve readable tiles.
    pages = []
    for offset in range(0, len(ids), 6):
        canvas = np.concatenate([rows[i] for i in ids[offset:offset + 6]], axis=0)
        path = output / ("before_after_contact_sheet.jpg" if offset == 0 else f"before_after_{offset // 6 + 1}.jpg")
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError("Cannot write comparison sheet")
        pages.append(str(path))
    fps, count = new.meta["video"]["fps"], new.meta["covered_frames"]
    if len(old.tracks):
        areas = (old.tracks["box"][:, 2] - old.tracks["box"][:, 0]) * (old.tracks["box"][:, 3] - old.tracks["box"][:, 1])
        start = max(0, int(old.tracks[np.argmax(areas)]["frame_index"]) / fps - 1)
    else:
        start = 0
    clip = render_private_clip(video, candidate, output / "debug_overlay_sample.mp4", start,
                               min(count / fps, start + 5), mode="pixelate", debug_overlay=True)
    result = {"video": str(video), "baseline": str(baseline), "candidate": str(candidate),
              "frame_ids": ids, "pages": pages, "debug_clip": str(clip.path)}
    write_json(output / "comparison.json", result)
    return result


def box_coverage(box, boxes):
    """Exact area covered by union of same-kind rectangular masks (no double counting)."""
    x1, y1, x2, y2 = box
    rectangles = [(max(x1, b.box[0]), max(y1, b.box[1]), min(x2, b.box[2]), min(y2, b.box[3])) for b in boxes]
    rectangles = [r for r in rectangles if r[2] > r[0] and r[3] > r[1]]
    xs = sorted({r[index] for r in rectangles for index in (0, 2)})
    area = 0.0
    for left, right in zip(xs, xs[1:]):
        spans = sorted((r[1], r[3]) for r in rectangles if r[0] <= left and r[2] >= right)
        length, end = 0.0, -float("inf")
        for bottom, top in spans:
            length += max(0, top - max(end, bottom))
            end = max(end, top)
        area += (right - left) * length
    return area / max(1e-9, (x2 - x1) * (y2 - y1))


def review_metrics(csv_path, versions, *, reviewer_type="codex_visual", coverage_threshold=.90):
    """Separate human/assistant evidence; uncertain and poster faces are excluded from FP."""
    with Path(csv_path).open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r.get("reviewer_type") == reviewer_type]
    results = []
    for row in rows:
        video, frame, kind = row["video"], int(row["frame_index"]), row["kind"]
        box = json.loads(row["bbox"])
        coverage = {name: box_coverage(box, [b for b in caches[video].boxes(frame) if b.kind == kind])
                    for name, caches in versions.items()}
        results.append({**row, "coverage": coverage})
    false = [r for r in results if r["label"] == "false_positive"]
    true = [r for r in results if r["label"] in {"true_face", "true_plate"}]
    masks = {name: sum(r["coverage"][name] >= .10 for r in false) for name in versions}
    covered = {name: sum(r["coverage"][name] >= coverage_threshold for r in true) for name in versions}
    fp_masks = {}
    for name, caches in versions.items():
        selected = set()
        for row in false:
            region = json.loads(row["bbox"])
            for box in caches[row["video"]].boxes(int(row["frame_index"])):
                if box.kind != row["kind"]:
                    continue
                # Classify a mask only if most of its area lies in a reviewed
                # clear-FP region; avoid calling a tiny intersection a false mask.
                intersection = max(0, min(region[2], box.box[2]) - max(region[0], box.box[0])) * max(
                    0, min(region[3], box.box[3]) - max(region[1], box.box[1]))
                area = (box.box[2] - box.box[0]) * (box.box[3] - box.box[1])
                if intersection / max(area, 1e-9) >= .5:
                    selected.add((row["video"], row["frame_index"], box.kind, box.track_id))
        fp_masks[name] = len(selected)
    return {"reviewer_type": reviewer_type, "reviewed_samples": len(results), "false_positive_samples": len(false),
            "true_samples": len(true), "false_positive_reviewed_regions_masked": masks,
            "false_positive_reviewed_masks": fp_masks,
            "true_regions_covered": covered, "coverage_threshold": coverage_threshold,
            "results": results, "note": "Region-level paired samples, not full-video recall. Codex annotations are not human labels."}
