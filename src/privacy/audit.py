from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .render import apply_privacy_to_frame, render_private_clip
from .tracks import load_privacy_tracks, write_json


def audit_video(video, cache, output, *, samples=24):
    tracks = load_privacy_tracks(cache, video)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    risk = json.loads((Path(cache) / "risk_frames.json").read_text(encoding="utf-8"))
    count, fps = tracks.meta["covered_frames"], tracks.meta["video"]["fps"]
    normal_ids = np.linspace(0, count - 1, samples, dtype=int).tolist()
    risk_ids = sorted(
        {i for entry in risk for i in range(max(0, entry["frame_index"] - 1), min(count, entry["frame_index"] + 2))}
    )
    if len(risk_ids) > samples:
        risk_ids = [risk_ids[i] for i in np.linspace(0, len(risk_ids) - 1, samples, dtype=int)]
    needed = set(normal_ids + risk_ids)
    thumbs = {}
    cap = cv2.VideoCapture(str(video))
    try:
        for i in range(max(needed) + 1):
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Privacy audit decode failed")
            if i in needed:
                private = apply_privacy_to_frame(frame, tracks.boxes(i), "pixelate")
                tile = cv2.resize(private, (384, 216))
                cv2.putText(
                    tile, f"frame {i} / {i / fps:.3f}s", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
                thumbs[i] = tile
    finally:
        cap.release()
    paths = []
    for name, ids in (("anonymized_contact_sheet.jpg", normal_ids), ("risk_contact_sheet.jpg", risk_ids or normal_ids)):
        canvas = np.zeros((216 * ((len(ids) + 3) // 4), 384 * 4, 3), dtype=np.uint8)
        for pos, i in enumerate(ids):
            canvas[(pos // 4) * 216 : (pos // 4 + 1) * 216, (pos % 4) * 384 : (pos % 4 + 1) * 384] = thumbs[i]
        path = output / name
        if not cv2.imwrite(str(path), canvas):
            raise RuntimeError("Cannot write privacy audit sheet")
        paths.append(str(path))
    # Start the short clip near the first observed face/plate, if possible.
    first = int(tracks.tracks["frame_index"][0]) if len(tracks.tracks) else 0
    start = max(0.0, first / fps - 0.5)
    clip = render_private_clip(
        video, cache, output / "private_test.mp4", start, min(count / fps, start + 5), mode="pixelate"
    )
    summary = {
        "sheets": paths,
        "clip": str(clip.path),
        "sampled_frames": normal_ids,
        "risk_sampled_frames": risk_ids,
        "risk_frames": len(risk),
        "note": "Anonymized review aid, not proof of recall. Compare originals locally without saving raw crops.",
    }
    write_json(output / "audit.json", summary)
    return summary
