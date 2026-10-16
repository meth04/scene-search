"""Fail-closed product adapter for the existing privacy-v2.1 implementation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import cv2

from demo.product_runtime import requested_device
from hybrid_v2.embeddinggemma2 import digest, identity
from src.privacy.config import PrivacyConfig
from src.privacy.pipeline import scan_video_privacy
from src.privacy.render import render_private_clip
from src.privacy.tracks import ROI_FORMAT, load_privacy_tracks, write_json


def configuration():
    # FP32 avoids the already documented GTX1650 plate pre-NMS FP16 failure.
    # Detection thresholds, mandatory global passes and ROI refinement stay intact.
    return PrivacyConfig(
        profile="gtx1650",
        plate_device="cpu" if requested_device() == "cpu" else None,
        quality_mode="v2",
        detect_every=1,
        face_roi=True,
        plate_roi=True,
        small_object_rescue=True,
        overlap=True,
        batch_size=1,
        cpu_threads=4,
        fp16=False,
    )


def prepare(raw, folder, duration):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    cfg = configuration()
    scan_video_privacy(raw, folder / "tracks", cfg)
    load_privacy_tracks(folder / "tracks", raw)
    safe = folder / "safe.mp4"
    safe.unlink(missing_ok=True)  # Only private, unpublished work is replaceable.
    render_private_clip(raw, folder / "tracks", safe, 0, duration, mode="pixelate")
    cap = cv2.VideoCapture(str(safe))
    try:
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or not cv2.imwrite(str(folder / "poster.jpg"), frame):
        raise ValueError("PRIVACY_POSTER_FAILED")
    write_json(
        folder / "receipt.json",
        {
            "format": ROI_FORMAT,
            "source_sha256": digest(raw),
            "config_sha256": identity(asdict(cfg)),
            "complete": True,
            "human_reviewed": False,
            "start_sec": 0,
            "end_sec": duration,
            "hashes": {
                str(p.relative_to(folder)): digest(p)
                for p in folder.rglob("*")
                if p.is_file() and p.name != "receipt.json"
            },
        },
    )


def validate(raw, folder, duration):
    import json

    folder = Path(folder)
    receipt = json.loads((folder / "receipt.json").read_text(encoding="utf-8"))
    if (
        receipt["format"] != ROI_FORMAT
        or receipt["complete"] is not True
        or receipt["source_sha256"] != digest(raw)
        or receipt["end_sec"] != duration
        or receipt["config_sha256"] != identity(asdict(configuration()))
    ):
        raise ValueError("PRIVACY_RECEIPT_INVALID")
    for name, checksum in receipt["hashes"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or digest(path) != checksum:
            raise ValueError("PRIVACY_ARTIFACT_INVALID")
    load_privacy_tracks(folder / "tracks", raw)
    return receipt
