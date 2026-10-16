from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

V1_FORMAT = "scene-search-privacy-v1"
FORMAT = "scene-search-privacy-v2"
ROI_FORMAT = "scene-search-privacy-v2.1"
V1_DTYPE = np.dtype(
    [
        ("frame_index", "<i8"),
        ("timestamp_sec", "<f8"),
        ("kind", "u1"),
        ("track_id", "<i8"),
        ("box", "<f4", (4,)),
        ("confidence", "<f4"),
        ("source", "u1"),
    ]
)
STATES = ("TRUSTED", "CONFIRMED", "PROVISIONAL", "TRACKER_ONLY", "STALE", "SUSPICIOUS", "LEGACY")
DTYPE = np.dtype(V1_DTYPE.descr + [
    ("state", "u1"), ("confirmation_count", "<i4"), ("quality_score", "<f4"),
    ("source_quality", "<f4"), ("tracker_only_run", "<i4"),
])
KINDS = ("face", "plate")
SOURCES = ("detector", "tracker", "interpolated")
ORIGINS = ("legacy", "global_face", "roi_face", "global_plate", "roi_plate", "head_fallback")
ROI_DTYPE = np.dtype(DTYPE.descr + [("origin", "u1"), ("parent_roi_id", "<i8"), ("detector", "U32")])


def cache_format(config):
    if config.get("quality_mode") != "v2":
        return V1_FORMAT
    return ROI_FORMAT if any(config.get(k) for k in ("face_roi", "plate_roi", "head_fallback", "small_object_rescue")) else FORMAT


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def probe(video):
    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, frame = cap.read()
        if not ok or frame is None or not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise ValueError(f"Invalid or undecodable video: {video}")
        return {
            "path": str(Path(video).resolve()),
            "sha256": digest(video),
            "bytes": Path(video).stat().st_size,
            "fps": fps,
            "frame_count": count,
            "width": frame.shape[1],
            "height": frame.shape[0],
        }
    finally:
        cap.release()


@dataclass(frozen=True)
class Detection:
    kind: str
    box: tuple[float, float, float, float]
    confidence: float
    landmarks: tuple[tuple[float, float], ...] | None = None
    origin: str = "legacy"
    parent_roi_id: int = -1
    detector: str = ""


@dataclass(frozen=True)
class TrackBox:
    frame_index: int
    timestamp_sec: float
    kind: str
    track_id: int
    box: tuple[float, float, float, float]
    confidence: float
    source: str
    state: str = "LEGACY"
    confirmation_count: int = 0
    quality_score: float = 0.0
    source_quality: float = 0.0
    tracker_only_run: int = 0
    origin: str = "legacy"
    parent_roi_id: int = -1
    detector: str = ""


@dataclass
class PrivacyScanResult:
    cache: Path
    meta: dict
    tracks: np.ndarray


@dataclass
class PrivacyTracks:
    meta: dict
    tracks: np.ndarray
    offsets: np.ndarray

    def boxes(self, frame_index):
        if not 0 <= frame_index < self.meta["covered_frames"]:
            raise ValueError("PRIVACY_CACHE_INCOMPLETE: requested frame has not been scanned")
        rows = self.tracks[self.offsets[frame_index] : self.offsets[frame_index + 1]]
        return [
            TrackBox(
                int(r["frame_index"]),
                float(r["timestamp_sec"]),
                KINDS[r["kind"]],
                int(r["track_id"]),
                tuple(map(float, r["box"])),
                float(r["confidence"]),
                SOURCES[r["source"]],
                *([STATES[r["state"]], int(r["confirmation_count"]), float(r["quality_score"]),
                   float(r["source_quality"]), int(r["tracker_only_run"])] if "state" in self.tracks.dtype.names else []),
                *([ORIGINS[r["origin"]], int(r["parent_roi_id"]), str(r["detector"])]
                  if "origin" in self.tracks.dtype.names else []),
            )
            for r in rows
        ]


def save_tracks(cache, meta, boxes, risks):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    version = cache_format(meta.get("config", {}))
    if (cache / "privacy_meta.json").exists():
        old = json.loads((cache / "privacy_meta.json").read_text(encoding="utf-8"))
        if old.get("format") != version:
            raise ValueError("Refusing to overwrite a different privacy cache version; use a separate --cache directory")
    data = np.array(
        [
            (
                b.frame_index,
                b.timestamp_sec,
                KINDS.index(b.kind),
                b.track_id,
                b.box,
                b.confidence,
                SOURCES.index(b.source),
            ) + ((STATES.index(b.state), b.confirmation_count, b.quality_score, b.source_quality, b.tracker_only_run)
                 if version in {FORMAT, ROI_FORMAT} else ())
            + ((ORIGINS.index(b.origin), b.parent_roi_id, b.detector) if version == ROI_FORMAT else ())
            for b in boxes
        ],
        dtype=ROI_DTYPE if version == ROI_FORMAT else DTYPE if version == FORMAT else V1_DTYPE,
    )
    data.sort(order=["frame_index", "kind", "track_id"])
    offsets = np.concatenate(([0], np.cumsum(np.bincount(data["frame_index"], minlength=meta["covered_frames"]))))
    # A unique generation plus metadata committed last prevents mixed scan generations.
    filename = f"tracks_{uuid4().hex}.npz"
    with (cache / filename).open("wb") as stream:
        np.savez_compressed(stream, tracks=data, offsets=offsets)
    meta = {**meta, "format": version, "tracks_file": filename, "tracks_sha256": digest(cache / filename)}
    write_json(cache / "risk_frames.json", risks)
    write_json(cache / "privacy_meta.json", meta)
    return PrivacyScanResult(cache, meta, data)


def load_privacy_tracks(cache, video):
    cache = Path(cache)
    try:
        meta = json.loads((cache / "privacy_meta.json").read_text(encoding="utf-8"))
        source = probe(video)
        if meta.get("format") not in {FORMAT, V1_FORMAT, ROI_FORMAT} or not meta.get("complete"):
            raise ValueError("unsupported/incomplete scan")
        if any(
            meta["video"][key] != source[key] for key in ("sha256", "bytes", "frame_count", "width", "height", "fps")
        ):
            raise ValueError("source video identity mismatch")
        filename = meta["tracks_file"]
        if Path(filename).name != filename or not filename.endswith(".npz"):
            raise ValueError("invalid tracks path")
        path = cache / filename
        if digest(path) != meta["tracks_sha256"]:
            raise ValueError("tracks checksum mismatch")
        with np.load(path, allow_pickle=False) as archive:
            data, offsets = archive["tracks"], archive["offsets"]
        count = source["frame_count"]
        if (
            data.dtype != (ROI_DTYPE if meta["format"] == ROI_FORMAT else DTYPE if meta["format"] == FORMAT else V1_DTYPE)
            or data.ndim != 1
            or offsets.shape != (count + 1,)
            or offsets.dtype.kind not in "iu"
            or offsets[0] != 0
            or offsets[-1] != len(data)
            or (np.diff(offsets) < 0).any()
            or meta["covered_frames"] != count
        ):
            raise ValueError("invalid track coverage/schema")
        if len(data):
            indices = data["frame_index"]
            coords = data["box"]
            if (
                indices.min() < 0
                or indices.max() >= count
                or (np.diff(indices) < 0).any()
                or not np.array_equal(np.diff(offsets), np.bincount(indices, minlength=count))
                or (data["kind"] > 1).any()
                or (data["source"] > 2).any()
                or not np.isfinite(coords).all()
                or not np.isfinite(data["confidence"]).all()
                or (data["confidence"] < 0).any()
                or (data["confidence"] > 1).any()
                or (coords[:, 2:] <= coords[:, :2]).any()
                or (coords < 0).any()
                or (coords[:, [0, 2]] > source["width"]).any()
                or (coords[:, [1, 3]] > source["height"]).any()
                or not np.allclose(data["timestamp_sec"], indices / source["fps"], atol=1e-6)
            ):
                raise ValueError("invalid boxes/alignment")
            if meta["format"] == ROI_FORMAT and (data["origin"] >= len(ORIGINS)).any():
                raise ValueError("invalid detection origin")
            if meta["format"] in {FORMAT, ROI_FORMAT} and (
                (data["state"] >= len(STATES)).any() or (data["confirmation_count"] < 0).any()
                or (data["tracker_only_run"] < 0).any()
                or any(not np.isfinite(data[name]).all() or (data[name] < 0).any() or (data[name] > 1).any()
                       for name in ("quality_score", "source_quality"))
            ):
                raise ValueError("invalid v2 quality fields")
        return PrivacyTracks(meta, data, offsets)
    except (OSError, KeyError, TypeError, IndexError, ValueError) as exc:
        raise ValueError(f"PRIVACY_CACHE_INVALID: {cache}: {exc}. Run privacy-scan; raw output is disabled.") from exc
