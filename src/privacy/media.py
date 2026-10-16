"""Cache-only media gateways. Call AFTER source authorization and revalidation."""

from pathlib import Path
from uuid import uuid4

from .config import ROOT
from .render import ffmpeg_binary, render_private_clip
from .tracks import digest, load_privacy_tracks, write_json


def find_privacy_cache(video, source_checksum, index_root=None):
    video = Path(video)
    candidates = [video.with_name(video.name + ".privacy")]
    root = Path(index_root or ROOT / "outputs/local_hybrid_indexes")
    if root.is_dir():
        candidates += [p / "privacy" for p in root.iterdir() if p.is_dir()]
    for cache in candidates:
        try:
            import json

            meta = json.loads((cache / "privacy_meta.json").read_text(encoding="utf-8"))
            if meta["video"]["sha256"] == source_checksum:
                load_privacy_tracks(cache, video)
                return cache
        except (OSError, ValueError, KeyError):
            continue
    raise ValueError("PRIVACY_CACHE_REQUIRED: scan this original video before preview/export")


def private_full_media(video, cache):
    """Keep the original timeline for existing browser anchor/trim controls."""
    import json

    if not ffmpeg_binary():
        raise ValueError("Private browser media requires FFmpeg H.264 encoding; install FFmpeg and put it on PATH")

    tracks = load_privacy_tracks(cache, video)
    cache = Path(cache)
    identity = tracks.meta["tracks_sha256"]
    receipt = cache / "private_media.json"
    try:
        record = json.loads(receipt.read_text(encoding="utf-8"))
        name = record["file"]
        if Path(name).name == name and record["tracks_sha256"] == identity and digest(cache / name) == record["sha256"]:
            return cache / name
    except (OSError, KeyError, ValueError):
        pass
    target = cache / f"private_{uuid4().hex}.mp4"
    info = tracks.meta["video"]
    render_private_clip(video, cache, target, 0, info["frame_count"] / info["fps"], mode="pixelate")
    write_json(receipt, {"file": target.name, "sha256": digest(target), "tracks_sha256": identity})
    return target
