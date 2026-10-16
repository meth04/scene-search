"""Localization-only privacy scanning and cached media rendering, isolated from retrieval."""

from .config import PrivacyConfig
from .pipeline import scan_video_privacy
from .render import apply_privacy_to_frame, render_private_clip
from .tracks import Detection, PrivacyScanResult, TrackBox, load_privacy_tracks

__all__ = [
    "Detection",
    "PrivacyConfig",
    "PrivacyScanResult",
    "TrackBox",
    "apply_privacy_to_frame",
    "load_privacy_tracks",
    "render_private_clip",
    "scan_video_privacy",
]
