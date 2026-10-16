"""Thin privacy CLI bridge; original Hybrid metadata is always read-only."""

from dataclasses import asdict
from pathlib import Path

from src.privacy import PrivacyConfig, render_private_clip, scan_video_privacy
from src.privacy.tracks import digest


def scan_options(parser):
    parser.add_argument("--privacy-segment-manifest", type=Path, help="Known source boundaries; reset all privacy state explicitly")
    parser.add_argument("--privacy-quality", choices=("v1", "v2"), default="v1", help="v2 is opt-in pending human recall review")
    parser.add_argument("--face-low-conf", type=float, default=0.45)
    parser.add_argument("--face-high-conf", type=float, default=0.68)
    parser.add_argument("--plate-low-conf", type=float, default=0.12)
    parser.add_argument("--plate-high-conf", type=float, default=0.40)
    parser.add_argument("--privacy-profile", choices=("auto", "cpu", "gtx1650", "rtx3090"), default="auto")
    parser.add_argument("--detect-every", type=int)
    parser.add_argument("--face-weights")
    parser.add_argument("--plate-weights")
    parser.add_argument("--face-score", type=float, default=0.45)
    parser.add_argument("--plate-score", type=float, default=0.12)
    parser.add_argument("--face-size", type=int)
    parser.add_argument("--plate-imgsz", type=int)
    parser.add_argument("--plate-device")
    parser.add_argument("--privacy-batch-size", type=int)
    parser.add_argument("--privacy-fp32", action="store_true")
    parser.add_argument("--privacy-face-roi", action="store_true")
    parser.add_argument("--privacy-plate-roi", action="store_true")
    parser.add_argument("--privacy-small-object-rescue", action="store_true", help="Conditional face/plate crops with bounded stable-plate cooldown; requires v2")
    parser.add_argument("--privacy-head-fallback", action="store_true", help="Opt-in guarded head masking for stable person ROIs; requires v2")
    parser.add_argument("--privacy-overlap", action="store_true", help="Overlap CPU faces and GPU plates with one bounded worker; requires v2 and every-frame globals")
    parser.add_argument("--privacy-object-weights")
    parser.add_argument("--privacy-object-interval", type=int, default=3)
    parser.add_argument("--privacy-max-face-rois", type=int, default=2)
    parser.add_argument("--privacy-max-plate-rois", type=int, default=2)
    parser.add_argument("--privacy-face-roi-size", type=int, default=320)
    parser.add_argument("--privacy-plate-roi-size", type=int, default=640)
    parser.add_argument("--privacy-head-fraction", type=float, default=0.40)
    parser.add_argument("--privacy-head-padding", type=float, default=0.20)
    parser.add_argument("--privacy-vehicle-padding", type=float, default=0.15)
    parser.add_argument("--privacy-roi-face-score", type=float, default=0.68)
    parser.add_argument("--privacy-roi-plate-score", type=float, default=0.40)
    parser.add_argument("--privacy-cpu-threads", type=int, default=4)


def render_options(parser, *, debug=False):
    if debug:
        parser.add_argument("--privacy-debug-overlay", action="store_true", help="Local engineering labels; normal output omits labels")
    parser.add_argument(
        "--privacy", action="store_true", default=True, help="Cached anonymization (default for media output)"
    )
    parser.add_argument("--no-privacy", action="store_false", dest="privacy", help="Explicit internal raw-media export")
    parser.add_argument("--privacy-mode", choices=("gaussian", "pixelate"), default="gaussian")
    parser.add_argument("--privacy-strength", choices=("strong",), default="strong")
    parser.add_argument("--privacy-block-size", type=int, default=16)


def config(args):
    names = (
        "detect_every",
        "face_weights",
        "plate_weights",
        "face_score",
        "plate_score",
        "face_size",
        "plate_imgsz",
        "plate_device",
        "face_low_conf", "face_high_conf", "plate_low_conf", "plate_high_conf",
    )
    values = {name: getattr(args, name) for name in names if getattr(args, name, None) is not None}
    if args.privacy_object_weights:
        values["object_weights"] = args.privacy_object_weights
    return PrivacyConfig(
        profile=args.privacy_profile, quality_mode=args.privacy_quality, batch_size=args.privacy_batch_size, fp16=not args.privacy_fp32,
        face_roi=args.privacy_face_roi, plate_roi=args.privacy_plate_roi,
        small_object_rescue=args.privacy_small_object_rescue, head_fallback=args.privacy_head_fallback,
        overlap=args.privacy_overlap, object_interval=args.privacy_object_interval,
        max_face_rois_per_frame=args.privacy_max_face_rois,
        max_plate_rois_per_frame=args.privacy_max_plate_rois,
        face_roi_size=args.privacy_face_roi_size, plate_roi_size=args.privacy_plate_roi_size,
        head_fraction=args.privacy_head_fraction, head_padding=args.privacy_head_padding,
        vehicle_padding=args.privacy_vehicle_padding, roi_face_score=args.privacy_roi_face_score,
        roi_plate_score=args.privacy_roi_plate_score, cpu_threads=args.privacy_cpu_threads, **values
    )


def video_for_index(index, video=None):
    from .local_index import read_meta

    meta = read_meta(index)
    source = Path(video or meta["video"]["path"])
    if digest(source) != meta["video"]["sha256"]:
        raise ValueError("Privacy source differs from indexed original video")
    return source


def scan_index(index, video=None, cfg=None, segment_manifest=None):
    from .local_index import read_meta
    segments = segment_manifest or read_meta(index).get("segments")
    return scan_video_privacy(video_for_index(index, video), Path(index) / "privacy", cfg, segment_manifest=segments)


def run(args):
    if args.command == "privacy-scan":
        if args.index:
            if args.cache:
                from .local_index import read_meta
                result = scan_video_privacy(video_for_index(args.index, args.video), args.cache, config(args), segment_manifest=args.privacy_segment_manifest or read_meta(args.index).get("segments"))
            else:
                result = scan_index(args.index, args.video, config(args), args.privacy_segment_manifest)
        else:
            if not args.video:
                raise ValueError("privacy-scan requires --index or --video")
            result = scan_video_privacy(
                args.video, args.cache or args.video.with_name(args.video.name + ".privacy"), config(args), segment_manifest=args.privacy_segment_manifest
            )
        return {"cache": str(result.cache), **result.meta["metrics"], "risk_frames": result.meta["risk_frames"]}
    if args.command == "extract":
        video = video_for_index(args.index)
        if args.privacy:
            return asdict(
                render_private_clip(
                    video,
                    getattr(args, "privacy_cache", None) or Path(args.index) / "privacy",
                    args.output,
                    args.start,
                    args.end,
                    mode=args.privacy_mode,
                    strength=args.privacy_strength,
                    block_size=args.privacy_block_size,
                    debug_overlay=args.privacy_debug_overlay,
                )
            )
        from .local_index import read_meta
        from .video_export import extract_clip

        extract_clip(video, args.output, args.start, args.end, read_meta(args.index)["video"]["fps"])
        return {"path": str(args.output), "privacy": False}
    if args.command == "privacy-audit":
        from src.privacy.audit import audit_video

        audit = audit_video(video_for_index(args.index), args.cache or Path(args.index) / "privacy", args.output)
        return {key: audit[key] for key in ("sheets", "clip", "risk_frames")}
    if args.command == "privacy-benchmark":
        from src.privacy.benchmark import benchmark

        report = benchmark(args.video, args.output, config(args), args.max_frames)
        return {
            "report": str(args.output / "benchmark.json"),
            "results": [
                {key: row[key] for key in ("detect_every", "scan_fps", "cached_render_fps")}
                for row in report["results"]
            ],
        }
    raise ValueError("Unknown privacy command")
