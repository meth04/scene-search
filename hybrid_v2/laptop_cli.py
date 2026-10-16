"""Windows-first local Hybrid CLI, with one subprocess per indexing phase."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .local_runtime import BGE, CHECKPOINTS, HF_CACHE, ROOT, SIGLIP


def run_child(command):
    # API jobs inherit redirected logs; Windows helper processes need no GUI console.
    subprocess.run(command, cwd=ROOT, check=True,
                   creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)


def positive(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def qwen_options(parser):
    parser.add_argument("--qwen-mode", choices=("local", "server"))
    parser.add_argument("--qwen-config", type=Path, help="JSON/YAML Qwen settings; CLI > environment > file > defaults")


def runtime_options(parser, defaults=False):
    default = lambda value: value if defaults else None
    parser.add_argument("--device", choices=("cuda", "cpu"), default=default("cpu"))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default=default("fp16"))
    parser.add_argument("--visual-batch-size", type=positive, default=default(1))
    parser.add_argument("--vision-chunk-size", type=positive, default=default(4))
    parser.add_argument("--yolo-device", choices=("cuda", "cpu"), default=default("cpu"))
    parser.add_argument("--bge-device", choices=("cuda", "cpu"), default=default("cpu"))
    parser.add_argument("--hf-cache-dir", type=Path, default=default(HF_CACHE))
    parser.add_argument("--local-files-only", action="store_true", default=True, help="Always offline; downloads are a separate explicit setup command")
    parser.add_argument("--siglip-model", default=None)
    parser.add_argument("--siglip-revision", default=None)
    parser.add_argument("--bge-model", default=default(BGE))
    parser.add_argument("--bge-revision", default=None)
    parser.add_argument("--detector", type=Path, default=default(ROOT/"data/model-cache/yolo11n.pt"))
    parser.add_argument("--track-fps", type=float, default=default(5.))
    parser.add_argument("--yolo-imgsz", type=positive, default=default(640))
    parser.add_argument("--text-batch-size", type=positive, default=default(8))
    qwen_options(parser)
    parser.add_argument("--qwen-base-url", default=None)
    parser.add_argument("--vlm-model", default=None)
    parser.add_argument("--qwen-runtime-description", default=None)
    parser.add_argument("--caption-tile-width", type=positive, default=default(256))
    parser.add_argument("--caption-tile-height", type=positive, default=default(144))
    parser.add_argument("--caption-frames", type=positive, default=default(6))
    parser.add_argument("--caption-mode", choices=("literal", "temporal"), default=default("literal"))
    parser.add_argument("--caption-jpeg-quality", type=positive, default=default(86))


def checkpoints_options(parser):
    for name, path in CHECKPOINTS.items():
        if name != "stage4b":
            parser.add_argument(f"--{name}-checkpoint", type=Path, default=path)


def build_parser():
    from .privacy_cli import render_options, scan_options

    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    pre = commands.add_parser("preflight", help="Read-only environment/assets check")
    pre.add_argument("--operation", choices=("index", "search", "caption"), default="index")
    pre.add_argument("--video", type=Path, default=ROOT/"data/test-video/test.mp4")
    runtime_options(pre, True)
    checkpoints_options(pre)
    index = commands.add_parser("index", help="Prepare or resume sequential inference phases")
    index.add_argument("--video", type=Path)
    index.add_argument("--output", type=Path, required=True)
    index.add_argument("--window-sec", type=float, default=10.)
    index.add_argument("--stride-sec", type=float, default=5.)
    index.add_argument("--max-windows", type=positive)
    index.add_argument("--segment-manifest", type=Path, help="Known contiguous segments; semantic windows never cross boundaries")
    index.add_argument("--from-phase", choices=("prepare", "visual", "motion", "fuse", "frame-project", "caption", "text-index", "validate"), default="prepare")
    index.add_argument("--through", choices=("prepare", "visual", "motion", "fuse", "frame-project", "caption", "text-index", "validate"), default="validate")
    runtime_options(index, True)
    checkpoints_options(index)
    index.add_argument("--privacy", action="store_true", help="Scan original media separately after indexing")
    scan_options(index)
    phase = commands.add_parser("phase", help="Run a single resumable phase")
    phase.add_argument("--index", type=Path, required=True)
    phase.add_argument("--phase", choices=("visual", "motion", "fuse", "frame-project", "caption", "text-index", "validate"), required=True)
    runtime_options(phase)
    recap = commands.add_parser("recap", help="Create a caption/text-only derivative; reuse verified visual arrays")
    recap.add_argument("--source-index", type=Path, required=True)
    recap.add_argument("--output", type=Path, required=True)
    recap.add_argument("--resume", action="store_true", help="Resume only the same verified derivative/configuration")
    qwen_options(recap)
    recap.add_argument("--caption-mode", choices=("literal", "temporal"), default="temporal")
    recap.add_argument("--caption-frames", type=positive, default=8)
    recap.add_argument("--caption-tile-width", type=positive, default=256)
    recap.add_argument("--caption-tile-height", type=positive, default=144)
    recap.add_argument("--caption-jpeg-quality", type=positive, default=86)
    recap.add_argument("--qwen-base-url")
    recap.add_argument("--vlm-model")
    recap.add_argument("--qwen-runtime-description")
    recap.add_argument("--bge-device", choices=("cpu", "cuda"), default="cpu")
    recap.add_argument("--text-batch-size", type=positive, default=8)
    dense = commands.add_parser("dense-recap", help="New dense metadata derivative with verified visual reuse")
    dense.add_argument("--source-index", type=Path, required=True)
    dense.add_argument("--output", type=Path, required=True)
    dense.add_argument("--resume", action="store_true")
    qwen_options(dense)
    dense.add_argument("--caption-frames", type=positive, default=10)
    dense.add_argument("--caption-tile-width", type=positive, default=256)
    dense.add_argument("--caption-tile-height", type=positive, default=144)
    dense.add_argument("--caption-jpeg-quality", type=positive, default=86)
    dense.add_argument("--qwen-base-url")
    dense.add_argument("--vlm-model")
    dense.add_argument("--qwen-runtime-description")
    dense.add_argument("--bge-device", choices=("cpu",), default="cpu")
    dense.add_argument("--text-batch-size", type=positive, default=8)
    dense_phase = commands.add_parser("dense-phase", help="Resume one dense inference phase")
    dense_phase.add_argument("--index", type=Path, required=True)
    dense_phase.add_argument("--phase", choices=("metadata", "text", "validate"), required=True)
    val = commands.add_parser("validate")
    val.add_argument("--index", type=Path, required=True)
    search = commands.add_parser("search")
    search.add_argument("--index", type=Path, required=True)
    search.add_argument("--query", required=True)
    for name, value in (("top-k",10), ("caption-k",100), ("stage3-k",100), ("rerank-k",50)):
        search.add_argument("--"+name, type=positive, default=value)
    search.add_argument("--intent-mode", choices=("rule", "none", "qwen"), default="rule")
    search.add_argument("--ranker", choices=("caption", "stage3", "base", "stage5", "dense", "dense-entity", "embeddinggemma2", "driveevidence", "driveevidence_v2", "driveevidence_v3", "driveevidence_v4", "driveevidence_v5", "driveevidence_v6"), default="stage5")
    search.add_argument("--candidate-k", type=positive, default=50, help="Dense-entity candidate budget before constraint reranking")
    search.add_argument("--debug-entity", action="store_true", help="Explain constraints on the single best actor")
    search.add_argument("--decision-provider", choices=("none", "laya"))
    search.add_argument("--deployment-mode", choices=("eg2_only", "shadow", "active"), help="V6 only; overrides DRIVEEVIDENCE_V6_DEPLOYMENT_MODE")
    search.add_argument("--laya-device", choices=("auto", "cuda", "cpu"))
    search.add_argument("--laya-model-path")
    search.add_argument("--laya-batch-size", type=positive)
    search.add_argument("--laya-candidate-count", type=positive)
    search.add_argument("--temporal-iou-threshold", type=float, default=.5)
    search.add_argument("--no-temporal-dedup", action="store_true", help="Keep all dense-entity windows; raw ranking is always saved")
    search.add_argument("--debug-scores", action="store_true", help="Print candidate features and scoring diagnostics")
    search.add_argument("--candidate-table", type=Path, help="Write full candidate diagnostics to a new file outside the index")
    search.add_argument("--not-found-stats", action="store_true", help="Report margins only; no calibrated rejection threshold")
    search.add_argument("--qwen-base-url", default="http://127.0.0.1:8080/v1")
    search.add_argument("--vlm-model")
    search.add_argument("--siglip-full-model", action="store_true", help="Use the full CPU foundation instead of the verified text-only loader")
    search.add_argument("--extract-top-k", type=int, default=0)
    search.add_argument("--result-dir", type=Path)
    render_options(search)
    scan = commands.add_parser("privacy-scan", help="Attach a privacy cache without rebuilding retrieval")
    scan.add_argument("--index", type=Path)
    scan.add_argument("--cache", type=Path, help="Separate versioned cache; defaults to INDEX/privacy or VIDEO.privacy")
    scan.add_argument("--video", type=Path)
    scan_options(scan)
    bench = commands.add_parser("privacy-benchmark", help="Compare detection intervals 1/2/3 on original media")
    bench.add_argument("--video", type=Path, required=True)
    bench.add_argument("--output", type=Path, default=ROOT/"outputs/privacy-benchmark")
    bench.add_argument("--max-frames", type=positive, default=300)
    scan_options(bench)
    extract = commands.add_parser("extract", help="Export an aligned cached private clip")
    extract.add_argument("--index", type=Path, required=True)
    extract.add_argument("--start", type=float, required=True)
    extract.add_argument("--end", type=float, required=True)
    extract.add_argument("--output", type=Path, required=True)
    extract.add_argument("--privacy-cache", type=Path)
    render_options(extract, debug=True)
    audit = commands.add_parser("privacy-audit", help="Anonymized contact sheets and short private clip")
    audit.add_argument("--index", type=Path, required=True)
    audit.add_argument("--output", type=Path, required=True)
    audit.add_argument("--cache", type=Path)
    return parser


def settings(args):
    excluded = {"command", "video", "output", "index", "phase", "from_phase", "through", "window_sec", "stride_sec", "max_windows", "operation", "source_index", "resume", "segment_manifest"}
    privacy_keys = {"privacy", "detect_every", "face_weights", "plate_weights", "face_score", "plate_score",
                    "face_size", "plate_imgsz", "plate_device", "face_low_conf", "face_high_conf",
                    "plate_low_conf", "plate_high_conf"}
    values = {k: str(v.absolute()) if isinstance(v, Path) else v for k, v in vars(args).items()
              if k not in excluded | privacy_keys and not k.startswith("privacy_") and not k.endswith("_checkpoint")}
    explicit = {k: values[k] for k in ("qwen_mode", "qwen_base_url", "vlm_model") if values.get(k) is not None}
    if explicit:
        values["qwen_cli"] = explicit
    return {k: v for k, v in values.items() if v is not None}


def preflight(args):
    from .local_runtime import load_checkpoint, resolve_model
    from .local_index import load_stage5
    from .vlm_client import VLMClient
    checks = []
    def add(name, fn, needed=True):
        try:
            value = fn()
            checks.append({"item": name, "status": "PASS", "detail": str(value)})
        except Exception as exc:
            checks.append({"item": name, "status": "FAIL" if needed else "WARN", "detail": str(exc)})
    def package(name):
        if importlib.util.find_spec(name) is None:
            raise RuntimeError(f"Missing {name}; setup: .\\.venv\\Scripts\\python.exe -m pip install -r requirements_laptop_hybrid.txt")
        return "installed"
    for name in ("stage2b", "stage3a", "stage5"):
        path = getattr(args, name+"_checkpoint")
        add(name+" checkpoint", lambda p=path: load_checkpoint(p).get("format"))
    add("Stage5 feature schema/calibration", lambda: len(load_stage5(args.stage5_checkpoint)["feature_names"]))
    add("Stage4B checkpoint (not consumed by Hybrid search)", lambda: load_checkpoint(CHECKPOINTS["stage4b"]).get("format"), False)
    add("SigLIP cached snapshot", lambda: resolve_model(args.siglip_model or SIGLIP, args.hf_cache_dir, args.siglip_revision, True))
    add("BGE cached snapshot", lambda: resolve_model(args.bge_model, args.hf_cache_dir, args.bge_revision, True))
    add("OpenCV", lambda: __import__("cv2").__version__)
    import torch
    add("PyTorch", lambda: f"{torch.__version__}, CUDA available={torch.cuda.is_available()}, native BF16={torch.cuda.is_bf16_supported(including_emulation=False) if torch.cuda.is_available() else False}")
    if "cuda" in (args.device, args.yolo_device, args.bge_device):
        def cuda():
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable; use CPU flags")
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            return f"{p.name}, total={total/2**30:.2f} GiB free={free/2**30:.2f} GiB; configured precision={args.precision}"
        add("GPU/VRAM", cuda)
    needed = args.operation == "index"
    add("Ultralytics", lambda: package("ultralytics"), needed)
    add("ByteTrack lap dependency", lambda: package("lap"), needed)
    def detector():
        if not args.detector.is_file():
            raise RuntimeError(f"Missing {args.detector}; run .\\.venv\\Scripts\\python.exe scripts\\prefetch_hybrid_assets.py --yolo")
        return args.detector
    add("YOLO11n weights", detector, needed)
    add("Test video", lambda: args.video if args.video.is_file() else (_ for _ in ()).throw(FileNotFoundError(args.video)), args.operation != "search")
    if args.operation != "search":
        from .qwen_provider import create_provider
        client = create_provider(cli=settings(args))
        try:
            add("Selected Qwen endpoint (vision needs separate smoke test)", client.healthcheck, args.operation == "caption")
        finally:
            client.close()
    executable = shutil.which("llama-server") or shutil.which("llama")
    checks.append({"item": "Optional llama executable", "status": "PASS" if executable else "WARN", "detail": executable or "not on PATH; external endpoint may still be usable"})
    for value in checks:
        print(f"{value['status']:4} {value['item']}: {value['detail']}")
    return 1 if any(value["status"] == "FAIL" for value in checks) else 0


def main(argv=None):
    args = build_parser().parse_args(argv)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["PYTHONUNBUFFERED"] = "1"
    sys.dont_write_bytecode = True
    import torch
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    try:
        if args.command in {"privacy-scan", "privacy-benchmark", "privacy-audit", "extract"}:
            from .privacy_cli import run
            print(json.dumps(run(args), indent=2, default=str))
            return 0
        if hasattr(args, "track_fps") and args.track_fps is not None:
            import math
            if not math.isfinite(args.track_fps) or args.track_fps <= 0:
                raise ValueError("track-fps must be finite and positive")
        if args.command == "preflight":
            return preflight(args)
        if args.command == "dense-recap":
            from .dense_index import prepare_dense, verify_reuse
            target = prepare_dense(args.source_index, args.output, settings(args), args.resume)
            for phase in ("metadata", "text", "validate"):
                cmd = [sys.executable, "-B", "-m", "hybrid_v2.laptop_cli", "dense-phase", "--index", str(target), "--phase", phase]
                print("[dense subprocess]", subprocess.list2cmdline(cmd), flush=True)
                run_child(cmd)
            print(json.dumps(verify_reuse(args.source_index,target),indent=2))
            return 0
        if args.command == "dense-phase":
            from .dense_index import run_dense_phase
            run_dense_phase(args.index,args.phase)
            return 0
        if args.command == "recap":
            from .local_video_recap import prepare_recaption, verify_recaption
            target = prepare_recaption(args.source_index, args.output, settings(args), args.resume)
            for phase in ("caption", "text-index", "validate"):
                cmd = [sys.executable, "-B", "-m", "hybrid_v2.laptop_cli", "phase", "--index", str(target), "--phase", phase]
                print("[recap subprocess]", subprocess.list2cmdline(cmd), flush=True)
                run_child(cmd)
            print(json.dumps(verify_recaption(args.source_index, target), indent=2))
            return 0
        if args.command == "index":
            from .local_index import PHASES
            from .local_video_index import prepare
            start, stop = PHASES.index(args.from_phase), PHASES.index(args.through)
            if start > stop:
                raise ValueError("from-phase must precede through")
            if start == 0:
                if args.video is None:
                    raise ValueError("--video required for prepare; use --from-phase to resume")
                prepare(args.output, args.video, settings(args),
                        {n: getattr(args, n+"_checkpoint") for n in ("stage2b", "stage3a", "stage5")},
                        args.window_sec, args.stride_sec, args.max_windows, args.segment_manifest)
            # Children consume stored settings. Only explicitly supplied runtime flags override on resume.
            supplied = list(argv if argv is not None else sys.argv[1:])
            for phase in PHASES[max(1,start):stop+1]:
                cmd = [sys.executable, "-B", "-m", "hybrid_v2.laptop_cli", "phase", "--index", str(args.output.absolute()), "--phase", phase]
                for key, value in settings(args).items():
                    option = "--"+key.replace("_", "-")
                    if option in supplied and value is not None and not isinstance(value, bool):
                        cmd += [option, str(value)]
                print("[phase subprocess]", subprocess.list2cmdline(cmd), flush=True)
                run_child(cmd)
            if args.privacy:
                from .privacy_cli import config, scan_index
                result = scan_index(args.output, args.video, config(args))
                print("Privacy cache:", result.cache, "Scan FPS:", result.meta["metrics"]["scan_fps"])
            return 0
        if args.command == "phase":
            from .local_video_index import run_phase
            run_phase(args.index, args.phase, settings(args))
        elif args.command == "validate":
            from .local_index import read_meta, validate_index
            from .embeddinggemma2 import FORMAT as EG2_FORMAT
            if read_meta(args.index).get("format") == "driveevidence-v6":
                from .driveevidence_v6_selected import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v6", "windows": len(validated["rows"]), "degraded": bool(validated["evidence_fallback"])}, indent=2))
                return 0
            if read_meta(args.index).get("format") == "driveevidence-v5":
                from .driveevidence_v5 import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v5", "windows": len(validated["rows"]), "degraded": bool(validated["evidence_fallback"])}, indent=2))
                return 0
            if read_meta(args.index).get("format") == "driveevidence-v4":
                from .driveevidence_v4 import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v4", "windows": len(validated["rows"])}, indent=2))
                return 0
            if read_meta(args.index).get("format") == "driveevidence-v3":
                from .driveevidence_v3 import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v3", "windows": len(validated["rows"])}, indent=2))
            elif read_meta(args.index).get("format") == "driveevidence-v2":
                from .driveevidence_v2 import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v2", "windows": len(validated["rows"])}, indent=2))
            elif read_meta(args.index).get("format") == "driveevidence-v1":
                from .driveevidence import load_index
                validated = load_index(args.index)
                print(json.dumps({"ready": True, "format": "driveevidence-v1",
                                  "windows": len(validated["rows"]), "fusion": validated["meta"]["fusion"]}, indent=2))
            elif read_meta(args.index).get("format") == EG2_FORMAT:
                from .embeddinggemma2_index import validate_index as validate_eg2
                validated = validate_eg2(args.index)
                print(json.dumps({"ready": validated["ready"], "format": EG2_FORMAT,
                    "windows": len(validated["rows"]), "dimension": validated["matrix"].shape[1],
                    "config": validated["meta"]["config"]}, indent=2))
            else:
                print(json.dumps(validate_index(args.index), indent=2))
        elif args.command == "search":
            from .local_video_search import search
            values = vars(args).copy()
            values.pop("command")
            search(**values)
        return 0
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
