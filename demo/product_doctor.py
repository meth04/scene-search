"""Non-destructive local diagnostics. No model inference, downloads or reserve reads."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from hybrid_v2.embeddinggemma2 import ROOT, digest, worker_python_path
from src.privacy.render import ffmpeg_binary


def verify_assets():
    manifest = json.loads((ROOT / "configs/product/assets.json").read_text(encoding="utf-8"))
    assets = manifest["assets"]
    for name, checksum in {**assets, **manifest.get("code", {})}.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or digest(path) != checksum:
            raise ValueError("PRODUCT_ASSET_MISSING_OR_CHANGED:" + name)
    return assets


def doctor():
    report = {
        "checks": {},
        "ready": False,
        "remote_inference": False,
        "raw_query_logging": False,
        "default_mode": "eg2_only",
        "scientific_status": "HOLD",
    }
    try:
        report["checks"]["assets"] = {"passed": True, "count": len(verify_assets())}
    except Exception as exc:
        report["checks"]["assets"] = {"passed": False, "detail": str(exc)}
    for key, python, modules in (
        ("backend", sys.executable, ["fastapi", "uvicorn", "av", "numpy", "pydantic"]),
        (
            "frontend",
            os.getenv("SCENESEARCH_FRONTEND_PYTHON", str(ROOT / ".venv_demo/Scripts/python.exe")),
            ["streamlit", "httpx"],
        ),
        (
            "worker",
            worker_python_path(),
            ["torch", "sentence_transformers", "transformers", "ultralytics", "av", "cv2"],
        ),
    ):
        script = (
            "import importlib.metadata,json; print(json.dumps({n:importlib.metadata.version(n) for n in "
            + repr(
                [
                    "opencv-python" if n == "cv2" else "sentence-transformers" if n == "sentence_transformers" else n
                    for n in modules
                ]
            )
            + "}))"
        )
        if key == "worker":
            script += (
                "; import torch; from demo.product_runtime import execution_device; "
                "device=execution_device(torch); print(torch.cuda.get_device_name(0) if device=='cuda' else 'CPU (slower indexing)')"
            )
        try:
            result = subprocess.run(
                [str(python), "-B", "-c", script],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=60,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            report["checks"][key] = {
                "passed": result.returncode == 0,
                "detail": result.stdout.strip() if result.returncode == 0 else result.stderr[-1500:],
            }
        except Exception as exc:
            report["checks"][key] = {"passed": False, "detail": str(exc)}
    binary = ffmpeg_binary()
    report["checks"]["ffmpeg"] = {"passed": bool(binary)}
    managed_probe = ROOT / "data/bin/ffmpeg/bin/ffprobe.exe"
    probe = (
        str(managed_probe)
        if managed_probe.is_file()
        else shutil.which("ffprobe") or str(ROOT / "outputs/ffprobe-main/package/ffprobe.exe")
    )
    report["checks"]["ffprobe"] = {"passed": Path(probe).is_file()}
    if binary:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        report["checks"]["ffmpeg"].update(passed=result.returncode == 0, version=result.stdout.splitlines()[0])
    report["ready"] = all(v["passed"] for v in report["checks"].values())
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = doctor()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ready"] else 1)
