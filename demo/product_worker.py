"""Product-owned transport for the unchanged, offline frozen EG2/V6 worker."""

from __future__ import annotations

import os
import subprocess
import uuid
from pathlib import Path

import numpy as np
import psutil

from hybrid_v2.driveevidence_v6_worker import DriveEvidenceV6Worker
from hybrid_v2.embeddinggemma2 import ROOT, cached_model_path, decode_frames, normalized, worker_python_path


class ProductWorker(DriveEvidenceV6Worker):
    def load(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self
            cached_model_path()
            env = {
                **os.environ,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "CUDA_VISIBLE_DEVICES": "0",
                "PYTHONUTF8": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
            }
            log = Path(os.getenv("SCENESEARCH_PRODUCT_ROOT", str(ROOT / "outputs/product"))) / "logs/worker.stderr.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stderr:
                self.process = subprocess.Popen(
                    [str(worker_python_path()), "-B", "-m", "demo.product_native_worker"],
                    cwd=ROOT,
                    env=env,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr,
                    text=True,
                    encoding="utf-8",
                    bufsize=1,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            try:
                self.metrics = self.request("load", config=self.config)
            except Exception:
                self.close()
                raise
            return self

    def encode_video(self, path, config):
        # The same five RGB endpoint frames and official video processor as
        # EmbeddingGemma2Encoder.encode_video; no new pooling or representation.
        frames, timestamps = decode_frames(path, config["frame_count"])
        root = Path(os.getenv("SCENESEARCH_PRODUCT_ROOT", str(ROOT / "outputs/product"))) / "tmp"
        root.mkdir(parents=True, exist_ok=True)
        target = root / (uuid.uuid4().hex + ".npy")
        np.save(target, frames, allow_pickle=False)
        self._temporary_frames = getattr(self, "_temporary_frames", []) + [target]
        vector, encoded = self.encode_evidence(
            target, list(range(config["frame_count"])), visual_tokens=config["vision_tokens"]
        )
        return normalized(vector), {"frame_timestamps": timestamps, **encoded}

    def encode_evidence(self, path, indices, **kwargs):
        result = super().encode_evidence(path, indices, **kwargs)
        # The frozen child mmaps its latest file. Windows permits removal after
        # a different evidence file is loaded or the child has stopped.
        remaining = []
        for target in getattr(self, "_temporary_frames", []):
            try:
                if target == Path(path):
                    remaining.append(target)
                else:
                    target.unlink(missing_ok=True)
            except PermissionError:
                remaining.append(target)
        self._temporary_frames = remaining
        return result

    def close(self):
        # Windows venv python.exe is a launcher with a real interpreter child.
        # Waiting only for the launcher does not release that child's mmap yet.
        with self.lock:
            descendants = []
            if self.process is not None:
                try:
                    descendants = psutil.Process(self.process.pid).children(recursive=True)
                except psutil.NoSuchProcess:
                    pass
            super().close()
            _, alive = psutil.wait_procs(descendants, timeout=2)
            for child in alive:
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            _, alive = psutil.wait_procs(alive, timeout=10)
            if alive:
                raise RuntimeError("PRODUCT_WORKER_STOP_FAILED")
        for target in getattr(self, "_temporary_frames", []):
            target.unlink(missing_ok=True)
        self._temporary_frames = []
