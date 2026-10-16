"""V6-owned persistent offline EG2 process; chronological crops encoded as video."""
from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys

import numpy as np

from .embeddinggemma2 import ROOT, EmbeddingGemma2Encoder, EmbeddingGemma2Worker, cached_model_path, worker_python_path


class DriveEvidenceV6Worker(EmbeddingGemma2Worker):
    def load(self):
        with self.lock:
            if self.process is not None and self.process.poll() is None:
                return self
            cached_model_path()
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "0",
                   "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1"}
            log = ROOT / "outputs/driveevidence_v6/logs/worker.stderr.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("a", encoding="utf-8") as stderr:
                self.process = subprocess.Popen([str(worker_python_path()), "-B", "-m", "hybrid_v2.driveevidence_v6_worker"],
                    cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr,
                    text=True, encoding="utf-8", bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            try:
                self.metrics = self.request("load", config=self.config)
            except Exception:
                self.close()
                raise
            return self

    def encode_evidence(self, path, indices, *, boxes=None, visual_tokens=140):
        result = self.request("evidence", path=str(path), indices=indices, boxes=boxes, visual_tokens=visual_tokens)
        return np.asarray(result["embedding"], np.float32), result["metrics"]


def main():
    def blocked(*args, **kwargs):
        raise RuntimeError("V6_REMOTE_INFERENCE_PROHIBITED")
    socket.socket.connect = blocked
    socket.create_connection = blocked
    encoder, cached_path, frames = None, None, None
    for line in sys.stdin:
        request = json.loads(line)
        response = {"id": request["id"]}
        try:
            with contextlib.redirect_stdout(sys.stderr):
                operation = request["operation"]
                if operation == "load":
                    import torch
                    if torch.cuda.device_count() != 1 or "GTX 1650" not in torch.cuda.get_device_name(0):
                        raise RuntimeError("GTX1650_ONLY")
                    encoder = EmbeddingGemma2Encoder(request["config"]).load()
                    encoder.model.requires_grad_(False)
                    result = {**encoder.metrics, "training": False, "gradients": False}
                elif operation == "text":
                    vector, metrics = encoder.encode_text(request["query"])
                    result = {"embedding": vector.tolist(), "metrics": metrics}
                elif operation == "state":
                    result = {"training": bool(encoder.model.training),
                              "any_parameter_requires_grad": any(p.requires_grad for p in encoder.model.parameters()),
                              "any_gradient_tensor": any(p.grad is not None for p in encoder.model.parameters()),
                              "model_load_count": encoder.load_count, "revision": encoder.config["revision"],
                              "dtype": encoder.config["dtype"], "GPU": encoder.metrics["GPU"]}
                elif operation == "evidence":
                    from PIL import Image
                    indices = request["indices"]
                    if not indices or len(indices) > 16 or any(b <= a for a, b in zip(indices, indices[1:])):
                        raise ValueError("V6_FRAME_ORDER_OR_BOUND")
                    tokens = request["visual_tokens"]
                    if tokens not in (70, 140, 280):
                        raise ValueError("V6_TOKEN_BUDGET")
                    if cached_path != request["path"]:
                        frames = np.load(request["path"], mmap_mode="r", allow_pickle=False)
                        cached_path = request["path"]
                    selected = [frames[i] for i in indices]
                    boxes = request.get("boxes")
                    if boxes is not None:
                        if len(boxes) != len(indices):
                            raise ValueError("V6_TUBE_BOX_COUNT")
                        crops = []
                        for frame, (x1, y1, x2, y2) in zip(selected, boxes):
                            h, w = frame.shape[:2]
                            if not 0 <= x1 < x2 <= w or not 0 <= y1 < y2 <= h:
                                raise ValueError("V6_CONTEXT_OUT_OF_BOUNDS")
                            crops.append(frame[y1:y2, x1:x2])
                        # Same canvas preserves a chronological video sequence with variable observed boxes.
                        size = (max(c.shape[1] for c in crops), max(c.shape[0] for c in crops))
                        selected = [np.asarray(Image.fromarray(c).resize(size, Image.Resampling.BILINEAR)) for c in crops]
                    array = np.ascontiguousarray(np.stack(selected))
                    # One-frame semantic units use the official image path; windows/tubes the video path.
                    item = {"image": Image.fromarray(array[0])} if len(indices) == 1 else {"video": array}
                    processing = {"images": {"max_soft_tokens": tokens}} if len(indices) == 1 else {"video": {"do_sample_frames": False, "max_soft_tokens": tokens}}
                    vector, metrics = encoder._encode(item, processing_kwargs=processing)
                    result = {"embedding": vector.tolist(), "metrics": {**metrics, "frame_indices": indices,
                              "visual_tokens": tokens, "chronological_video": len(indices) > 1}}
                else:
                    raise ValueError("UNSUPPORTED_V6_OPERATION")
            response["result"] = result
        except Exception as exc:
            response["error"] = {"code": getattr(exc, "code", "V6_WORKER_FAILED"), "detail": repr(exc)}
        print(json.dumps(response), flush=True)
    if encoder is not None:
        encoder.close()


if __name__ == "__main__":
    main()
