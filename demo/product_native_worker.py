"""Product JSON-lines protocol with frozen evidence semantics and portable FP32."""

from __future__ import annotations

import contextlib
import json
import socket
import sys

import numpy as np

from demo.product_runtime import ProductEncoder


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
                    encoder = ProductEncoder(request["config"]).load()
                    encoder.model.requires_grad_(False)
                    result = {**encoder.metrics, "training": False, "gradients": False}
                elif operation == "text":
                    vector, metrics = encoder.encode_text(request["query"])
                    result = {"embedding": vector.tolist(), "metrics": metrics}
                elif operation == "state":
                    result = {
                        "training": bool(encoder.model.training),
                        "any_parameter_requires_grad": any(p.requires_grad for p in encoder.model.parameters()),
                        "any_gradient_tensor": any(p.grad is not None for p in encoder.model.parameters()),
                        "model_load_count": encoder.load_count,
                        "revision": encoder.config["revision"],
                        "dtype": encoder.config["dtype"],
                        "GPU": encoder.metrics["GPU"],
                    }
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
                        selected = [
                            np.asarray(Image.fromarray(c).resize(size, Image.Resampling.BILINEAR)) for c in crops
                        ]
                    array = np.ascontiguousarray(np.stack(selected))
                    # One-frame semantic units use the official image path; windows/tubes the video path.
                    item = {"image": Image.fromarray(array[0])} if len(indices) == 1 else {"video": array}
                    processing = (
                        {"images": {"max_soft_tokens": tokens}}
                        if len(indices) == 1
                        else {"video": {"do_sample_frames": False, "max_soft_tokens": tokens}}
                    )
                    vector, metrics = encoder._encode(item, processing_kwargs=processing)
                    result = {
                        "embedding": vector.tolist(),
                        "metrics": {
                            **metrics,
                            "frame_indices": indices,
                            "visual_tokens": tokens,
                            "chronological_video": len(indices) > 1,
                        },
                    }
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
