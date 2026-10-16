"""Portable FP32 execution for the product; frozen research encoders stay intact."""

from __future__ import annotations

import json
import os
import time

import numpy as np

from hybrid_v2.embeddinggemma2 import (
    DIMENSION,
    QUERY_PREFIX,
    ROOT,
    EmbeddingGemma2Encoder,
    EmbeddingGemma2Error,
    cached_model_path,
    normalized,
    safe_dtype,
)


def requested_device():
    value = os.getenv("SCENESEARCH_DEVICE")
    if value is None:
        try:
            value = json.loads((ROOT / "outputs/product/runtime.json").read_text())["device"]
        except (OSError, ValueError, KeyError):
            value = "auto"
    if value not in {"auto", "cpu", "cuda"}:
        raise ValueError("SCENESEARCH_DEVICE must be auto, cpu or cuda")
    return value


def execution_device(torch):
    requested = requested_device()
    if requested == "cuda" and not torch.cuda.is_available():
        raise EmbeddingGemma2Error("PRODUCT_CUDA_UNAVAILABLE", "Use setup -Device cpu")
    return "cuda" if requested != "cpu" and torch.cuda.is_available() else "cpu"


class ProductEncoder(EmbeddingGemma2Encoder):
    def load(self):
        with self.lock:
            if self.model is not None:
                return self
            import psutil
            import torch
            from sentence_transformers import SentenceTransformer

            if self.config["dtype"] != "float32":
                raise EmbeddingGemma2Error("PRODUCT_FP32_REQUIRED")
            self.device = execution_device(torch)
            # Keep the exact verified Turing/GTX1650 loading path.
            if self.device == "cuda" and safe_dtype(torch) == torch.float32:
                return super().load()
            torch.set_num_threads(4)
            if self.device == "cuda":
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
            started = time.perf_counter()
            process = psutil.Process()
            before = process.memory_info().rss
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            factory = self.model_factory or SentenceTransformer
            self.model = factory(
                str(cached_model_path()),
                device=self.device,
                local_files_only=True,
                model_kwargs={"torch_dtype": torch.float32},
                config_kwargs={"audio_config": None},
            )
            self.model.eval()
            self.model.requires_grad_(False)
            if self.device == "cuda":
                torch.cuda.synchronize()
            if self.model.prompts.get("SearchQuery") != QUERY_PREFIX:
                self.close()
                raise EmbeddingGemma2Error("EG2_OFFICIAL_PROMPT_MISMATCH")
            if self.model.get_sentence_embedding_dimension() != DIMENSION:
                self.close()
                raise EmbeddingGemma2Error("EG2_DIMENSION_MISMATCH")
            if any("audio" in n for n, _ in self.model.named_parameters()):
                self.close()
                raise EmbeddingGemma2Error("EG2_AUDIO_ENCODER_LOADED")
            if any(p.dtype != torch.float32 or p.device.type != self.device for p in self.model.parameters()):
                self.close()
                raise EmbeddingGemma2Error("PRODUCT_PARAMETER_CONTRACT_INVALID")
            self.load_count += 1
            self.metrics = {
                "GPU": torch.cuda.get_device_name(0) if self.device == "cuda" else "CPU",
                "device": self.device,
                "dtype": "float32",
                "CUDA": torch.version.cuda,
                "PyTorch": torch.__version__,
                "audio_encoder_loaded": False,
                "cold_load_seconds": time.perf_counter() - started,
                "model_load_RAM_bytes": process.memory_info().rss,
                "model_load_RAM_delta_bytes": process.memory_info().rss - before,
                "model_load_VRAM_bytes": torch.cuda.memory_allocated() if self.device == "cuda" else 0,
                "model_load_peak_VRAM_bytes": torch.cuda.max_memory_allocated() if self.device == "cuda" else 0,
                "load_count": self.load_count,
            }
            return self

    def _encode(self, inputs, **kwargs):
        with self.lock:
            self.load()
            if self.device == "cuda":
                return super()._encode(inputs, **kwargs)
            import psutil
            import torch

            started = time.perf_counter()
            with torch.inference_mode():
                result = self.model.encode(
                    inputs,
                    batch_size=1,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    **kwargs,
                )
            vector = normalized(result).reshape(-1)
            return vector, {
                "seconds": time.perf_counter() - started,
                "peak_VRAM_bytes": 0,
                "RAM_bytes": psutil.Process().memory_info().rss,
                "norm": float(np.linalg.norm(vector)),
                "finite": True,
            }
