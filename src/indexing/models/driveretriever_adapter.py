from __future__ import annotations

import hashlib
import importlib.metadata
import threading
from pathlib import Path

import numpy as np

from src.indexing.config import DriveRetrieverConfig
from src.indexing.domain.embedding_contract import DriveRetrieverContract
from src.indexing.domain.entities import Embedding, stable_hash


def file_checksum(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_embedding(vector) -> Embedding:
    array = np.asarray(vector, dtype=np.float32)
    if array.shape != (512,) or not np.isfinite(array).all() or abs(float(np.linalg.norm(array))-1) > 1e-3:
        raise ValueError("EMBEDDING_CONTRACT_MISMATCH: expected normalized finite FP32[512]")
    return Embedding(vector=array.tolist(), dimension=512, normalized=True)


class FakeDriveRetrieverAdapter:
    """Deterministic fixture encoder. Never accepted by production publication."""
    def __init__(self, config: DriveRetrieverConfig):
        self.contract = DriveRetrieverContract(
            requested_revision=config.revision, resolved_revision=config.revision,
            checkpoint_sha256=stable_hash("fake-driveretriever-v1"),
            architecture={"fixture": "fake-driveretriever-v1"},
            processor={"fixture": True}, tokenizer={"fixture": True},
            runtime={"fixture": True, "dtype": "float32"}, is_fake=True,
        )

    def encode_text(self, queries: list[str]) -> list[Embedding]:
        return [self._vector(q.encode()) for q in queries]

    def encode_video_batch(self, windows: list[list]) -> list[Embedding]:
        if not windows or any(len(w) != 12 for w in windows):
            raise ValueError("every window must contain 12 RGB frames")
        return [self._vector(np.asarray(w[0]).tobytes()) for w in windows]

    @staticmethod
    def _vector(seed: bytes) -> Embedding:
        values = np.random.default_rng(int.from_bytes(hashlib.sha256(seed).digest()[:8], "big")).normal(size=512)
        return checked_embedding(values / np.linalg.norm(values))


class DriveRetrieverAdapter:
    def __init__(self, config: DriveRetrieverConfig):
        import torch

        from src.indexing.models.driveretriever import DriveRetriever, load_heads

        device = torch.device(config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA embedding requested but unavailable. Install CUDA-enabled torch/torchvision "
                "and check the NVIDIA driver, or set driveretriever.device: cpu and precision: fp32."
            )
        loaded = load_heads(Path(config.checkpoint))
        if config.checkpoint_sha256 and loaded.checkpoint_sha256 != config.checkpoint_sha256:
            raise ValueError("EMBEDDING_CONTRACT_MISMATCH: checkpoint checksum differs")
        recorded = loaded.config.get("backbone_revision") or loaded.config.get("model_revision")
        if recorded and recorded != config.revision:
            raise ValueError("checkpoint backbone revision differs from requested revision")
        self.model = DriveRetriever(loaded, device, config.precision,
                                   revision=config.revision, local_files_only=config.local_files_only)
        hidden = loaded.heads.temporal_pos.shape[-1]
        if (self.model.backbone.config.vision_config.hidden_size != hidden
                or self.model.backbone.config.text_config.hidden_size != hidden):
            raise ValueError("EMBEDDING_CONTRACT_MISMATCH: backbone and head feature sizes differ")
        import json

        from huggingface_hub import try_to_load_from_cache
        artifact_hashes = {}
        cache = str(Path("data/model-cache/huggingface").resolve())
        filenames = ["model.safetensors", "pytorch_model.bin", "model.safetensors.index.json", "pytorch_model.bin.index.json"]
        for filename in list(filenames):
            location = try_to_load_from_cache(loaded.backbone_name, filename, cache_dir=cache,
                                             revision=self.model.resolved_revision)
            if isinstance(location, str) and Path(location).is_file() and filename.endswith("index.json"):
                filenames.extend(json.loads(Path(location).read_text(encoding="utf-8"))["weight_map"].values())
        for filename in sorted(set(filenames)):
            location = try_to_load_from_cache(loaded.backbone_name, filename, cache_dir=cache,
                                             revision=self.model.resolved_revision)
            if isinstance(location, str) and Path(location).is_file():
                artifact_hashes[filename] = file_checksum(location)
        tokenizer = self.model.processor.tokenizer
        processor = self.model.processor.image_processor
        tokenizer_config_path = try_to_load_from_cache(
            loaded.backbone_name, "tokenizer_config.json", cache_dir=cache,
            revision=self.model.resolved_revision,
        )
        if not isinstance(tokenizer_config_path, str) or not Path(tokenizer_config_path).is_file():
            raise ValueError("EMBEDDING_CONTRACT_MISMATCH: cached tokenizer configuration is missing")
        tokenizer_config = json.loads(Path(tokenizer_config_path).read_text(encoding="utf-8"))
        # Hash the actual tokenizer vocabulary and image processing configuration.
        self.contract = DriveRetrieverContract(
            requested_revision=config.revision, resolved_revision=self.model.resolved_revision,
            checkpoint_sha256=loaded.checkpoint_sha256,
            backbone_sha256=artifact_hashes,
            runtime_lock_hash=file_checksum(Path(__file__).resolve().parents[3] / "requirements.lock"),
            architecture={"state_shapes": {k: list(v.shape) for k, v in loaded.heads.state_dict().items()},
                          "temporal_layers": len(loaded.heads.temporal.layers),
                          "temporal_heads": loaded.heads.temporal.layers[0].self_attn.num_heads,
                          "temporal_dropout": loaded.heads.temporal.layers[0].dropout.p},
            processor={"revision": self.model.resolved_revision, "config": processor.to_dict()},
            tokenizer={"revision": self.model.resolved_revision, "class": type(tokenizer).__name__,
                       "vocabulary_hash": stable_hash(tokenizer.get_vocab()),
                       "special_tokens": tokenizer.special_tokens_map,
                       "configuration": tokenizer_config,
                       "configuration_sha256": file_checksum(tokenizer_config_path),
                       "tokenizer_json_hash": hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()},
            runtime={"torch": torch.__version__, "transformers": importlib.metadata.version("transformers"),
                     "implementation_hash": stable_hash({name: file_checksum(Path(__file__).with_name(name))
                                                         for name in ("driveretriever.py", "driveretriever_adapter.py")}),
                     "precision": config.precision, "device_type": self.model.device.type,
                     "heads_dtype": "float32", "backbone_dtype": str(next(self.model.backbone.parameters()).dtype)},
            max_text_length=int(loaded.config.get("max_text_length", 64)),
        )
        self.lock = threading.RLock()
        self.batch_size = config.batch_size
        self.adjustments = []

    def encode_text(self, queries: list[str]) -> list[Embedding]:
        with self.lock:
            return [checked_embedding(self.model.encode_text(q).numpy()) for q in queries]

    def encode_video_batch(self, windows: list[list]) -> list[Embedding]:
        import torch
        result, offset = [], 0
        with self.lock:
            while offset < len(windows):
                size = min(self.batch_size, len(windows)-offset)
                try:
                    output = self.model.encode_video(windows[offset:offset+size])
                    result.extend(checked_embedding(v.numpy()) for v in output)
                    offset += size
                except torch.cuda.OutOfMemoryError:
                    if size == 1:
                        raise
                    self.batch_size = max(1, size // 2)
                    self.adjustments.append({"reason": "cuda_oom", "batch_size": self.batch_size})
                    torch.cuda.empty_cache()
        return result


_ENCODERS = {}
_ENCODER_LOCK = threading.RLock()


def build_driveretriever(config: DriveRetrieverConfig):
    # Rehash the artifact before looking in cache so file replacement cannot reuse old weights.
    digest = "fake" if config.provider == "fake" else file_checksum(config.checkpoint)
    key = stable_hash({"config": config.model_dump(mode="json"), "checkpoint": digest})
    with _ENCODER_LOCK:
        if key not in _ENCODERS:
            cls = FakeDriveRetrieverAdapter if config.provider == "fake" else DriveRetrieverAdapter
            _ENCODERS[key] = cls(config)
        return _ENCODERS[key]
