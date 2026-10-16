"""DriveRetriever epoch-4 inference heads and strict checkpoint loading.

The training artifact contains only trainable heads. The frozen SigLIP2 backbone
is loaded separately from the model identifier recorded in the artifact.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional

DEFAULT_BACKBONE = "google/siglip2-base-patch16-224"
DEFAULT_BACKBONE_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DEFAULT_CHECKPOINT = Path("data/model-cache/driveretriever-epoch4/best.pt")


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x * self.score(x).softmax(dim=1)).sum(dim=1)


class DriveRetrieverHeads(nn.Module):
    """Training-compatible parameter tree, including auxiliary metadata heads."""

    def __init__(
        self,
        hidden_dim: int,
        num_frames: int,
        embedding_dim: int,
        temporal_layers: int,
        temporal_heads: int,
        temporal_dropout: float,
    ):
        super().__init__()
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, hidden_dim))
        self.logit_scale = nn.Parameter(torch.tensor(0.0))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=temporal_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=temporal_dropout,
            batch_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers, norm=nn.LayerNorm(hidden_dim))
        self.pool = AttentionPool(hidden_dim)
        self.clip_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
        self.frame_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, embedding_dim))
        self.weather_head = nn.Linear(hidden_dim, 4)
        self.road_head = nn.Linear(hidden_dim, 3)
        self.binary_head = nn.Linear(hidden_dim, 4)

    def encode_video_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1:] != self.temporal_pos.shape[1:]:
            raise ValueError(
                f"Expected [batch, {self.temporal_pos.shape[1]}, "
                f"{self.temporal_pos.shape[2]}] frame features; got {tuple(features.shape)}"
            )
        temporal = self.temporal(features + self.temporal_pos)
        return functional.normalize(self.clip_proj(self.pool(temporal)).float(), dim=-1)

    def encode_text_features(self, features: torch.Tensor) -> torch.Tensor:
        return functional.normalize(self.text_proj(features).float(), dim=-1)


@dataclass
class LoadedHeads:
    heads: DriveRetrieverHeads
    config: dict[str, Any]
    checkpoint_sha256: str
    top_level_keys: list[str]
    state_key: str
    epoch: int | None
    backbone_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _state_from_checkpoint(checkpoint: dict[str, Any]) -> tuple[str, dict[str, torch.Tensor]]:
    for key in ("trainable_state_dict", "model_state_dict", "state_dict", "model_state"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(isinstance(v, torch.Tensor) for v in value.values()):
            return key, value
    if checkpoint and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return "<root>", checkpoint
    raise ValueError("Checkpoint has no recognizable tensor state dict")


def load_heads(checkpoint_path: Path) -> LoadedHeads:
    path = checkpoint_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"DriveRetriever checkpoint not found:\n  {path}\n"
            "Copy your epoch-4 best.pt to that path or pass --checkpoint."
        )
    # weights_only prevents arbitrary pickle object construction. CPU loading avoids GPU spikes.
    digest_before = sha256_file(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if sha256_file(path) != digest_before:
        raise ValueError("Checkpoint changed while loading")
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint root must be a dictionary")
    state_key, state = _state_from_checkpoint(checkpoint)
    if all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    config = checkpoint.get("config") or {}
    if not isinstance(config, dict):
        raise ValueError("Checkpoint config must be a dictionary")
    backbone = checkpoint.get("backbone_name") or config.get("model_name") or DEFAULT_BACKBONE
    if backbone != DEFAULT_BACKBONE:
        raise ValueError(f"Unsupported backbone {backbone!r}; expected {DEFAULT_BACKBONE!r}")
    frame_count = int(config.get("num_frames", 12))
    dim = int(config.get("embedding_dim", 512))
    hidden_dim = state.get("temporal_pos", torch.empty(0)).shape[-1]
    if hidden_dim <= 0 or frame_count != 12 or dim != 512:
        raise ValueError(
            f"Unsupported checkpoint geometry: frames={frame_count}, embedding_dim={dim}, hidden_dim={hidden_dim}"
        )
    heads = DriveRetrieverHeads(
        hidden_dim=hidden_dim,
        num_frames=frame_count,
        embedding_dim=dim,
        temporal_layers=int(config.get("temporal_layers", 3)),
        temporal_heads=int(config.get("temporal_heads", 8)),
        temporal_dropout=float(config.get("temporal_dropout", 0.1)),
    )
    expected = heads.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    mismatches = [
        f"{key}: checkpoint {tuple(state[key].shape)} != model {tuple(expected[key].shape)}"
        for key in expected.keys() & state.keys()
        if expected[key].shape != state[key].shape
    ]
    if missing or unexpected or mismatches:
        raise ValueError(
            "Checkpoint is incompatible with the DriveRetriever inference architecture.\n"
            f"Missing keys: {missing}\nUnexpected keys: {unexpected}\nShape mismatches: {mismatches}\n"
            "Refusing to run retrieval with partially loaded weights."
        )
    incompatible = heads.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("Strict load returned incompatible keys")
    heads.eval()
    return LoadedHeads(heads, config, digest_before, list(checkpoint), state_key, checkpoint.get("epoch"), backbone)


def check_embeddings(embedding: torch.Tensor, *, expected_ndim: int) -> None:
    if embedding.ndim != expected_ndim or embedding.shape[-1] != 512:
        raise ValueError(f"Expected {expected_ndim}D embeddings with 512 features; got {tuple(embedding.shape)}")
    if not torch.isfinite(embedding).all():
        raise ValueError("Embeddings contain non-finite values")
    norms = torch.linalg.vector_norm(embedding.float(), dim=-1)
    if not torch.all(torch.abs(norms - 1.0) <= 1e-3):
        raise ValueError(f"Embeddings are not L2-normalized; norms range {norms.min():.6f}..{norms.max():.6f}")


class DriveRetriever:
    def __init__(self, loaded: LoadedHeads, device: torch.device, precision: str = "auto", *, revision: str | None = None, local_files_only: bool = False):
        from transformers import AutoModel, AutoProcessor

        self.device = device
        self.config = loaded.config
        self.heads = loaded.heads.to(device).eval()
        revision = revision or (
            self.config.get("backbone_revision") or self.config.get("model_revision") or DEFAULT_BACKBONE_REVISION
        )
        cache_dir = str(Path("data/model-cache/huggingface").resolve())
        kwargs = {"revision": revision, "local_files_only": local_files_only}
        try:
            self.processor = AutoProcessor.from_pretrained(loaded.backbone_name, cache_dir=cache_dir, **kwargs)
            self.backbone = (
                AutoModel.from_pretrained(loaded.backbone_name, cache_dir=cache_dir, **kwargs).to(device).eval()
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load frozen SigLIP2 backbone {loaded.backbone_name!r}. "
                "A network connection may be needed once to populate data/model-cache/huggingface. "
                f"Original error: {exc}"
            ) from exc
        self.resolved_revision = getattr(self.backbone.config, "_commit_hash", None) or revision or "unrecorded"
        if precision == "auto":
            self.amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
        elif precision == "fp16":
            if device.type != "cuda":
                raise ValueError("fp16 precision requires CUDA")
            self.amp_dtype = torch.float16
        elif precision == "bf16":
            if device.type != "cuda" or not torch.cuda.is_bf16_supported():
                raise ValueError("bf16 precision requires a BF16-capable CUDA device")
            self.amp_dtype = torch.bfloat16
        else:
            self.amp_dtype = None

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None)

    @torch.inference_mode()
    def encode_text(self, query: str) -> torch.Tensor:
        limit = int(self.config.get("max_text_length", 64))
        token_ids = self.processor.tokenizer(query, add_special_tokens=True, truncation=False)["input_ids"]
        if len(token_ids) > limit:
            raise ValueError(f"QUERY_TOO_LONG: query contains {len(token_ids)} tokens; maximum is {limit}")
        tokens = self.processor(
            text=[query],
            padding="max_length",
            truncation=False,
            max_length=limit,
            return_tensors="pt",
        )
        tokens = {key: value.to(self.device) for key, value in tokens.items() if key in {"input_ids", "attention_mask"}}
        with self._autocast():
            output = self.backbone.get_text_features(**tokens)
            pooled = output if isinstance(output, torch.Tensor) else output.pooler_output
            vector = self.heads.encode_text_features(pooled.float()).squeeze(0)
        vector = vector.float().cpu()
        check_embeddings(vector, expected_ndim=1)
        return vector

    @torch.inference_mode()
    def encode_video(self, frames: list[list[Any]]) -> torch.Tensor:
        if not frames or any(len(clip) != self.heads.temporal_pos.shape[1] for clip in frames):
            raise ValueError("Each candidate must contain exactly the checkpoint's frame count")
        flat = [frame for clip in frames for frame in clip]
        inputs = self.processor(images=flat, return_tensors="pt")
        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
            if key in {"pixel_values", "pixel_attention_mask", "spatial_shapes"}
        }
        pieces = []
        chunk_size = int(self.config.get("vision_chunk_size", 64))
        with self._autocast():
            for start in range(0, len(flat), chunk_size):
                output = self.backbone.get_image_features(
                    **{key: value[start : start + chunk_size] for key, value in inputs.items()}
                )
                pieces.append(output if isinstance(output, torch.Tensor) else output.pooler_output)
            features = torch.cat(pieces).reshape(len(frames), len(frames[0]), -1)
            vectors = self.heads.encode_video_features(features.float()).float().cpu()
        check_embeddings(vectors, expected_ndim=2)
        return vectors
