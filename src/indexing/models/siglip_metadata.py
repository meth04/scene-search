"""Original SigLIP image/text similarity for bounded indexing batches.

Sources: https://huggingface.co/google/siglip2-base-patch16-224 and
https://huggingface.co/docs/transformers/model_doc/siglip2
"""

from pathlib import Path
from typing import Any, Self

import numpy as np

from src.indexing.domain.scene_metadata import LIGHTING_LABELS, WEATHER_LABELS, FrameMetadataScores
from src.indexing.metadata_config import SceneMetadataConfig


class SiglipMetadataAdapter:
    """Cache prompt features and score RGB frames with the original model.

    Attributes:
        config: Pinned descriptions and inference settings.
        model: Evaluation-mode SigLIP, independent of DriveRetriever heads.
        processor: Processor loaded from the same immutable snapshot.
        is_fake: Whether injected test inference is being used.
        artifact_hashes: Checksums of the locally resolved model and preprocessing files.
    """

    def __init__(self, config: SceneMetadataConfig, *, model: Any, processor: Any, is_fake: bool = True) -> None:
        """Encode all concrete descriptions once, without loading another model.

        Args:
            config: Explicit model, prompt and batching settings.
            model: Prepared model with pooled image/text features and scale/bias.
            processor: Matching text and image preprocessing callable.
            is_fake: Injected models are test fixtures unless built by from_config.

        Raises:
            ValueError: If prompt features are malformed or nonfinite.
            Exception: If text preprocessing or model inference fails.
        """
        import torch

        self.config, self.model, self.processor, self.is_fake = config, model, processor, is_fake
        self.artifact_hashes: dict[str, str] = {}
        self.prompts = [config.weather_prompts[k] for k in WEATHER_LABELS] + [config.lighting_prompts[k] for k in LIGHTING_LABELS]
        inputs = self._inputs(processor(text=self.prompts, padding="max_length", truncation=True, return_tensors="pt"))
        with torch.inference_mode():
            self.text_features = self._normalized(model.get_text_features(**inputs))
        if self.text_features.shape[0] != 6:
            raise ValueError("metadata model must return six prompt features")

    @classmethod
    def from_config(cls, config: SceneMetadataConfig) -> Self:
        """Load original SigLIP and preprocessing from an immutable HF snapshot.

        Args:
            config: Pinned revision and cache settings; offline loading is default.

        Returns:
            Real adapter with cached prompt features and artifact checksums.

        Raises:
            RuntimeError: If CUDA is requested but unavailable.
            ValueError: If the snapshot does not contain safe original weights.
            Exception: If artifacts, tokenizer or model cannot be loaded.
        """
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoProcessor

        from src.indexing.models.driveretriever_adapter import file_checksum

        if config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA metadata requested but unavailable; use cpu/fp32")
        root = Path(snapshot_download(repo_id=config.model_id, revision=config.revision,
                                     cache_dir=config.cache_dir, local_files_only=config.local_files_only))
        if root.name != config.revision or not (root / "model.safetensors").is_file():
            raise ValueError("metadata requires the pinned original safetensors snapshot")
        filenames = ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json")
        hashes = {name: file_checksum(root / name) for name in filenames}
        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[config.precision]
        model = AutoModel.from_pretrained(root, local_files_only=True, trust_remote_code=False, use_safetensors=True,
                                         dtype=dtype).to(config.device).eval()
        processor = AutoProcessor.from_pretrained(root, local_files_only=True, trust_remote_code=False)
        adapter = cls(config, model=model, processor=processor, is_fake=False)
        adapter.artifact_hashes = hashes
        return adapter

    def _inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Move model inputs to the configured device and image precision.

        Args:
            inputs: Processor tensor mapping, without mutation.

        Returns:
            Input tensors on the inference device; integer text IDs remain integers.
        """
        import torch

        dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[self.config.precision]
        return {key: value.to(device=self.config.device, dtype=dtype if value.is_floating_point() else value.dtype)
                for key, value in inputs.items()}

    @staticmethod
    def _normalized(output: Any) -> Any:
        """Extract pooled features across supported Transformers return formats.

        Args:
            output: Tensor or model output with a pooler_output tensor.

        Returns:
            Finite L2-normalized two-dimensional features in model precision.

        Raises:
            ValueError: If features are malformed, zero or nonfinite.
        """
        import torch

        features = output if isinstance(output, torch.Tensor) else output.pooler_output
        if features.ndim != 2 or not torch.isfinite(features).all():
            raise ValueError("metadata features must be finite matrices")
        norms = features.norm(p=2, dim=-1, keepdim=True)
        if (norms <= 0).any():
            raise ValueError("metadata features must be nonzero")
        return features / norms

    def score_frames(self, frames: list[np.ndarray]) -> list[FrameMetadataScores]:
        """Score RGB uint8 H×W×3 images in bounded consecutive batches.

        Args:
            frames: Nonempty images in source order; an empty list is allowed.

        Returns:
            One score record per frame in the same order; [] for empty input.

        Raises:
            ValueError: If RGB inputs, features or scores are invalid.
            Exception: If image preprocessing or inference fails; never converted to unknown.
        """
        import torch

        for frame in frames:
            if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3
                    or frame.shape[2] != 3 or min(frame.shape[:2]) == 0):
                raise ValueError("metadata frames must be nonempty RGB uint8 HxWx3")
        results = []
        with torch.inference_mode():
            for start in range(0, len(frames), self.config.batch_size):
                batch = frames[start:start + self.config.batch_size]
                inputs = self._inputs(self.processor(images=batch, return_tensors="pt"))
                image = self._normalized(self.model.get_image_features(**inputs))
                if image.shape[0] != len(batch):
                    raise ValueError("metadata image features changed frame count")
                logits = image @ self.text_features.T
                logits = logits * self.model.logit_scale.to(logits.device).exp() + self.model.logit_bias.to(logits.device)
                if not torch.isfinite(logits).all():
                    raise ValueError("metadata logits must be finite")
                for row in logits.sigmoid().float().cpu().tolist():
                    results.append(FrameMetadataScores(weather=dict(zip(WEATHER_LABELS, row[:4], strict=True)),
                                                       lighting=dict(zip(LIGHTING_LABELS, row[4:], strict=True))))
        return results
