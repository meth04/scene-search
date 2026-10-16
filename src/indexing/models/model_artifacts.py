"""Resolve verified model artifacts without loading weights or downloading files."""

import json
from pathlib import Path

from src.indexing.config import ModelConfig
from src.indexing.domain.errors import FatalIndexingError
from src.indexing.models.hf_loader import HuggingFaceLoader


def local_model(config: ModelConfig) -> tuple[Path, str]:
    """Verify a local model or an already verified Hugging Face snapshot.

    Args:
        config: Pinned revision and local directory or Hugging Face cache settings.

    Returns:
        Verified directory and resolved revision. Local models retain the
        configured revision; cached snapshots use their verification receipt.

    Raises:
        FatalIndexingError: If artifacts, checksums, receipt or revision are invalid,
            or the provider cannot resolve local artifacts.
    """
    if config.provider == "local" and config.local_path:
        root = Path(config.local_path)
        HuggingFaceLoader().verify(root)
        return root, config.revision
    if config.provider == "huggingface":
        from huggingface_hub import snapshot_download

        root = Path(
            snapshot_download(
                repo_id=config.repo_id, revision=config.revision, cache_dir=config.cache_dir, local_files_only=True
            )
        )
        receipt = root / ".scenesearch_verified.json"
        if not receipt.is_file():
            raise FatalIndexingError("Hugging Face snapshot is not verified; run models prefetch")
        verified = json.loads(receipt.read_text(encoding="utf-8"))
        if verified.get("requested_revision") != config.revision:
            raise FatalIndexingError("verified snapshot revision differs from configured revision")
        HuggingFaceLoader().verify(root)
        return root, verified["resolved_revision"]
    raise FatalIndexingError("model artifacts require local or huggingface provider")
