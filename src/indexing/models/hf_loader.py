from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.indexing.config import ModelConfig
from src.indexing.domain.entities import ModelManifest
from src.indexing.domain.errors import FatalIndexingError


class HuggingFaceLoader:
    """Prefetches pinned snapshots and verifies their handoff manifest."""

    def prefetch(self, config: ModelConfig) -> tuple[Path, str, ModelManifest]:
        if config.provider != "huggingface" or not config.repo_id:
            raise ValueError("prefetch requires a huggingface model config")
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise FatalIndexingError("Install huggingface-hub to prefetch models") from exc
        token = os.getenv("HF_TOKEN") or None
        try:
            resolved = HfApi(token=token).repo_info(
                config.repo_id, revision=config.revision, files_metadata=False
            ).sha
            patterns = list(config.allow_patterns or [])
            if patterns and "scenesearch_model.json" not in patterns:
                patterns.append("scenesearch_model.json")
            local = snapshot_download(
                repo_id=config.repo_id,
                revision=config.revision,
                allow_patterns=patterns or None,
                cache_dir=config.cache_dir,
                token=token,
            )
        except Exception as exc:
            raise FatalIndexingError(f"Cannot prefetch {config.repo_id}: {exc}") from exc
        manifest = self.verify(Path(local))
        receipt = {
            "repo_id": config.repo_id,
            "requested_revision": config.revision,
            "resolved_revision": resolved,
            "manifest": manifest.model_dump(mode="json"),
        }
        (Path(local) / ".scenesearch_verified.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return Path(local), resolved, manifest

    def verify(self, directory: Path) -> ModelManifest:
        manifest_path = directory / "scenesearch_model.json"
        if not manifest_path.is_file():
            raise FatalIndexingError(f"Missing model manifest: {manifest_path}")
        manifest = ModelManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        for relative, expected in manifest.sha256.items():
            target = (directory / relative).resolve()
            try:
                target.relative_to(directory.resolve())
            except ValueError as exc:
                raise FatalIndexingError(f"Artifact escapes model directory: {relative}") from exc
            if not target.is_file():
                raise FatalIndexingError(f"Missing model artifact: {relative}")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            if digest.lower() != expected.lower():
                raise FatalIndexingError(f"Checksum mismatch: {relative}")
        return manifest

    def verified_snapshot(self, directory: str | Path) -> tuple[Path, dict]:
        root = Path(directory)
        receipt_path = root / ".scenesearch_verified.json"
        if not receipt_path.is_file():
            raise FatalIndexingError("Model cache has no verification receipt; run models prefetch")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        self.verify(root)
        return root, receipt
