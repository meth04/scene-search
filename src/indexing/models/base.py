from pathlib import Path

from src.indexing.domain.entities import ModelDescriptor, ModelManifest


def descriptor_from_manifest(
    manifest: ModelManifest,
    provider: str,
    revision: str,
    resolved_revision: str,
    device: str,
    repo_id: str | None = None,
) -> ModelDescriptor:
    return ModelDescriptor(
        task=manifest.task,
        provider=provider,
        repo_id=repo_id,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        model_version=manifest.model_version,
        artifact_checksums=manifest.sha256,
        framework=manifest.framework,
        device=device,
        contract_version=manifest.contract_version,
    )


def read_model_manifest(model_dir: str | Path) -> ModelManifest:
    path = Path(model_dir) / "scenesearch_model.json"
    return ModelManifest.model_validate_json(path.read_text(encoding="utf-8"))
