from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.indexing.domain.entities import IndexJob, stable_hash
from src.indexing.domain.states import JobStatus
from src.indexing.domain.windows import source_uuid
from src.indexing.models.driveretriever_adapter import build_driveretriever, file_checksum
from src.indexing.video.window_sampling import generate_windows


def prepare(indexer: Any, command: Any, actor: Any = None, *, request_id: str | None = None,
            request_hash: str | None = None) -> IndexJob:
    """Inspect authorized sources and atomically register a reproducible index job.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        command: Validated operation inputs, including any required version and request identity.
        actor: Authenticated actor, tenant and current source access scopes.
        request_id: Caller identity for idempotent request handling.
        request_hash: Hash used to detect reuse of a request ID with changed inputs.

    Returns:
        Existing or newly queued index job; repeated requests retain the original job.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
        PermissionError: If a source lies outside the actor's tenant or access scope.
    """
    from src.indexing.application.window_orchestrator import source_build_hash

    if (request_id is None) != (request_hash is None):
        raise ValueError("request ID and hash must be supplied together")
    config, manifest = command.config, command.manifest
    if actor:
        for item in manifest.videos:
            if (
                str(item.metadata.get("tenant_id", "default")) != actor.tenant_id
                or str(item.metadata.get("access_scope_id", "default")) not in actor.access_scope_ids
            ):
                raise PermissionError("SOURCE_FORBIDDEN")
    encoder = indexer.encoder or build_driveretriever(config.driveretriever)
    indexer.encoder = encoder
    assets = [indexer.reader.inspect(manifest.source_id, item, actor) for item in manifest.videos]
    verified = {
        "source_id": manifest.source_id,
        "dataset_version": manifest.dataset_version,
        "videos": [a.model_dump(mode="json") for a in assets],
    }
    source_digest = source_build_hash()
    version = stable_hash(
        {
            "manifest": verified,
            "pipeline": config.config_hash,
            "contract": encoder.contract.hash,
            "source_build_hash": source_digest,
        }
    )[:24]
    job_id = source_uuid("job", version)
    existing = indexer.jobs.get("job", job_id)
    if existing and request_id is None:
        return IndexJob.model_validate(existing)
    collection = indexer.vectors.ensure(version, encoder.contract.hash, metadata_enabled=config.scene_metadata is not None)
    catalog = {
        "index_version": version,
        "dataset_version": manifest.dataset_version,
        "collection_name": collection,
        "embedding_contract": encoder.contract.model_dump(mode="json"),
        "embedding_contract_hash": encoder.contract.hash,
        "pipeline_config_hash": config.config_hash,
        "state": "staging",
        "capabilities": {
            "objects": "not_generated",
            "weather": "not_generated",
            "lighting": "not_generated",
            "video": True,
            "camera": any(a.camera is not None for a in assets),
            "captured_at": any(a.captured_at is not None for a in assets),
            "region": any(a.region is not None for a in assets),
        },
        "quality_status": "not_evaluated",
    }
    catalog["encoder_config"] = config.driveretriever.model_dump(mode="json")
    if config.scene_metadata:
        catalog["metadata_contract"] = config.scene_metadata.contract()
    catalog["tenant_ids"] = sorted({a.tenant_id for a in assets})
    tasks = [
        {"asset_id": a.asset_id, "window_index": w.window_index}
        for a in assets
        for w in generate_windows(a.duration_ms)
    ]
    job = IndexJob(
        job_id=job_id,
        index_version=version,
        dataset_version=manifest.dataset_version,
        pipeline_config_hash=config.config_hash,
        config_snapshot=config.model_dump(mode="json"),
        status=JobStatus.QUEUED,
        total_items=len(tasks),
    )
    import subprocess

    try:
        code_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        code_revision = "unavailable"
    with indexer.jobs.transaction() as tx:
        if request_id:
            previous_request = tx.get("index_request", request_id)
            if previous_request:
                if previous_request["request_hash"] != request_hash:
                    raise ValueError("IDEMPOTENCY_KEY_REUSED")
                return IndexJob.model_validate(tx.get("job", previous_request["job_id"]))
            tx.put(
                "index_request",
                request_id,
                {
                    "request_hash": request_hash,
                    "job_id": job_id,
                    "actor_id": actor.actor_id if actor else "cli",
                    "tenant_id": actor.tenant_id if actor else "default",
                },
            )
        if tx.get("job", job_id):
            return IndexJob.model_validate(tx.get("job", job_id))
        if not tx.get("catalog", version):
            indexer.jobs.save_catalog(catalog, tx)
        for asset in assets:
            indexer.jobs.save_asset(asset, tx)
        tx.put("job", job_id, job)
        tx.put(
            "job_input",
            job_id,
            {
                "manifest": manifest.model_dump(mode="json"),
                "verified": verified,
                "manifest_hash": stable_hash(verified),
                "tasks": tasks,
                "contract_hash": encoder.contract.hash,
                "code_revision": code_revision,
                "source_code_hash": source_digest,
                "requester": actor.model_dump(mode="json") if actor else None,
                "dependency_lock_hash": file_checksum("requirements.lock")
                if Path("requirements.lock").is_file()
                else None,
            },
        )
        tx.put(
            "index_audit",
            source_uuid("audit", job_id, "created"),
            {
                "action": "created",
                "job_id": job_id,
                "actor_id": actor.actor_id if actor else "cli",
                "at": datetime.now(UTC).isoformat(),
            },
        )
    return job
