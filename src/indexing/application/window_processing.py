from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from src.indexing.config import PipelineConfig
from src.indexing.domain.embedding_contract import DriveRetrieverContract
from src.indexing.domain.entities import Embedding
from src.indexing.domain.scene_metadata import AGGREGATION_VERSION, SCHEMA_VERSION, SceneMetadata
from src.indexing.domain.windows import VideoAsset, WindowResult, WindowSample, source_uuid
from src.indexing.models.driveretriever_adapter import build_driveretriever, file_checksum
from src.indexing.telemetry import WINDOWS, stage
from src.indexing.video.window_sampling import Window


def process_group(indexer: Any, job_id: str, start: int = 0,
                  heartbeat: Callable[[dict], None] | None = None) -> dict:
    """Process one checkpoint group while preserving source, contract and worker checks.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        job_id: Persisted job identifier.
        start: First task offset to process.
        heartbeat: Optional callback invoked while processing tasks.

    Returns:
        Status, next task offset and completion flag for the workflow coordinator.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
        PermissionError: If the requester no longer has access to a source.
        Exception: If decoding exhausts retries or a storage operation fails.
    """
    from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator, source_build_hash

    raw = indexer.jobs.get("job", job_id)
    if not raw:
        raise ValueError("JOB_NOT_FOUND")
    if raw["status"] in {"paused", "cancelled", "ready", "published"}:
        return {"status": raw["status"], "next": start, "done": raw["status"] != "paused"}
    from src.indexing.config import PipelineConfig

    snapshot = PipelineConfig.model_validate(raw["config_snapshot"])
    if snapshot.config_hash != indexer.config.config_hash:
        retained = WindowIndexingOrchestrator(snapshot, repository=indexer.jobs, vectors=indexer.vectors)
        return retained.process_group(job_id, start, heartbeat)
    encoder = indexer.encoder or build_driveretriever(snapshot.driveretriever)
    indexer.encoder = encoder
    inputs = indexer.jobs.get("job_input", job_id)
    if inputs["source_code_hash"] != source_build_hash():
        raise ValueError("JOB_BUILD_CHANGED: resume with the original worker code")
    if encoder.contract.hash != inputs["contract_hash"]:
        raise ValueError("EMBEDDING_CONTRACT_MISMATCH")
    with indexer.jobs.transaction() as tx:
        current = tx.get("job", job_id)
        if current["status"] in {"paused", "cancelled", "ready", "published"}:
            return {"status": current["status"], "next": start, "done": current["status"] != "paused"}
        current["status"] = "running"
        tx.put("job", job_id, current)
    tasks = inputs["tasks"]
    stop = min(len(tasks), start + snapshot.driveretriever.window_group_size)
    pending, images = _collect_pending_windows(indexer, raw, inputs, tasks[start:stop], encoder, heartbeat)
    with stage("video_encode"):
        embeddings = encoder.encode_video_batch(images) if images else []
    for (asset, window, scene_id, samples), embedding, frames in zip(pending, embeddings, images, strict=True):
        metadata = _score_window_metadata(indexer, job_id, scene_id, frames, heartbeat) if snapshot.scene_metadata else None
        result = _window_result(raw, snapshot, encoder.contract, asset, window, scene_id, samples, embedding, metadata)
        _checkpoint_window(indexer, job_id, result, asset, heartbeat)
    job = _record_group_progress(indexer, job_id, encoder)
    return {"status": job["status"], "next": stop, "done": stop == len(tasks)}


def _collect_pending_windows(indexer, raw, inputs, tasks, encoder, heartbeat):
    """Verify sources and decode only windows not already reconciled.

    Args:
        indexer: Coordinator holding source, storage and decoder dependencies.
        raw: Persisted job fields including index version.
        inputs: Saved source tasks and requester identity.
        tasks: Tasks belonging to the current checkpoint group.
        encoder: Encoder with the pinned segmentation contract.
        heartbeat: Optional workflow heartbeat callback.

    Returns:
        Pending window tuples and their decoded image batches, in task order.

    Raises:
        ValueError: If a source or an existing checkpoint is no longer valid.
        PermissionError: If the requester no longer has access to a source.
        Exception: If decoding fails after the applicable retry policy.
    """
    pending, images = [], []
    verified_assets = set()
    job_id = raw["job_id"]
    for task in tasks:
        if heartbeat:
            heartbeat({"job_id": job_id, "window_index": task["window_index"]})
        asset, path = _verify_task_source(indexer, inputs, task, verified_assets)
        k = task["window_index"]
        window = Window(k, k * 5000, min(k * 5000 + 10000, asset.duration_ms))
        scene_id = source_uuid(
            "scene",
            asset.checksum,
            asset.video_id,
            asset.camera,
            window.start_ms,
            window.end_ms,
            encoder.contract.segmentation_version,
        )
        if _reconcile_saved_window(indexer, job_id, raw["index_version"], scene_id, asset):
            continue
        selected = _decode_with_retry(indexer, raw["job_id"], scene_id, path, asset, window)
        samples = [
            WindowSample(sample_index=i, target_ms=window.targets[i], frame=ref)
            for i, (ref, pixels) in enumerate(selected)
        ]
        pending.append((asset, window, scene_id, samples))
        images.append([pixels for ref, pixels in selected])
    return pending, images


def _verify_task_source(indexer, inputs, task, verified_assets):
    """Recheck live source access and checksum once per asset in the group.

    Args:
        indexer: Coordinator holding source reader and repository.
        inputs: Saved requester identity.
        task: Current asset and window task.
        verified_assets: Set updated after each successful checksum check.

    Returns:
        Authorized source asset and its resolved local path.

    Raises:
        ValueError: If the source is unavailable, outside the roots or checksum-changed.
        PermissionError: If source registration or requester access has changed.
    """
    from src.retrieval.domain.entities import ActorContext

    asset = indexer.jobs.asset(task["asset_id"])
    if not asset.available:
        raise ValueError("SOURCE_UNAVAILABLE")
    requester = ActorContext.model_validate(inputs["requester"]) if inputs.get("requester") else None
    indexer.reader.check_current_registration(asset, requester)
    path = indexer.reader.path(asset.uri)
    if asset.asset_id not in verified_assets:
        if file_checksum(path) != asset.checksum:
            raise ValueError("SOURCE_CHECKSUM_MISMATCH")
        verified_assets.add(asset.asset_id)
    return asset, path


def _reconcile_saved_window(indexer: Any, job_id: str, index_version: str, scene_id: str, asset: VideoAsset) -> bool:
    """Recover staged data without regenerating its saved metadata.

    Args:
        indexer: Coordinator holding repository and vector store.
        job_id: Job owning the checkpoint.
        index_version: Pinned window index version.
        scene_id: Stable window identifier.
        asset: Currently verified source asset.

    Returns:
        True if an existing window is checkpointed and needs no decoding.

    Raises:
        ValueError: If a completed window diverges from its vector point.
    """
    saved = indexer.jobs.get_window(index_version, scene_id)
    if not saved:
        return False
    if indexer.config.scene_metadata and not metadata_matches_config(saved, indexer.config):
        raise ValueError("METADATA_CONTRACT_MISMATCH: saved window cannot be checkpointed")
    point = indexer.vectors.get(indexer.vectors.collection_name(index_version), scene_id)
    if indexer._point_matches(saved, asset, point):
        if saved.status != "checkpointed":
            saved.status = "checkpointed"
            indexer.jobs.save_window(job_id, saved, complete=True)
        return True
    if saved.status == "checkpointed":
        raise ValueError("CHECKPOINT_DIVERGED: completed window no longer matches its point")
    if saved.metadata is not None:
        _checkpoint_window(indexer, job_id, saved, asset, None)
        return True
    return False


def metadata_matches_config(window: WindowResult, config: PipelineConfig) -> bool:
    """Check window metadata against the worker's immutable configuration.

    Args:
        window: Saved result whose provenance is inspected.
        config: Pipeline configuration pinned by the owning job.

    Returns:
        True for matching metadata, or for an untouched legacy window when disabled.
    """
    settings = config.scene_metadata
    if settings is None:
        return window.metadata is None
    if window.metadata is None or window.pipeline_config_hash != config.config_hash:
        return False
    for prediction, version in ((window.metadata.weather, settings.weather_prompt_version),
                                (window.metadata.lighting, settings.lighting_prompt_version)):
        if (prediction.model_id, prediction.model_revision, prediction.prompt_version) != (settings.model_id, settings.revision, version):
            return False
    return window.metadata_schema_version == SCHEMA_VERSION and window.aggregation_version == AGGREGATION_VERSION


def _score_window_metadata(indexer: Any, job_id: str, scene_id: str, frames: list,
                           heartbeat: Callable[[dict], None] | None) -> SceneMetadata:
    """Reuse sampled RGB frames, heartbeat per batch and record inference failures.

    Args:
        indexer: Worker holding optional lazy model and saved configuration.
        job_id: Owning job identifier for retry records.
        scene_id: Stable window identifier.
        frames: The same twelve RGB uint8 arrays used by DriveRetriever.
        heartbeat: Optional callback invoked at metadata batch boundaries.

    Returns:
        Independent aggregated predictions for the window.

    Raises:
        Exception: Original model, configuration or storage failure, recorded as metadata stage.
    """
    from src.indexing.aggregation.scene_metadata import aggregate_scene_metadata
    from src.indexing.models.siglip_metadata import SiglipMetadataAdapter

    config = indexer.config.scene_metadata
    try:
        with stage("metadata"):
            adapter = indexer.metadata_adapter
            if adapter is None:
                adapter = SiglipMetadataAdapter.from_config(config)
                indexer.metadata_adapter = adapter
            if adapter.config != config:
                raise ValueError("METADATA_CONTRACT_MISMATCH: adapter uses different settings")
            job = indexer.jobs.get("job", job_id)
            with indexer.jobs.transaction() as tx:
                catalog = tx.get("catalog", job["index_version"])
                previous = catalog.get("metadata_runtime", {})
                runtime = {"is_fake": bool(getattr(adapter, "is_fake", True)),
                           "artifact_hashes": getattr(adapter, "artifact_hashes", {})}
                if previous and previous != runtime:
                    raise ValueError("METADATA_CONTRACT_MISMATCH: scoring runtime changed during indexing")
                catalog["metadata_runtime"] = runtime
                indexer.jobs.save_catalog(catalog, tx)
            scores = []
            for start in range(0, len(frames), config.batch_size):
                if heartbeat:
                    heartbeat({"scene_id": scene_id, "stage": "metadata", "frame_offset": start})
                batch = frames[start:start + config.batch_size]
                result = adapter.score_frames(batch)
                if len(result) != len(batch):
                    raise ValueError("metadata adapter changed frame count")
                scores.extend(result)
            return aggregate_scene_metadata(scores, config)
    except Exception as exc:
        key = source_uuid("metadata-attempt", job_id, scene_id)
        previous = indexer.jobs.get("window_attempt", key) or {}
        indexer.jobs.put("window_attempt", key, {"job_id": job_id, "scene_id": scene_id, "stage": "metadata",
                                                "attempt": previous.get("attempt", 0) + 1, "error_type": type(exc).__name__})
        raise


def _decode_with_retry(indexer, job_id, scene_id, path, asset, window):
    """Decode with at most three attempts, recording each decode failure.

    Args:
        indexer: Coordinator holding decoder and job repository.
        job_id: Job identifier used for attempt records.
        scene_id: Stable window identifier.
        path: Authorized, checksum-verified source path.
        asset: Source timeline metadata.
        window: Fixed video interval to sample.

    Returns:
        Decoder-selected frame references and pixel arrays.

    Raises:
        Exception: The original decode error if nonretryable or attempts expire.
    """
    with stage("decode"):
        for attempt in range(1, 4):
            try:
                return indexer.decoder.sample(path, asset, window)
            except Exception as exc:
                retryable = str(exc).startswith("DECODE_EMPTY_WINDOW") or type(exc).__module__.startswith("av.")
                indexer.jobs.put(
                    "window_attempt",
                    source_uuid("attempt", job_id, scene_id),
                    {
                        "job_id": job_id,
                        "scene_id": scene_id,
                        "stage": "decode",
                        "attempt": attempt,
                        "error_type": type(exc).__name__,
                    },
                )
                if not retryable or attempt == 3:
                    raise


def _window_result(raw: dict, snapshot: PipelineConfig, contract: DriveRetrieverContract, asset: VideoAsset,
                   window: Window, scene_id: str, samples: list[WindowSample], embedding: Embedding,
                   metadata: SceneMetadata | None = None) -> WindowResult:
    """Materialize a staged window using the job's pinned source and model fields.

    Args:
        raw: Persisted job index and dataset versions.
        snapshot: Validated pipeline configuration saved with the job.
        contract: Pinned encoder contract.
        asset: Authorized source asset.
        window: Fixed video interval.
        scene_id: Stable window identifier.
        samples: Selected source-frame references.
        embedding: Validated encoder output.
        metadata: Aggregated predictions, or None for a legacy window.

    Returns:
        Staged result ready for repository and vector reconciliation.
    """
    return WindowResult(
        scene_id=scene_id,
        index_version=raw["index_version"],
        dataset_version=raw["dataset_version"],
        asset_id=asset.asset_id,
        video_id=asset.video_id,
        camera=asset.camera,
        window_index=window.window_index,
        start_ms=window.start_ms,
        end_ms=window.end_ms,
        partial=window.partial,
        tenant_id=asset.tenant_id,
        access_scope_id=asset.access_scope_id,
        samples=samples,
        embedding=embedding,
        embedding_contract_hash=contract.hash,
        pipeline_config_hash=snapshot.config_hash,
        checkpoint_sha256=contract.checkpoint_sha256,
        metadata=metadata,
        metadata_schema_version=SCHEMA_VERSION if metadata else None,
        aggregation_version=AGGREGATION_VERSION if metadata else None,
    )


def _checkpoint_window(indexer, job_id, result, asset, heartbeat):
    """Stage, upsert and reconcile a window before committing its checkpoint.

    Args:
        indexer: Coordinator holding repository and vector store.
        job_id: Job owning the checkpoint.
        result: Staged window, mutated to checkpointed after reconciliation.
        asset: Source asset used for the vector payload.
        heartbeat: Optional callback invoked after checkpoint publication.

    Raises:
        ValueError: If the written vector does not match the staged window.
    """
    indexer.jobs.save_window(job_id, result)
    collection = indexer.vectors.collection_name(result.index_version)
    with stage("vector_upsert"):
        indexer.vectors.upsert(collection, result, asset)
    if not indexer._point_matches(result, asset, indexer.vectors.get(collection, result.scene_id)):
        raise ValueError("RECONCILIATION_FAILED")
    result.status = "checkpointed"
    indexer.jobs.save_window(job_id, result, complete=True)
    WINDOWS.inc()
    if heartbeat:
        heartbeat({"scene_id": result.scene_id, "stage": "checkpointed"})


def _record_group_progress(indexer, job_id, encoder):
    """Atomically record checkpoint counts and the encoder's runtime report.

    Args:
        indexer: Coordinator holding the job repository.
        job_id: Processed job identifier.
        encoder: Encoder carrying contract and optional batch adjustments.

    Returns:
        Updated job fields, retaining any concurrent pause or cancellation.
    """
    with indexer.jobs.transaction() as tx:
        job = tx.get("job", job_id)
        job["processed_items"] = tx.execute(
            "SELECT COUNT(*) FROM window_checkpoints WHERE job_id=%s AND stage='checkpointed'", (job_id,)
        ).fetchone()[0]
        job["failed_items"] = 0
        job["updated_at"] = datetime.now(UTC).isoformat()
        tx.put("job", job_id, job)
        tx.put(
            "runtime_report",
            job_id,
            {
                "embedding_contract_hash": encoder.contract.hash,
                "batch_adjustments": getattr(encoder, "adjustments", []),
                "committed_windows": job["processed_items"],
            },
        )
    return job
