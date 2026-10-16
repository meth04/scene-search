from __future__ import annotations

from typing import Any

from src.indexing.domain.embedding_contract import DriveRetrieverContract
from src.indexing.domain.entities import ValidationGate, ValidationReport
from src.indexing.domain.windows import source_uuid
from src.indexing.models.driveretriever_adapter import build_driveretriever, checked_embedding, file_checksum
from src.indexing.video.window_sampling import pts_ms


def validate(indexer: Any, index_version: str, config: Any = None) -> ValidationReport:
    """Verify exact windows, samples, payloads, vectors and source checksums.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        index_version: Exact persisted index version.
        config: Validated configuration; saved snapshots retain their original serialized fields.

    Returns:
        Validation report with unchanged gate names and failure details.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator

    catalog = indexer.jobs.get("catalog", index_version)
    if not catalog:
        raise ValueError("INDEX_NOT_FOUND")
    frozen = catalog["state"] in {"published", "retired"}
    job = next((j for j in indexer.jobs.list("job") if j["index_version"] == index_version), None)
    if not job:
        raise ValueError("JOB_NOT_FOUND")
    if job["pipeline_config_hash"] != indexer.config.config_hash:
        from src.indexing.config import PipelineConfig

        retained = WindowIndexingOrchestrator(
            PipelineConfig.model_validate(job["config_snapshot"]), repository=indexer.jobs, vectors=indexer.vectors
        )
        return retained.validate(index_version)
    contract = DriveRetrieverContract.model_validate(catalog["embedding_contract"])
    windows = indexer.jobs.windows(index_version)
    points = {p["id"]: p for p in indexer.vectors.points(catalog["collection_name"])}
    expected = job["total_items"]
    inputs = indexer.jobs.get("job_input", job["job_id"])
    expected_ids = _expected_window_ids(indexer, inputs["tasks"], contract)
    counts_ok = expected > 0 and len(windows) == expected and len(points) == expected
    mismatch, missing_sources = _check_window_records(indexer, windows, points, contract, job, inputs)
    sample_count, sample_mismatches = _check_stored_samples(indexer, index_version, windows)
    mismatch.extend(sample_mismatches)
    encoder = indexer.encoder or build_driveretriever(indexer.config.driveretriever)
    encoder_ok = encoder.contract.hash == contract.hash
    if encoder_ok:
        checked_embedding(encoder.encode_text(["a vehicle on a road"])[0].vector)
    report = ValidationReport(
        index_version=index_version,
        gates=[
            ValidationGate(
                name="complete_windows",
                passed=counts_ok,
                details={"expected": expected, "windows": len(windows), "points": len(points)},
            ),
            ValidationGate(
                name="twelve_samples", passed=sample_count == expected * 12, details={"samples": sample_count}
            ),
            ValidationGate(
                name="exact_ids_payload_vectors",
                passed=not mismatch and set(points) == expected_ids == {w.scene_id for w in windows},
                details={"mismatches": mismatch},
            ),
            ValidationGate(
                name="verified_sources", passed=not missing_sources, details={"invalid_assets": missing_sources}
            ),
            ValidationGate(name="text_encoder", passed=encoder_ok),
        ],
    )
    if indexer.config.scene_metadata:
        report.gates.extend(_metadata_gates(indexer, catalog, windows))
    if inputs["manifest"]["source_id"].startswith("demo-upload-"):
        from src.indexing.video.browser_preview import browser_media_valid

        media_ok = all(
            browser_media_valid(
                indexer.jobs.get("browser_media", a["asset_id"]), indexer.jobs.asset(a["asset_id"]), indexer.reader
            )
            for a in inputs["verified"]["videos"]
        )
        report.gates.append(ValidationGate(name="browser_media", passed=media_ok))
    _save_validation(indexer, index_version, report, contract, job, catalog, frozen)
    return report


def metadata_publication_allowed(catalog: dict, *, production: bool) -> bool:
    """Keep fake inference and unevaluated policies out of production.

    Args:
        catalog: Pinned index metadata contract and scoring runtime.
        production: Whether publication targets production storage.

    Returns:
        Whether metadata provenance and quality permit publication at that level.
    """
    contract = catalog.get("metadata_contract")
    if contract is None:
        return True
    from src.indexing.metadata_config import SceneMetadataConfig

    try:
        settings = SceneMetadataConfig.model_validate(contract["config"])
        if settings.contract() != contract:
            return False
    except (ValueError, KeyError, TypeError):
        return False
    runtime = catalog.get("metadata_runtime")
    if not isinstance(runtime, dict) or type(runtime.get("is_fake")) is not bool:
        return False
    if not production:
        return True
    return runtime["is_fake"] is False and settings.quality_status == "evaluated" and settings.evaluation_report_sha256 is not None


def _metadata_gates(indexer: Any, catalog: dict, windows: list) -> list[ValidationGate]:
    """Inspect metadata completeness, contract and live payload index schema.

    Args:
        indexer: Worker with saved configuration and vector storage.
        catalog: Staging metadata contract and inference provenance.
        windows: Saved window records whose metadata must match the contract.

    Returns:
        Additional validation gates; local simulation is recorded explicitly.
    """
    from src.indexing.application.window_processing import metadata_matches_config

    expected = indexer.config.scene_metadata.contract()
    invalid = [w.scene_id for w in windows if not metadata_matches_config(w, indexer.config)]
    schema_ok = indexer.vectors.metadata_indexes_verified(catalog["collection_name"])
    return [
        ValidationGate(name="scene_metadata", passed=not invalid and catalog.get("metadata_contract") == expected,
                       details={"invalid_windows": invalid, "metadata_contract_hash": expected["hash"]}),
        ValidationGate(name="metadata_payload_indexes", passed=schema_ok is not False,
                       details={"backend": "local_simulation" if schema_ok is None else "qdrant", "server_verified": schema_ok}),
        ValidationGate(name="metadata_publication", passed=metadata_publication_allowed(catalog, production=indexer.config.storage.provider == "production")),
    ]


def _expected_window_ids(indexer, tasks, contract):
    """Compute the exact expected scene identifiers from saved tasks.

    Args:
        indexer: Coordinator holding the source-asset repository.
        tasks: Persisted asset and window-index tasks.
        contract: Pinned segmentation contract.

    Returns:
        Expected scene IDs for tasks whose source asset still exists.
    """
    expected_ids = set()
    for task in tasks:
        asset = indexer.jobs.asset(task["asset_id"])
        if asset:
            start = task["window_index"] * 5000
            end = min(start + 10000, asset.duration_ms)
            expected_ids.add(
                source_uuid(
                    "scene", asset.checksum, asset.video_id, asset.camera, start, end, contract.segmentation_version
                )
            )
    return expected_ids


def _check_window_records(indexer, windows, points, contract, job, inputs):
    """Check source availability, vector payloads and source-frame timelines.

    Args:
        indexer: Coordinator holding source, repository and vector checks.
        windows: Persisted window results in repository order.
        points: Vector points keyed by scene identifier.
        contract: Pinned encoder contract.
        job: Persisted job with its pipeline configuration hash.
        inputs: Saved requester identity and source inputs.

    Returns:
        Ordered mismatched scene IDs and unavailable source IDs; repeated
        IDs retain the original report's indication of multiple failures.
    """
    mismatch, missing_sources = [], []
    checked_sources = set()
    for window in windows:
        asset = indexer.jobs.asset(window.asset_id)
        if not asset or not asset.available:
            missing_sources.append(window.asset_id)
            continue
        if asset.asset_id not in checked_sources:
            try:
                from src.retrieval.domain.entities import ActorContext

                requester = ActorContext.model_validate(inputs["requester"]) if inputs.get("requester") else None
                indexer.reader.check_current_registration(asset, requester)
                if file_checksum(indexer.reader.path(asset.uri)) != asset.checksum:
                    missing_sources.append(asset.asset_id)
            except (OSError, ValueError):
                missing_sources.append(asset.asset_id)
            checked_sources.add(asset.asset_id)
        if window.status != "checkpointed" or not indexer._point_matches(window, asset, points.get(window.scene_id)):
            mismatch.append(window.scene_id)
        if (
            window.embedding_contract_hash != contract.hash
            or window.pipeline_config_hash != job["pipeline_config_hash"]
        ):
            mismatch.append(window.scene_id)
        if (window.tenant_id, window.access_scope_id, window.video_id, window.camera) != (
            asset.tenant_id,
            asset.access_scope_id,
            asset.video_id,
            asset.camera,
        ):
            mismatch.append(window.scene_id)
        for sample in window.samples:
            f = sample.frame
            actual = pts_ms(f.pts, asset.timeline_origin_pts, f.time_base_num, f.time_base_den)
            if (
                abs(actual - f.timestamp_ms) > 1e-6
                or f.stream_index != asset.stream_index
                or (f.time_base_num, f.time_base_den) != (asset.time_base_num, asset.time_base_den)
            ):
                mismatch.append(window.scene_id)
    return mismatch, missing_sources


def _check_stored_samples(indexer, index_version, windows):
    """Compare normalized SQL samples against each window's frame references.

    Args:
        indexer: Coordinator holding the window repository.
        index_version: Exact version whose samples are inspected.
        windows: Persisted window results in repository order.

    Returns:
        SQL sample count and ordered scene IDs for differing sample rows.
    """
    mismatch = []
    with indexer.jobs.transaction() as tx:
        sample_count = tx.execute(
            "SELECT COUNT(*) FROM window_samples WHERE index_version=%s", (index_version,)
        ).fetchone()[0]
        stored_samples = tx.execute(
            """SELECT s.scene_id,s.sample_index,s.frame_id,s.target_ms,s.pts_ms,f.value
                                   FROM window_samples s JOIN source_frames f ON f.frame_id=s.frame_id
                                   WHERE s.index_version=%s""",
            (index_version,),
        ).fetchall()
    import json

    normalized = {(str(row[0]), row[1]): row for row in stored_samples}
    for window in windows:
        for sample in window.samples:
            row = normalized.get((window.scene_id, sample.sample_index))
            if (
                not row
                or str(row[2]) != sample.frame.frame_id
                or abs(row[3] - sample.target_ms) > 1e-6
                or abs(row[4] - sample.frame.timestamp_ms) > 1e-6
                or json.loads(row[5]) != sample.frame.model_dump(mode="json")
            ):
                mismatch.append(window.scene_id)
    return sample_count, mismatch


def _save_validation(indexer: Any, index_version: str, report: ValidationReport, contract: DriveRetrieverContract,
                     job: dict, catalog: dict, frozen: bool) -> None:
    """Atomically persist the report, catalog state and job validation status.

    Args:
        indexer: Coordinator holding the window repository.
        index_version: Version receiving the validation report.
        report: Unchanged validation gate report.
        contract: Pinned encoder contract.
        job: Persisted job identifying the pipeline and job record.
        catalog: Catalog updated in place within the transaction.
        frozen: Whether published or retired state must be preserved.
    """
    with indexer.jobs.transaction() as tx:
        metadata_fields = {}
        if indexer.config.scene_metadata:
            metadata_fields["metadata_contract_hash"] = indexer.config.scene_metadata.contract()["hash"]
            if not frozen:
                from src.indexing.domain.scene_metadata import LIGHTING_LABELS, WEATHER_LABELS

                for group in ("weather", "lighting"):
                    catalog["capabilities"][group] = "generated" if report.passed else "not_generated"
                if report.passed:
                    catalog["metadata_filter_values"] = dict(weather=[*WEATHER_LABELS, "unknown"], lighting=[*LIGHTING_LABELS, "unknown"])
                else:
                    catalog.pop("metadata_filter_values", None)
        tx.put(
            "validation",
            index_version,
            {
                **report.model_dump(mode="json"),
                "passed": report.passed,
                "contract_hash": contract.hash,
                "pipeline_config_hash": job["pipeline_config_hash"],
                "quality_status": "not_evaluated",
                **metadata_fields,
            },
        )
        catalog["state"] = catalog["state"] if frozen else "ready" if report.passed else "staging"
        indexer.jobs.save_catalog(catalog, tx)
        current = tx.get("job", job["job_id"])
        current.update(
            status=current["status"] if frozen else "ready" if report.passed else "failed",
            failed_items=0 if report.passed else 1,
        )
        tx.put("job", job["job_id"], current)
