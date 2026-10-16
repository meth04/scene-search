from __future__ import annotations

import csv
import hashlib
import io
import json
from datetime import UTC, datetime, timedelta

from src.indexing.domain.entities import canonical_json
from src.indexing.domain.windows import source_uuid
from src.retrieval.application.window_common import require, utc_now
from src.retrieval.domain.entities import ActorContext


def run_export(export_service, job_id, phase="all"):
    """Run durable export phases from the immutable request snapshot.

    Args:
        export_service: Export coordinator owning the repository, object store and access checks.
        job_id: Persisted job identifier.
        phase: Export phase: validate, build, publish, or all.

    Returns:
        Public export job state after the requested phase.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    if phase == "all":
        for step in ("validate", "build", "publish"):
            result = export_service.run_export(job_id, step)
            if result["status"] in {"completed", "revoked", "expired"}:
                return result
        return result
    job = require(export_service.repository.get("export_job", job_id))
    if job["status"] in {"completed", "revoked", "expired"}:
        return export_service.public_job(job)
    snapshot = require(export_service.repository.get("export_snapshot", job["snapshot_id"]))
    from src.retrieval.config import ExportSettings

    settings = ExportSettings.model_validate(snapshot["settings"])
    actor = ActorContext.model_validate(snapshot["actor"])
    try:
        with export_service.repository.transaction() as tx:
            job = tx.get("export_job", job_id)
            if job["status"] in {"revoked", "expired", "completed"}:
                return export_service.public_job(job)
            job["status"] = "validating" if phase == "validate" else "running"
            tx.put("export_job", job_id, job)
        export_service._validate_sources(snapshot, actor)
        if phase == "validate":
            return export_service.public_job(job)
        if phase == "build":
            artifacts = export_service._build_artifacts(job_id, snapshot, settings)
            export_service.repository.put(
                "export_build", job_id, {"artifacts": artifacts, "snapshot_hash": snapshot["snapshot_hash"]}
            )
            return export_service.public_job(job)
        if phase != "publish":
            raise ValueError("EXPORT_PHASE_INVALID")
        build = require(export_service.repository.get("export_build", job_id), "EXPORT_BUILD_MISSING")
        if build["snapshot_hash"] != snapshot["snapshot_hash"]:
            raise ValueError("EXPORT_SNAPSHOT_CHANGED")
        artifacts = build["artifacts"]
        for artifact in artifacts.values():
            info = export_service.store.stat(artifact["key"])
            if not info or export_service._artifact_checksum(artifact["key"]) != artifact["sha256"]:
                raise ValueError("ARTIFACT_CHECKSUM_MISMATCH")
        export_service._validate_sources(snapshot, actor)
        marker_key = f"{settings.published_prefix}/{job_id}/complete.json"
        marker = canonical_json(
            {"snapshot_hash": snapshot["snapshot_hash"], "count": len(snapshot["items"]), "artifacts": artifacts}
        ).encode()
        export_service.store.put_staging(marker_key, io.BytesIO(marker), {})
        with export_service.repository.transaction() as tx:
            current = tx.get("export_job", job_id)
            if current["status"] == "revoked":
                return export_service.public_job(current)
            current.update(
                status="completed",
                completed_at=utc_now(),
                marker_key=marker_key,
                marker_sha256=hashlib.sha256(marker).hexdigest(),
                artifacts=artifacts,
                expires_at=(datetime.now(UTC) + timedelta(seconds=settings.artifact_ttl_seconds)).isoformat(),
            )
            tx.put("export_job", job_id, current)
            tx.put(
                "retrieval_audit",
                source_uuid("export-completed", job_id),
                {"action": "export_completed", "job_id": job_id, "at": utc_now()},
            )
        return export_service.public_job(current)
    except Exception as exc:
        with export_service.repository.transaction() as tx:
            current = tx.get("export_job", job_id)
            if current["status"] != "revoked":
                current.update(status="failed", error_code=type(exc).__name__)
                tx.put("export_job", job_id, current)
        raise


def _build_artifacts(export_service, job_id, snapshot, settings):
    # Manifest-only never reads source video bytes or invokes a model.
    """Build and publish manifest files with verified checksums and stable paths.

    Args:
        export_service: Export coordinator owning the repository, object store and access checks.
        job_id: Persisted job identifier.
        snapshot: Immutable export request snapshot.
        settings: Export settings retained in the snapshot.

    Returns:
        Artifact metadata keyed by manifest extension.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    rows = snapshot["items"]
    jsonl = "".join(canonical_json(row) + "\n" for row in rows).encode("utf-8")
    if len([json.loads(line) for line in jsonl.splitlines()]) != len(rows):
        raise ValueError("MANIFEST_VALIDATION_FAILED")
    outputs = {"jsonl": jsonl}
    if snapshot["config"]["manifest_format"] == "csv":
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            safe = {}
            for key, value in row.items():
                text = (
                    canonical_json(value)
                    if isinstance(value, (dict, list))
                    else str(value)
                    if value is not None
                    else ""
                )
                if text.lstrip().startswith(("=", "+", "-", "@")):
                    text = "'" + text
                safe[key] = text
            writer.writerow(safe)
        outputs["csv"] = stream.getvalue().encode("utf-8-sig")
    artifacts = {}
    for extension, content in outputs.items():
        staging = f"{settings.staging_prefix}/{job_id}/manifest.{extension}"
        published = f"{settings.published_prefix}/{job_id}/manifest.{extension}"
        export_service.store.put_staging(staging, io.BytesIO(content), {"snapshot_hash": snapshot["snapshot_hash"]})
        info = export_service.store.publish(staging, published)
        if info.sha256 != hashlib.sha256(content).hexdigest():
            raise ValueError("ARTIFACT_CHECKSUM_MISMATCH")
        artifacts[extension] = {"key": published, "sha256": info.sha256, "size_bytes": info.size_bytes}
    return artifacts
