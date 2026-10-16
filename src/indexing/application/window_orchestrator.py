from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.indexing.config import PipelineConfig
from src.indexing.domain.entities import IndexJob, stable_hash
from src.indexing.domain.states import JobStatus
from src.indexing.models.driveretriever_adapter import checked_embedding, file_checksum
from src.indexing.video.pts_decoder import PTSDecoder
from src.indexing.video.source_reader import SourceReader


def source_build_hash():
    root = Path(__file__).resolve().parents[2]
    return stable_hash(
        {str(p.relative_to(root)).replace("\\", "/"): file_checksum(p) for p in sorted(root.rglob("*.py"))}
    )


class WindowIndexingOrchestrator:
    """Coordinate preparation, processing, validation and publication of window indexes."""

    def __init__(self, config: PipelineConfig, *, repository: Any = None, vectors: Any = None,
                 encoder: Any = None, reader: SourceReader | None = None, decoder: PTSDecoder | None = None,
                 metadata_adapter: Any = None) -> None:
        """Bind index dependencies while leaving model loading lazy.

        Args:
            config: Validated pipeline settings.
            repository: Existing window repository, or None to build it.
            vectors: Existing vector store, or None to build it.
            encoder: Compatible encoder, or None to load it during preparation.
            reader: Authorized source reader, or None to construct it.
            decoder: PTS decoder, or None to construct it.
            metadata_adapter: Optional scoring adapter bound to this config; loading remains lazy.

        Raises:
            KeyError: If production settings omit DATABASE_URL.
        """
        from src.bootstrap import build_indexing_storage

        self.config = config
        self.jobs, self.vectors = build_indexing_storage(config, repository=repository, vectors=vectors)
        self.encoder = encoder
        self.metadata_adapter = metadata_adapter if config.scene_metadata else None
        self.reader = reader or SourceReader(
            config.driveretriever.source_roots, config.driveretriever.source_registry_path
        )
        self.decoder = decoder or PTSDecoder()

    def prepare(self, command, actor=None, *, request_id=None, request_hash=None):
        """Inspect authorized sources and atomically register a reproducible index job.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.
            request_id: Caller identity for idempotent request handling.
            request_hash: Hash used to detect reuse of a request ID with changed inputs.

        Returns:
            Existing or newly queued index job; repeated requests retain the original job.
        """
        from src.indexing.application.index_preparation import prepare

        return prepare(self, command, actor, request_id=request_id, request_hash=request_hash)

    def process_group(self, job_id, start=0, heartbeat=None):
        """Process one checkpoint group while preserving source, contract and worker checks.

        Args:
            job_id: Persisted job identifier.
            start: First task offset to process.
            heartbeat: Optional callback invoked while processing tasks.

        Returns:
            Status, next task offset and completion flag for the workflow coordinator.
        """
        from src.indexing.application.window_processing import process_group

        return process_group(self, job_id, start, heartbeat)

    @staticmethod
    def _point_matches(window, asset, point):
        if not point or point["id"] != window.scene_id or point["payload"] != window.payload(asset):
            return False
        try:
            checked_embedding(point["vector"])
            return bool(np.allclose(point["vector"], window.embedding.vector, atol=1e-6, rtol=1e-5))
        except (ValueError, TypeError):
            return False

    def _job_status(self, job_id, status):
        with self.jobs.transaction() as tx:
            raw = tx.get("job", job_id)
            if raw["status"] in {"paused", "cancelled"} and status == "failed":
                return
            raw["status"] = status
            raw["updated_at"] = datetime.now(UTC).isoformat()
            tx.put("job", job_id, raw)

    def run_job(self, job_id):
        from src.indexing.config import PipelineConfig

        saved = self.jobs.get_job(job_id)
        if saved and saved.pipeline_config_hash != self.config.config_hash:
            retained = WindowIndexingOrchestrator(
                PipelineConfig.model_validate(saved.config_snapshot), repository=self.jobs, vectors=self.vectors
            )
            return retained.run_job(job_id)
        start = 0
        try:
            while True:
                result = self.process_group(job_id, start)
                if result["done"] or result["status"] == "paused":
                    break
                start = result["next"]
            job = self.jobs.get_job(job_id)
            if job.status == JobStatus.RUNNING:
                inputs = self.jobs.get("job_input", job_id)
                if inputs["manifest"]["source_id"].startswith("demo-upload-"):
                    from src.indexing.video.browser_preview import browser_media_valid, prepare_browser_media

                    self._job_status(job_id, "validating")
                    for item in inputs["verified"]["videos"]:
                        asset = self.jobs.asset(item["asset_id"])
                        existing = self.jobs.get("browser_media", asset.asset_id)
                        if not browser_media_valid(existing, asset, self.reader):
                            self.jobs.put("browser_media", asset.asset_id, prepare_browser_media(asset, self.reader))
                self.validate(job.index_version, self.config)
            return self.jobs.get_job(job_id)
        except Exception as exc:
            with self.jobs.transaction() as tx:
                job = tx.get("job", job_id)
                job.update(
                    status=job["status"] if job["status"] in {"paused", "cancelled"} else "failed", failed_items=1
                )
                tx.put("job", job_id, job)
                tx.put("job_error", job_id, {"type": type(exc).__name__, "message": str(exc)})
            raise

    def run(self, command, resume=False):
        job = self.prepare(command)
        if job.status in {JobStatus.READY, JobStatus.PUBLISHED}:
            return job
        if job.status == JobStatus.CANCELLED:
            raise ValueError("JOB_CANCELLED")
        if resume:
            self._job_status(job.job_id, "queued")
        return self.run_job(job.job_id)

    def resume(self, command):
        return self.run(command, resume=True)

    def validate(self, index_version, config=None):
        """Verify exact windows, samples, payloads, vectors and source checksums.

        Args:
            index_version: Exact persisted index version.
            config: Validated configuration; saved snapshots retain their original serialized fields.

        Returns:
            Validation report with unchanged gate names and failure details.
        """
        from src.indexing.validation.window_index import validate

        return validate(self, index_version, config)

    def publish(self, index_version, alias="scenes_production", *, actor_id="cli", expected_generation=None):
        """Publish a validated index with generation checks and smoke-test recovery.

        Args:
            index_version: Exact persisted index version.
            alias: Active-index alias to update.
            actor_id: Actor identifier recorded in audit events.
            expected_generation: Expected active-index generation, or None to omit that comparison.

        Returns:
            Previously active index version, or None for the first publication.
        """
        from src.indexing.application.index_publication import publish

        return publish(self, index_version, alias, actor_id=actor_id, expected_generation=expected_generation)

    def _smoke(self, catalog, encoder, actor_id="cli"):
        """Check that the target index returns an authorized source window.

        Args:
            catalog: Pinned index catalog, capabilities and embedding contract.
            encoder: Existing compatible encoder.
            actor_id: Actor identifier recorded in audit events.
        """
        from src.indexing.application.index_publication import _smoke

        return _smoke(self, catalog, encoder, actor_id)

    def rollback(self, index_version, alias="scenes_production", *, actor_id="cli", expected_generation=None):
        """Restore an explicit index or deactivate the first published version.

        Args:
            index_version: Exact persisted index version.
            alias: Active-index alias to update.
            actor_id: Actor identifier recorded in audit events.
            expected_generation: Expected active-index generation, or None to omit that comparison.

        Returns:
            Previously active version, or None when no active index exists.
        """
        from src.indexing.application.index_publication import rollback

        return rollback(self, index_version, alias, actor_id=actor_id, expected_generation=expected_generation)

    def control(self, job_id, action):
        with self.jobs.transaction() as tx:
            job = tx.get("job", job_id)
            if not job:
                raise ValueError("JOB_NOT_FOUND")
            allowed = {"pause": "paused", "resume": "queued", "retry": "queued", "cancel": "cancelled"}
            if action not in allowed:
                raise ValueError("JOB_ACTION_INVALID")
            from_states = {
                "pause": {"queued", "running"},
                "resume": {"paused"},
                "retry": {"failed"},
                "cancel": {"queued", "running", "paused", "failed"},
            }
            if job["status"] not in from_states[action]:
                raise ValueError("JOB_STATE_CONFLICT")
            job["status"] = allowed[action]
            tx.put("job", job_id, job)
        return IndexJob.model_validate(job)
