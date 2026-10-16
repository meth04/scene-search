from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from src.indexing.domain.entities import stable_hash
from src.indexing.domain.windows import source_uuid
from src.retrieval.application.window_common import require, utc_now
from src.retrieval.application.window_result_set_service import WindowResultSetService
from src.retrieval.domain.errors import ConflictError, ForbiddenError, RetrievalError
from src.retrieval.ranking.pagination import SignedCursorService


class WindowExportService:
    """Queue immutable export snapshots and control their status and downloads."""

    def __init__(self, search, store):
        """Share search authorization and storage without loading model weights.

        Args:
            search: Search service supplying media access, repository and settings.
            store: Object store used for export artifacts and completion markers.
        """
        self.search = search
        self.repository = search.repository
        self.store = store
        self.config = search.config

    def request(self, command, actor):
        """Authorize reviewed results and atomically snapshot an idempotent export request.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Public queued or previously created export job.

        Raises:
            RetrievalError: If the query, filter or export request violates its supported contract.
            ConflictError: If versions, request identity or the current state prevent the operation.
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        if command.config.format != "manifest_only":
            raise RetrievalError("clip export is not available", details={"error_code": "EXPORT_TYPE_NOT_AVAILABLE"})
        request_hash = stable_hash(
            {
                "result_set_id": command.result_set_id,
                "version": command.expected_version,
                "config": command.config.model_dump(mode="json"),
            }
        )
        job_id = source_uuid("export", actor.actor_id, command.idempotency_key)
        old = self.repository.get("export_job", job_id)
        if old:
            self._job(job_id, actor)
            if old["request_hash"] != request_hash:
                raise ConflictError("IDEMPOTENCY_KEY_REUSED")
            return self.public_job(old)
        # Validate current assets before taking the short snapshot transaction.
        result_set = require(self.repository.get("set", command.result_set_id))
        WindowResultSetService.check_owner(result_set, actor)
        for item in result_set["items"]:
            self.search.media.result(item["result_id"], actor)
        return self._snapshot_request(command, actor, job_id, request_hash)

    @staticmethod
    def public_job(job):
        """Hide private artifact paths and idempotency fields in API responses.

        Args:
            job: Complete persisted export job.

        Returns:
            Copy containing public job fields only.
        """
        return {
            k: v
            for k, v in job.items()
            if k not in {"artifact_key", "marker_key", "idempotency_key", "request_hash", "artifacts"}
        }

    def status(self, job_id, actor):
        """Read public job status after checking requester and tenant ownership.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester.

        Returns:
            Public export job fields.

        Raises:
            ForbiddenError: If the job belongs to another actor or tenant.
        """
        return self.public_job(self._job(job_id, actor))

    def _job(self, job_id, actor):
        """Require an existing export owned by the actor and tenant.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester.

        Returns:
            Complete persisted job, including private artifact metadata.

        Raises:
            ForbiddenError: If requester or tenant ownership differs.
        """
        job = require(self.repository.get("export_job", job_id))
        if job["requester_id"] != actor.actor_id or job["tenant_id"] != actor.tenant_id:
            raise ForbiddenError("export is outside actor scope")
        return job

    def _validate_sources(self, snapshot, actor):
        """Recheck snapshot source availability, registration and version.

        Args:
            snapshot: Immutable export request snapshot.
            actor: Authenticated actor, tenant and current source access scopes.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        for ref in snapshot["source_refs"]:
            asset, _ = self.search.media.asset(ref["asset_id"], actor, verify=False)
            if asset.checksum != ref["checksum"]:
                raise ForbiddenError("SOURCE_VERSION_CHANGED")

    def _artifact_checksum(self, key):
        """Read and hash an artifact, closing its stream after consumption.

        Args:
            key: Object-store key identifying the artifact.

        Returns:
            Hexadecimal SHA-256 checksum of its bytes.
        """
        digest = hashlib.sha256()
        with self.store.open_read(key) as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def run_export(self, job_id, phase="all"):
        """Run durable export phases from the immutable request snapshot.

        Args:
            job_id: Persisted job identifier.
            phase: Export phase: validate, build, publish, or all.

        Returns:
            Public export job state after the requested phase.
        """
        from src.retrieval.application.window_export_runner import run_export

        return run_export(self, job_id, phase)

    def _build_artifacts(self, job_id, snapshot, settings):
        # Manifest-only never reads source video bytes or invokes a model.
        """Build and publish manifest files with verified checksums and stable paths.

        Args:
            job_id: Persisted job identifier.
            snapshot: Immutable export request snapshot.
            settings: Export settings retained in the snapshot.

        Returns:
            Artifact metadata keyed by manifest extension.
        """
        from src.retrieval.application.window_export_runner import _build_artifacts

        return _build_artifacts(self, job_id, snapshot, settings)

    def _downloadable(self, job_id, actor):
        """Require an intact, unexpired export and currently authorized sources.

        Persist the expired status when the artifact lifetime has elapsed.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester and current source scopes.

        Returns:
            Authorized job and its immutable export snapshot.

        Raises:
            ConflictError: If incomplete, expired, revoked or checksum-invalid.
            ForbiddenError: If ownership or source access is no longer valid.
        """
        job = self._job(job_id, actor)
        if job["status"] != "completed" or job["revoked_at"]:
            raise ConflictError("EXPORT_NOT_AVAILABLE")
        if datetime.fromisoformat(job["expires_at"]) <= datetime.now(UTC):
            job["status"] = "expired"
            self.repository.put("export_job", job_id, job)
            raise ConflictError("EXPORT_EXPIRED")
        if not self.store.exists(job["marker_key"]):
            raise ConflictError("EXPORT_INCOMPLETE")
        marker_info = self.store.stat(job["marker_key"])
        if not marker_info or self._artifact_checksum(job["marker_key"]) != job["marker_sha256"]:
            raise ConflictError("EXPORT_INCOMPLETE")
        snapshot = require(self.repository.get("export_snapshot", job["snapshot_id"]))
        self._validate_sources(snapshot, actor)
        for artifact in job["artifacts"].values():
            info = self.store.stat(artifact["key"])
            if not info or self._artifact_checksum(artifact["key"]) != artifact["sha256"]:
                raise ConflictError("EXPORT_ARTIFACT_INVALID")
        return job, snapshot

    def create_download_url(self, job_id, actor):
        """Issue a time-limited download grant for a verified export artifact.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester and current source scopes.

        Returns:
            Signed manifest URL and its lifetime in seconds.

        Raises:
            ConflictError: If the export is unavailable or damaged.
            ForbiddenError: If ownership or source access fails.
        """
        job, snapshot = self._downloadable(job_id, actor)
        extension = snapshot["config"]["manifest_format"]
        signer = SignedCursorService(self.config.cursor_signing_key, self.config.export.download_ttl_seconds)
        token = signer.sign(job_id, snapshot["snapshot_hash"], 0, extension)
        return {
            "url": f"/api/v1/export-media/{job_id}?token={token}",
            "expires_in_seconds": self.config.export.download_ttl_seconds,
        }

    def download(self, job_id, token, actor):
        """Verify the signed grant and open its authorized manifest artifact.

        Args:
            job_id: Export job identifier.
            token: Signed, expiring grant bound to the snapshot and extension.
            actor: Authenticated requester and current source scopes.

        Returns:
            Readable artifact stream and extension; the caller closes the stream.

        Raises:
            ForbiddenError: If the grant, ownership or source access fails.
            ConflictError: If the export is unavailable or damaged.
        """
        payload = self.search.cursor.verify(token)
        job, snapshot = self._downloadable(job_id, actor)
        if (
            payload["run_id"] != job_id
            or payload["index_version"] != snapshot["snapshot_hash"]
            or payload["result_hash"] not in job["artifacts"]
        ):
            raise ForbiddenError("DOWNLOAD_GRANT_INVALID")
        extension = payload["result_hash"]
        return self.store.open_read(job["artifacts"][extension]["key"]), extension

    def revoke(self, job_id, actor):
        """Atomically revoke an owned export and record its audit event.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester.

        Returns:
            Public revoked job fields.

        Raises:
            ForbiddenError: If requester or tenant ownership differs.
        """
        self._job(job_id, actor)
        with self.repository.transaction() as tx:
            job = tx.get("export_job", job_id)
            job.update(status="revoked", revoked_at=utc_now())
            tx.put("export_job", job_id, job)
            tx.put(
                "retrieval_audit",
                source_uuid("revoke", job_id),
                {"action": "export_revoked", "actor_id": actor.actor_id, "job_id": job_id, "at": utc_now()},
            )
        return self.public_job(job)

    def retry(self, job_id, actor):
        """Requeue a failed export while retaining its original snapshot.

        Args:
            job_id: Export job identifier.
            actor: Authenticated requester.

        Returns:
            Public queued job fields after resetting its outbox entry.

        Raises:
            ForbiddenError: If requester or tenant ownership differs.
            ConflictError: If the job is not failed.
        """
        self._job(job_id, actor)
        with self.repository.transaction() as tx:
            job = tx.get("export_job", job_id)
            if job["status"] != "failed":
                raise ConflictError("EXPORT_STATE_CONFLICT")
            job["status"] = "queued"
            tx.put("export_job", job_id, job)
            tx.execute("UPDATE window_export_outbox SET dispatched=0 WHERE job_id=%s", (job_id,))
        return self.public_job(job)

    async def dispatch_outbox(self, client=None):
        """Dispatch pending exports and mark each successfully dispatched entry.

        Args:
            client: Temporal client, or None to run exports locally.

        Treat an already-started Temporal workflow as successfully dispatched.
        Other dispatch failures propagate and leave the entry pending.
        """
        with self.repository.transaction() as tx:
            rows = tx.execute("SELECT job_id,workflow_id FROM window_export_outbox WHERE dispatched=0").fetchall()
        for job_id, workflow_id in rows:
            if client:
                from temporalio.common import WorkflowIDReusePolicy
                from temporalio.exceptions import WorkflowAlreadyStartedError

                from src.indexing.workflows.workflow import ManifestExportWorkflow

                try:
                    await client.start_workflow(
                        ManifestExportWorkflow.run,
                        job_id,
                        id=workflow_id,
                        task_queue=self.config.workflow.task_queue,
                        id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
                    )
                except WorkflowAlreadyStartedError:
                    pass
            else:
                try:
                    import asyncio

                    await asyncio.to_thread(self.run_export, job_id)
                except Exception:
                    # Local failures are durable; retry is an explicit job action.
                    pass
            with self.repository.transaction() as tx:
                tx.execute("UPDATE window_export_outbox SET dispatched=1 WHERE job_id=%s", (job_id,))

    def _snapshot_request(self, command, actor, job_id, request_hash):
        """Check current result-set state and persist one immutable export.

        Args:
            command: Versioned export request with idempotency key and format.
            actor: Authenticated owner and current source scopes.
            job_id: Stable export identifier derived from actor and request key.
            request_hash: Hash detecting a reused key with changed inputs.

        Returns:
            Public fields of the existing or newly queued export job.

        Raises:
            ConflictError: If request identity, set version or index differs.
            RetrievalError: If limits or reviewed-result eligibility fail.
            ForbiddenError: If set ownership or current source access fails.
        """
        with self.repository.transaction() as tx:
            old = tx.get("export_job", job_id)
            if old:
                if old["request_hash"] != request_hash:
                    raise ConflictError("IDEMPOTENCY_KEY_REUSED")
                return self.public_job(old)
            value = require(tx.get("set", command.result_set_id))
            WindowResultSetService.check_owner(value, actor)
            if value["version"] != command.expected_version:
                raise ConflictError("RESULT_SET_CONFLICT")
            if not value["items"] or len(value["items"]) > self.config.export.max_items:
                raise RetrievalError("EXPORT_ITEM_LIMIT")
            rows, refs = self._snapshot_rows(tx, value, job_id, command, actor)
            if sum(r["end_ms"] - r["start_ms"] for r in rows) > self.config.export.max_total_duration_ms:
                raise RetrievalError("EXPORT_DURATION_LIMIT")
            if len({r["index_version"] for r in rows}) != 1:
                raise ConflictError("RESULT_SET_INDEX_CONFLICT")
            body = {
                "items": rows,
                "source_refs": refs,
                "actor": actor.model_dump(mode="json"),
                "result_set_id": value["result_set_id"],
                "result_set_version": value["version"],
                "config": command.config.model_dump(mode="json"),
                "settings": self.config.export.model_dump(mode="json"),
                "policy_version": "source_unmodified_internal_v1",
            }
            snapshot_id = stable_hash(body)
            job = {
                "job_id": job_id,
                "requester_id": actor.actor_id,
                "tenant_id": actor.tenant_id,
                "idempotency_key": command.idempotency_key,
                "request_hash": request_hash,
                "snapshot_id": snapshot_id,
                "status": "queued",
                "workflow_id": "export/" + job_id,
                "created_at": utc_now(),
                "expires_at": None,
                "revoked_at": None,
            }
            tx.put("export_snapshot", snapshot_id, {**body, "snapshot_hash": snapshot_id})
            tx.put("export_job", job_id, job)
            tx.execute(
                "INSERT INTO window_export_outbox(job_id,workflow_id) VALUES(%s,%s)", (job_id, job["workflow_id"])
            )
            tx.put(
                "retrieval_audit",
                source_uuid("export-request", job_id),
                {"action": "export_queued", "actor_id": actor.actor_id, "job_id": job_id, "at": utc_now()},
            )
        return self.public_job(job)

    def _snapshot_rows(self, tx, value, job_id, command, actor):
        """Authorize and copy reviewed results using the caller's transaction.

        Args:
            tx: Existing snapshot transaction; no nested transaction is opened.
            value: Owned, version-checked result set.
            job_id: Stable export job identifier.
            command: Export settings included in each snapshot row.
            actor: Authenticated tenant and current source access scopes.

        Returns:
            Ordered manifest rows and checksum-pinned source references.

        Raises:
            RetrievalError: If a result is missing or lacks a confirmed review.
            ForbiddenError: If a source is unavailable or outside actor scope.
        """
        rows, refs = [], []
        for item in value["items"]:
            result = tx.get("result", item["result_id"])
            if not result or result["review_status"] != "relevant" or not result["review_range"]:
                raise RetrievalError("EXPORT_NOT_ELIGIBLE")
            run = tx.get("run", result["run_id"])
            asset = tx.get("asset", result["asset_id"])
            if (
                not asset
                or not asset["available"]
                or asset["tenant_id"] != actor.tenant_id
                or asset["access_scope_id"] not in actor.access_scope_ids
            ):
                raise ForbiddenError("SOURCE_FORBIDDEN")
            rows.append(self._snapshot_row(tx, job_id, command, result, run, asset))
            refs.append({"asset_id": asset["asset_id"], "checksum": asset["checksum"]})
        return rows, refs

    def _snapshot_row(self, tx, job_id, command, result, run, asset):
        """Map a confirmed result to the existing manifest schema.

        Keep fields explicit so exported provenance and privacy policy remain
        reviewable together; this function performs no writes.

        Args:
            tx: Existing snapshot transaction used to read session and review.
            job_id: Export identifier used to derive a stable item identifier.
            command: Export settings included in the config hash.
            result: Eligible reviewed result.
            run: Original search query, filters and ranking configuration.
            asset: Authorized source asset and checksum.

        Returns:
            Manifest row preserving source, review and model provenance.
        """
        session = tx.get("session", run["session_id"])
        contract = session["index_snapshot"]
        confirmed = result["review_range"]
        return {
            "result_id": result["result_id"],
            "representative_scene_id": result["scene_id"],
            "export_item_id": source_uuid("export-item", job_id, result["result_id"]),
            "member_scene_ids": result["member_scene_ids"],
            "representative_frame_id": result["representative_frame_id"],
            "video_id": result["video_id"],
            "asset_id": result["asset_id"],
            "source_sha256": asset["checksum"],
            "source_asset_version": result["asset_id"],
            "anchor_ms": result["anchor_ms"],
            "start_ms": confirmed["start_ms"],
            "end_ms": confirmed["end_ms"],
            "reviewed_start_ms": confirmed["start_ms"],
            "reviewed_end_ms": confirmed["end_ms"],
            "query_original": run["query_original"],
            "query_normalized": run["query_normalized"],
            "query_effective": run["query_effective"],
            "filters": {k: v for k, v in run["filters"].items() if k not in {"tenant_id", "access_scope_ids"}},
            "review_event_id": result["review_event_id"],
            "review_version": result["version"],
            "reviewer_reference": tx.get("review_event", result["review_event_id"])["actor_id"],
            "review_status": "relevant",
            "subject_version": result["subject_version"],
            "dataset_version": result["dataset_version"],
            "index_version": result["index_version"],
            "embedding_contract_hash": contract["embedding_contract_hash"],
            "embedding_contract_version": contract["embedding_contract"]["contract_version"],
            "ranking_config_version": run["ranking_config_version"],
            "ranking_config_hash": run["ranking_config_hash"],
            "vector_score": result["vector_score"],
            "privacy_status": "not_processed",
            "media_kind": "source_unmodified",
            "media_policy_version": result["media_policy_version"],
            "media_input": "source_unmodified",
            "privacy_policy_version": None,
            "export_config_hash": stable_hash(command.config),
            "clip_file": None,
            "clip_uri": None,
            "clip_sha256": None,
        }
