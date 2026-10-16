from __future__ import annotations

import time
import unicodedata
from typing import Any
from uuid import uuid4

from src.indexing.domain.embedding_contract import DriveRetrieverContract
from src.indexing.domain.entities import stable_hash
from src.indexing.domain.windows import source_uuid
from src.indexing.models.driveretriever_adapter import build_driveretriever, checked_embedding
from src.indexing.telemetry import stage
from src.indexing.video.source_reader import SourceReader
from src.retrieval.application.media_access import MediaAccess
from src.retrieval.application.query_cache import QueryCache
from src.retrieval.application.window_common import require, utc_now
from src.retrieval.domain.errors import (
    ConflictError,
    EncoderNotReadyError,
    ForbiddenError,
    NotFoundError,
    RetrievalError,
    SearchBusyError,
)
from src.retrieval.domain.errors import EmbeddingContractMismatchError as ContractError
from src.retrieval.domain.window_filters import WindowFilter
from src.retrieval.ranking.pagination import SignedCursorService
from src.retrieval.ranking.window_nms import rank_window_candidates


class WindowSearchService:
    """Manage pinned search sessions and authorized, ranked window results."""

    def __init__(self, config, *, repository=None, vectors=None, encoder=None):
        """Bind service dependencies and initialize local coordination state.

        Args:
            config: Validated configuration; saved snapshots retain their original serialized fields.
            repository: Existing window repository, or None to construct it.
            vectors: Existing vector store, or None to construct it.
            encoder: Existing compatible encoder.

        Raises:
            KeyError: If required production backend settings are absent.
        """
        from src.bootstrap import build_retrieval_storage

        self.config = config
        self.repository, self.vectors = build_retrieval_storage(config, repository=repository, vectors=vectors)
        self.encoder = encoder
        self.reader = SourceReader(config.driveretriever.source_roots, config.driveretriever.source_registry_path)
        self.cursor = SignedCursorService(config.cursor_signing_key, config.search.cursor_ttl_seconds)
        self.media = MediaAccess(config, self.repository, self.reader, self.cursor)
        self.query_cache = QueryCache(
            config.query_cache_ttl_seconds, config.query_cache_max_entries, config.max_concurrent_searches
        )
        self.search_slots = self.query_cache.search_slots

    def _catalog(self, index_id, actor=None):
        """Resolve a published or retired index and verify its contract and tenant.

        Args:
            index_id: Published alias or concrete index version.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Pinned index catalog record.

        Raises:
            ConflictError: If versions, request identity or the current state prevent the operation.
            ContractError: If the encoder or embedding does not match the pinned contract.
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        active = self.repository.active(index_id)
        version = active["index_version"] if active else index_id
        catalog = require(self.repository.get("catalog", version), "INDEX_NOT_FOUND")
        if catalog["state"] not in {"published", "retired"}:
            raise ConflictError("INDEX_NOT_READY")
        contract = DriveRetrieverContract.model_validate(catalog["embedding_contract"])
        if contract.hash != catalog["embedding_contract_hash"]:
            raise ContractError("EMBEDDING_CONTRACT_MISMATCH")
        if actor and catalog.get("tenant_ids") and actor.tenant_id not in catalog["tenant_ids"]:
            raise ForbiddenError("INDEX_FORBIDDEN")
        return catalog

    def create_session(self, command, actor):
        """Verify index and encoder compatibility before persisting an owned session.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Public session fields and pinned index capabilities.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        if not actor.access_scope_ids:
            raise ForbiddenError("empty source scope")
        index_id = getattr(command, "index_id", None) or self.config.search.production_alias
        catalog = self._catalog(index_id, actor)
        # Check encoder/contract before persisting an unusable session.
        self._encoder(catalog)
        session = {
            "session_id": str(uuid4()),
            "owner_id": actor.actor_id,
            "tenant_id": actor.tenant_id,
            "title": command.title,
            "index_id": index_id,
            "index_snapshot": catalog,
            "created_at": utc_now(),
        }
        self.repository.put("session", session["session_id"], session)
        return self.public_session(session)

    @staticmethod
    def public_session(session: dict) -> dict:
        """Expose session metadata without leaking the private catalog snapshot.

        Args:
            session: Persisted session record.

        Returns:
            Public session fields, index version, contract hash and capabilities.
        """
        catalog = session["index_snapshot"]
        return {k: v for k, v in session.items() if k != "index_snapshot"} | {
            "index_version": catalog["index_version"],
            "embedding_contract_hash": catalog["embedding_contract_hash"],
            "media_policy_version": catalog["embedding_contract"]["media_policy_version"],
            "capabilities": catalog["capabilities"],
            **({"metadata_filter_values": catalog["metadata_filter_values"]} if "metadata_filter_values" in catalog else {}),
        }

    def session(self, session_id, actor):
        """Load a session and enforce its actor and tenant ownership.

        Args:
            session_id: Persisted session identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Authorized session record.
        """
        value = self._session(session_id, actor)
        return self.public_session(value)

    def _session(self, session_id, actor):
        """Load the full private session after checking actor and tenant ownership.

        Args:
            session_id: Persisted session identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Session record including its pinned index snapshot.
        """
        return self.media.session(session_id, actor)

    def _encoder(self, catalog):
        """Reuse or load the encoder pinned by the session catalog.

        Args:
            catalog: Pinned index catalog, capabilities and embedding contract.

        Returns:
            Encoder whose contract matches the catalog hash.

        Raises:
            ContractError: If the encoder or embedding does not match the pinned contract.
            EncoderNotReadyError: If the pinned encoder cannot be loaded.
        """
        from src.indexing.config import DriveRetrieverConfig

        encoder = self.encoder
        if encoder is None or encoder.contract.hash != catalog["embedding_contract_hash"]:
            saved = catalog.get("encoder_config")
            try:
                encoder = build_driveretriever(
                    DriveRetrieverConfig.model_validate(saved) if saved else self.config.driveretriever
                )
            except ValueError as exc:
                raise ContractError(str(exc)) from exc
            except (ImportError, FileNotFoundError, RuntimeError) as exc:
                raise EncoderNotReadyError("the pinned text encoder is unavailable") from exc
        if encoder.contract.hash != catalog["embedding_contract_hash"]:
            raise ContractError("EMBEDDING_CONTRACT_MISMATCH")
        self.encoder = encoder
        return encoder

    def _asset(self, asset_id, actor, *, verify=False):
        """Delegate source authorization and version verification to media access.

        Args:
            asset_id: Source asset identifier.
            actor: Authenticated actor, tenant and current source access scopes.
            verify: Whether to read source bytes and verify the checksum as well as file metadata.

        Returns:
            Source asset and resolved local path.
        """
        return self.media.asset(asset_id, actor, verify=verify)

    def _result(self, result_id, actor):
        """Delegate result authorization and review-subject validation to media access.

        Args:
            result_id: Persisted search-result identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Authorized result record and representative window.
        """
        return self.media.result(result_id, actor)

    @staticmethod
    def subject_version(window):
        """Hash the representative window identity, interval and media policy.

        Args:
            window: Representative source window.

        Returns:
            Stable review-subject hash.
        """
        return MediaAccess.subject_version(window)

    def _text_vector(self, normalized, actor, catalog):
        """Reuse an actor-scoped cached vector or encode and validate the query.

        Args:
            normalized: Query text after Unicode and whitespace normalization.
            actor: Authenticated actor, tenant and current source access scopes.
            catalog: Pinned index catalog, capabilities and embedding contract.

        Returns:
            Validated query embedding vector.

        Raises:
            ContractError: If the encoder or embedding does not match the pinned contract.
            RetrievalError: If the query, filter or export request violates its supported contract.
        """
        encoder = self._encoder(catalog)
        key = self.query_cache.key(normalized, actor, encoder.contract)
        cached = self.query_cache.get(key)
        if cached is not None:
            return cached
        try:
            embedding = encoder.encode_text([normalized])[0]
        except ValueError as exc:
            if str(exc).startswith("QUERY_TOO_LONG"):
                raise RetrievalError(str(exc), details={"error_code": "QUERY_TOO_LONG"}) from exc
            raise ContractError(str(exc)) from exc
        checked_embedding(embedding.vector)
        self.query_cache.put(key, embedding.vector)
        return embedding.vector

    def execute(self, command, actor):
        """Enforce search capacity and release the slot on success or failure.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Search run, result page and optional signed cursor.

        Raises:
            SearchBusyError: If every configured search slot is occupied.
        """
        if not self.search_slots.acquire(blocking=False):
            raise SearchBusyError("search capacity is currently exhausted")
        try:
            return self._execute(command, actor)
        finally:
            self.search_slots.release()

    def _execute(self, command: Any, actor: Any) -> dict:
        """Create an idempotent search run and persist authorized ranked results.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Search run, result page and optional signed cursor.

        Raises:
            RetrievalError: If the query, filter or export request violates its supported contract.
            ConflictError: If versions, request identity or the current state prevent the operation.
        """
        if command.page_size > self.config.search.page_size_max:
            raise RetrievalError("PAGE_SIZE_INVALID")
        session = self._session(command.session_id, actor)
        catalog = session["index_snapshot"]
        if command.index_id not in (session["index_id"], catalog["index_version"]):
            raise ConflictError("SESSION_INDEX_CONFLICT")
        filters = WindowFilter.from_request(command.filters, actor, catalog)
        normalized = " ".join(unicodedata.normalize("NFC", command.query_text).split())
        if not normalized or len(normalized) > 1000:
            raise RetrievalError("QUERY_INVALID")
        request_hash = stable_hash(
            {"query": command.query_text, "filters": command.filters, "index": catalog["index_version"]}
        )
        run_id = source_uuid("run", command.session_id, command.request_key)
        with self.repository.transaction() as tx:
            old = tx.get("run", run_id)
            if old and old["request_hash"] != request_hash:
                raise ConflictError("IDEMPOTENCY_KEY_REUSED")
            if not old:
                run = {
                    "run_id": run_id,
                    "session_id": command.session_id,
                    "request_key": command.request_key,
                    "request_hash": request_hash,
                    "status": "running",
                    "query_original": command.query_text,
                    "query_normalized": normalized,
                    "query_effective": normalized,
                    "filters": filters.model_dump(mode="json"),
                    "filters_hash": stable_hash(filters),
                    "index_version": catalog["index_version"],
                    "collection_name": catalog["collection_name"],
                    "embedding_contract_hash": catalog["embedding_contract_hash"],
                    "embedding_contract_version": "driveretriever_window_epoch4_source_v1",
                    "ranking_config_version": "window-nms-v1",
                    "ranking_config_hash": stable_hash({"iou": self.config.nms_iou, "limit": self.config.result_limit}),
                    "candidate_config_hash": stable_hash(
                        {"initial": self.config.initial_candidates, "max": self.config.maximum_candidates}
                    ),
                    "created_at": utc_now(),
                    "result_count": 0,
                    "result_ids": [],
                    "trace_id": str(uuid4()),
                }
                if catalog.get("metadata_contract"):
                    metadata_contract = catalog["metadata_contract"]
                    run.update(metadata_contract=metadata_contract,
                               metadata_schema_version=metadata_contract["metadata_schema_version"],
                               aggregation_version=metadata_contract["aggregation_version"])
                tx.put("run", run_id, run)
        if old:
            return self.page(run_id, None, actor, command.page_size)
        timings = {}
        budget_start = time.monotonic()
        try:
            started = time.monotonic()
            with stage("text_encode"):
                vector = self._text_vector(normalized, actor, catalog)
            timings["encode_ms"] = (time.monotonic() - started) * 1000
            limit = self.config.initial_candidates
            started = time.monotonic()
            with stage("search_query"):
                hits = self.vectors.query(catalog["collection_name"], vector, filters, limit)
            timings["query_ms"] = (time.monotonic() - started) * 1000
            started = time.monotonic()
            safe = self._safe_candidates(hits, filters, actor)
            groups = rank_window_candidates(safe, self.config.nms_iou, self.config.result_limit)
            # Expansion happens once inside this run, never when reading the next page.
            if (
                len(groups) < command.page_size
                and len(hits) == limit
                and limit < self.config.maximum_candidates
                and time.monotonic() - budget_start < self.config.candidate_budget_seconds
            ):
                limit = self.config.maximum_candidates
                hits = self.vectors.query(catalog["collection_name"], vector, filters, limit)
                safe = self._safe_candidates(hits, filters, actor)
                groups = rank_window_candidates(safe, self.config.nms_iou, self.config.result_limit)
            results = [self._make_result(run_id, i + 1, group) for i, group in enumerate(groups)]
            timings["hydrate_rank_ms"] = (time.monotonic() - started) * 1000
            run.update(
                status="completed",
                result_count=len(results),
                result_ids=[r["result_id"] for r in results],
                result_list_hash=stable_hash([r["result_id"] for r in results]),
                candidate_count=len(hits),
                candidate_budget_exhausted=len(hits) == self.config.maximum_candidates,
                timings=timings,
            )
            with self.repository.transaction() as tx:
                for result in results:
                    tx.put("result", result["result_id"], result)
                    tx.execute(
                        "INSERT INTO window_result_refs(result_id,run_id,index_version,scene_id,rank) VALUES(%s,%s,%s,%s,%s)",
                        (result["result_id"], run_id, result["index_version"], result["scene_id"], result["rank"]),
                    )
                tx.put("run", run_id, run)
                tx.put(
                    "retrieval_audit",
                    run_id,
                    {"action": "search_completed", "actor_id": actor.actor_id, "run_id": run_id, "at": utc_now()},
                )
        except Exception as exc:
            run.update(status="failed", error_code=type(exc).__name__)
            self.repository.put("run", run_id, run)
            raise
        return self.page(run_id, None, actor, command.page_size)

    def _safe_candidates(self, hits, filters, actor):
        """Reject unauthorized, stale or mismatched vector candidates.

        Args:
            hits: Vector-search candidate records.
            filters: Validated tenant, scope and metadata filter.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Candidate records hydrated with verified source windows.
        """
        safe, verified = [], set()
        for hit in hits:
            if not filters.matches(hit["payload"]):
                continue
            window = self.repository.get_window(filters.index_version, hit["id"])
            if (
                not window
                or window.status != "checkpointed"
                or window.embedding_contract_hash != filters.embedding_contract_hash
            ):
                continue
            try:
                asset, _ = self._asset(window.asset_id, actor, verify=window.asset_id not in verified)
                verified.add(asset.asset_id)
            except (ForbiddenError, NotFoundError):
                continue
            if hit["payload"] != window.payload(asset):
                continue
            safe.append({**hit, "window": window})
        return safe

    def _make_result(self, run_id: str, rank: int, group: dict) -> dict:
        """Materialize a ranked group using its representative frame and interval.

        Args:
            run_id: Persisted search-run identifier.
            rank: One-based rank within the result list.
            group: Ranked candidates and their representative.

        Returns:
            Persistable result record with review identity and media policy.
        """
        rep = group["representative"]
        window = rep["window"]
        members = sorted(m["window"].scene_id for m in group["members"])
        result_id = source_uuid("result", run_id, stable_hash(members))
        anchor = (window.start_ms + window.end_ms) / 2
        sample = min(window.samples, key=lambda s: (abs(s.frame.timestamp_ms - anchor), s.sample_index))
        result = {
            "result_id": result_id,
            "run_id": run_id,
            "scene_id": window.scene_id,
            "index_version": window.index_version,
            "dataset_version": window.dataset_version,
            "asset_id": window.asset_id,
            "video_id": window.video_id,
            "camera": window.camera,
            "member_scene_ids": members,
            "members": [
                {
                    "scene_id": m["window"].scene_id,
                    "score": m["score"],
                    "start_ms": m["window"].start_ms,
                    "end_ms": m["window"].end_ms,
                }
                for m in group["members"]
            ],
            "rank": rank,
            "vector_score": rep["score"],
            "final_score": rep["score"],
            "start_ms": window.start_ms,
            "end_ms": window.end_ms,
            "partial": window.partial,
            "anchor_ms": anchor,
            "representative_frame_id": sample.frame.frame_id,
            "thumbnail_url": None,
            "privacy_status": "not_processed",
            "media_kind": "source_unmodified",
            "media_policy_version": window.media_policy_version,
            "subject_version": self.subject_version(window),
            "base_subject_version": self.subject_version(window),
            "review_status": "unreviewed",
            "review_note": None,
            "review_range": None,
            "version": 1,
            "created_at": utc_now(),
        }
        if window.metadata is not None:
            result.update(metadata=window.metadata.model_dump(mode="json"),
                          metadata_schema_version=window.metadata_schema_version,
                          aggregation_version=window.aggregation_version)
        return result

    def page(self, run_id, cursor, actor, limit=20):
        """Page a pinned result list while rechecking current access to each result.

        Args:
            run_id: Persisted search-run identifier.
            cursor: Signed pagination cursor, or None for the initial page.
            actor: Authenticated actor, tenant and current source access scopes.
            limit: Maximum number of result positions scanned for this page.

        Returns:
            Run, visible results and a cursor over scanned positions.

        Raises:
            RetrievalError: If the query, filter or export request violates its supported contract.
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        if not 1 <= limit <= min(100, self.config.search.page_size_max):
            raise RetrievalError("PAGE_SIZE_INVALID")
        run = require(self.repository.get("run", run_id))
        self._session(run["session_id"], actor)
        if run["status"] != "completed":
            return {"run": run, "results": [], "next_cursor": None}
        position = 0
        if cursor:
            payload = self.cursor.verify(cursor)
            if (
                payload["run_id"] != run_id
                or payload["index_version"] != run["index_version"]
                or payload["result_hash"] != run["result_list_hash"]
            ):
                raise ForbiddenError("CURSOR_INVALID")
            position = int(payload["last_rank"])
            if not 0 <= position <= len(run["result_ids"]):
                raise ForbiddenError("CURSOR_INVALID")
        results = []
        # The cursor follows scanned positions, even if every item has been revoked.
        stop = min(position + limit, len(run["result_ids"]))
        for result_id in run["result_ids"][position:stop]:
            try:
                result, _ = self._result(result_id, actor)
                results.append(result)
            except (ForbiddenError, NotFoundError, ConflictError):
                pass
        next_cursor = (
            self.cursor.sign(run_id, run["index_version"], stop, run["result_list_hash"])
            if stop < len(run["result_ids"])
            else None
        )
        return {"run": run, "results": results, "next_cursor": next_cursor}

    def create_preview_url(self, result_id, actor):
        """Authorize a result and persist a signed, time-limited media grant.

        Args:
            result_id: Persisted search-result identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Preview URL, interval, anchor, privacy status and grant lifetime.
        """
        return self.media.create_preview_url(result_id, actor)

    def media_path(self, grant_id, token, actor):
        """Verify a media grant and recheck live source access before serving bytes.

        Args:
            grant_id: Persisted signed-media grant identifier.
            token: Signed, expiring grant or cursor token.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Authorized source or prepared browser-preview path.
        """
        return self.media.media_path(grant_id, token, actor)

    def media_type(self, grant_id):
        """Resolve the stored preview type or infer the source container media type.

        Args:
            grant_id: Persisted signed-media grant identifier.

        Returns:
            HTTP media type string.
        """
        return self.media.media_type(grant_id)
