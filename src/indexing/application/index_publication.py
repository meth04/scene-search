from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from src.indexing.domain.windows import source_uuid
from src.indexing.models.driveretriever_adapter import build_driveretriever


def publish(indexer: Any, index_version: str, alias: str = "scenes_production", *, actor_id: str = "cli",
            expected_generation: int | None = None) -> str | None:
    """Publish a validated index with generation checks and smoke-test recovery.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        index_version: Exact persisted index version.
        alias: Active-index alias to update.
        actor_id: Actor identifier recorded in audit events.
        expected_generation: Expected active-index generation, or None to omit that comparison.

    Returns:
        Previously active index version, or None for the first publication.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    from src.indexing.config import DriveRetrieverConfig
    from src.indexing.validation.window_index import metadata_publication_allowed

    target = indexer.jobs.get("catalog", index_version)
    if not target:
        raise ValueError("PUBLISH_BLOCKED: index is missing")
    encoder = build_driveretriever(DriveRetrieverConfig.model_validate(target["encoder_config"]))
    before = indexer.jobs.active(alias)
    indexer._smoke(target, encoder, actor_id)
    with indexer.jobs.transaction() as tx:
        catalog = tx.get("catalog", index_version)
        report = tx.get("validation", index_version)
        if (
            not catalog
            or catalog["state"] not in {"ready", "published", "retired"}
            or not report
            or not report["passed"]
        ):
            raise ValueError("PUBLISH_BLOCKED: index must be validated and READY")
        if (
            report["contract_hash"] != catalog["embedding_contract_hash"]
            or encoder.contract.hash != catalog["embedding_contract_hash"]
        ):
            raise ValueError("EMBEDDING_CONTRACT_MISMATCH")
        if catalog["embedding_contract"]["is_fake"] and indexer.config.storage.provider == "production":
            raise ValueError("PUBLISH_BLOCKED: fixture model cannot publish in production")
        if not metadata_publication_allowed(catalog, production=indexer.config.storage.provider == "production"):
            raise ValueError("PUBLISH_BLOCKED: metadata requires real evaluated inference in production")
        metadata_contract = catalog.get("metadata_contract")
        if report.get("metadata_contract_hash") and not metadata_contract:
            raise ValueError("PUBLISH_BLOCKED: validated metadata contract is missing")
        if metadata_contract and report.get("metadata_contract_hash") != metadata_contract["hash"]:
            raise ValueError("PUBLISH_BLOCKED: metadata validation contract differs")
        row = tx.execute("SELECT index_version,generation FROM active_index WHERE scope=%s", (alias,)).fetchone()
        previous, generation = (row[0], row[1]) if row else (None, 0)
        if expected_generation is not None and expected_generation != generation:
            raise ValueError("ACTIVE_INDEX_CONFLICT")
        if previous == index_version:
            return previous
        tx.execute(
            """INSERT INTO active_index(scope,index_version,previous_version,generation) VALUES(%s,%s,%s,%s)
                    ON CONFLICT(scope) DO UPDATE SET index_version=EXCLUDED.index_version,previous_version=EXCLUDED.previous_version,generation=EXCLUDED.generation""",
            (alias, index_version, previous, generation + 1),
        )
        catalog["state"] = "published"
        indexer.jobs.save_catalog(catalog, tx)
        if previous:
            old = tx.get("catalog", previous)
            old["state"] = "retired"
            indexer.jobs.save_catalog(old, tx)
        tx.put(
            "index_audit",
            source_uuid("publish", alias, generation + 1),
            {
                "action": "publish",
                "version": index_version,
                "previous": previous,
                "actor_id": actor_id,
                "at": datetime.now(UTC).isoformat(),
            },
        )
        for job in tx.list("job"):
            if job["index_version"] == index_version:
                job["status"] = "published"
                tx.put("job", job["job_id"], job)
    try:
        indexer._smoke(target, encoder, actor_id)
    except Exception:
        with indexer.jobs.transaction() as tx:
            row = tx.execute("SELECT index_version,generation FROM active_index WHERE scope=%s", (alias,)).fetchone()
            if row and row[0] == index_version and row[1] == generation + 1:
                tx.execute(
                    "UPDATE active_index SET index_version=%s,previous_version=%s,generation=%s WHERE scope=%s",
                    (previous, before["previous_version"] if before else None, generation + 2, alias),
                )
                target["state"] = "ready"
                indexer.jobs.save_catalog(target, tx)
                if previous:
                    old = tx.get("catalog", previous)
                    old["state"] = "published"
                    indexer.jobs.save_catalog(old, tx)
                tx.put(
                    "index_audit",
                    source_uuid("publish-failed", alias, generation + 2),
                    {
                        "action": "publish_smoke_failed_rolled_back",
                        "version": index_version,
                        "at": datetime.now(UTC).isoformat(),
                    },
                )
        raise
    return previous


def _smoke(indexer, catalog, encoder, actor_id="cli"):
    """Check that the target index returns an authorized source window.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        catalog: Pinned index catalog, capabilities and embedding contract.
        encoder: Existing compatible encoder.
        actor_id: Actor identifier recorded in audit events.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    from src.retrieval.application.window_search_service import WindowSearchService
    from src.retrieval.config import RetrievalConfig
    from src.retrieval.domain.entities import ActorContext
    from src.retrieval.domain.window_filters import WindowFilter

    assets = [indexer.jobs.asset(w.asset_id) for w in indexer.jobs.windows(catalog["index_version"])]
    if not assets:
        raise ValueError("PUBLISH_BLOCKED: no source windows")
    actor = ActorContext(
        actor_id=actor_id,
        tenant_id=assets[0].tenant_id,
        access_scope_ids=sorted({a.access_scope_id for a in assets if a.tenant_id == assets[0].tenant_id}),
        roles=["Engineer"],
    )
    config = RetrievalConfig(
        model_family="driveretriever",
        root_dir=indexer.config.storage.root_dir,
        driveretriever=indexer.config.driveretriever,
    )
    service = WindowSearchService(config, repository=indexer.jobs, vectors=indexer.vectors, encoder=encoder)
    filters = WindowFilter.from_request({}, actor, catalog)
    vector = service._text_vector("a vehicle on a road", actor, catalog)
    hits = indexer.vectors.query(catalog["collection_name"], vector, filters, 5)
    if not service._safe_candidates(hits, filters, actor):
        raise ValueError("PUBLISH_BLOCKED: retrieval smoke did not return an authorized source window")


def rollback(indexer, index_version, alias="scenes_production", *, actor_id="cli", expected_generation=None):
    """Restore an explicit index or deactivate the first published version.

    Args:
        indexer: Coordinator owning the existing config, repository, vector store and model.
        index_version: Exact persisted index version.
        alias: Active-index alias to update.
        actor_id: Actor identifier recorded in audit events.
        expected_generation: Expected active-index generation, or None to omit that comparison.

    Returns:
        Previously active version, or None when no active index exists.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    if index_version is not None:
        return indexer.publish(index_version, alias, actor_id=actor_id, expected_generation=expected_generation)
    with indexer.jobs.transaction() as tx:
        row = tx.execute(
            "SELECT index_version,previous_version,generation FROM active_index WHERE scope=%s", (alias,)
        ).fetchone()
        if not row:
            return None
        current, previous, generation = row
        if expected_generation is not None and expected_generation != generation:
            raise ValueError("ACTIVE_INDEX_CONFLICT")
        if previous:
            raise ValueError("ROLLBACK_TARGET_REQUIRED: supply the previous version")
        tx.execute(
            "UPDATE active_index SET index_version=NULL,previous_version=%s,generation=%s WHERE scope=%s",
            (current, generation + 1, alias),
        )
        catalog = tx.get("catalog", current)
        catalog["state"] = "retired"
        indexer.jobs.save_catalog(catalog, tx)
        tx.put(
            "index_audit",
            source_uuid("deactivate", alias, generation + 1),
            {"action": "rollback_to_no_active", "version": current},
        )
        return current
