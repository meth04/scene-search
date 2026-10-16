-- Retrieval/review/export state. PostgreSQL remains the authoritative store.
-- Backfill Phase 3 catalog rows to the explicit text/video retrieval contract.
UPDATE index_catalog
SET embedding_contract = embedding_contract || jsonb_build_object(
    'contract_version', '1.1',
    'model_family', 'xclip',
    'requested_revision', embedding_contract #>> '{xclip_descriptor,requested_revision}',
    'resolved_revision', embedding_contract #>> '{xclip_descriptor,resolved_revision}',
    'model_version', embedding_contract #>> '{xclip_descriptor,model_version}',
    'processor_revision', embedding_contract #>> '{xclip_descriptor,resolved_revision}',
    'tokenizer_revision', embedding_contract #>> '{xclip_descriptor,resolved_revision}',
    'text_projection', 'checkpoint-default',
    'embedding_dimension', (embedding_contract->>'dimension')::integer,
    'normalization', 'l2',
    'distance_metric', 'cosine',
    'preprocessing_version', 'rgb-anonymized-v1',
    'query_language_policy', 'direct',
    'query_normalizer_version', 'unicode-whitespace-v1',
    'payload_schema_version', 'scene-payload-v1.1'
)
WHERE NOT embedding_contract ? 'contract_version';

CREATE TABLE IF NOT EXISTS search_sessions (
    session_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    title TEXT NOT NULL DEFAULT '', value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS search_runs (
    run_id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES search_sessions(session_id),
    request_key TEXT NOT NULL, index_version TEXT NOT NULL, collection_name TEXT NOT NULL,
    result_count INTEGER NOT NULL DEFAULT 0, value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), UNIQUE(session_id, request_key)
);

CREATE TABLE IF NOT EXISTS search_results (
    result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES search_runs(run_id),
    scene_id CHAR(64) NOT NULL REFERENCES scenes(scene_id), rank INTEGER NOT NULL CHECK(rank > 0),
    review_status TEXT NOT NULL CHECK(review_status IN ('unreviewed','relevant','not_relevant')),
    version INTEGER NOT NULL CHECK(version > 0), value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(run_id, rank), UNIQUE(run_id, scene_id)
);
CREATE INDEX IF NOT EXISTS ix_search_results_run_rank ON search_results(run_id, rank);

CREATE TABLE IF NOT EXISTS review_events (
    event_id TEXT PRIMARY KEY, result_id TEXT NOT NULL REFERENCES search_results(result_id),
    actor_id TEXT NOT NULL, old_status TEXT NOT NULL, new_status TEXT NOT NULL,
    result_version INTEGER NOT NULL, request_id TEXT NOT NULL, value JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS result_sets (
    result_set_id TEXT PRIMARY KEY, owner_id TEXT NOT NULL, tenant_id TEXT NOT NULL,
    version INTEGER NOT NULL CHECK(version > 0), status TEXT NOT NULL, value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS result_set_items (
    result_set_id TEXT NOT NULL REFERENCES result_sets(result_set_id),
    result_id TEXT NOT NULL REFERENCES search_results(result_id), scene_id CHAR(64) NOT NULL REFERENCES scenes(scene_id),
    added_by TEXT NOT NULL, added_at TIMESTAMPTZ NOT NULL DEFAULT now(), removed_at TIMESTAMPTZ,
    PRIMARY KEY(result_set_id, result_id, added_at)
);
CREATE INDEX IF NOT EXISTS ix_result_set_active_items ON result_set_items(result_set_id, removed_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_result_set_active_membership
ON result_set_items(result_set_id, result_id) WHERE removed_at IS NULL;

CREATE TABLE IF NOT EXISTS export_snapshots (
    snapshot_id TEXT PRIMARY KEY, result_set_id TEXT NOT NULL REFERENCES result_sets(result_set_id),
    result_set_version INTEGER NOT NULL, snapshot_hash CHAR(64) NOT NULL UNIQUE,
    value JSONB NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE TABLE IF NOT EXISTS export_snapshot_items (
    snapshot_id TEXT NOT NULL REFERENCES export_snapshots(snapshot_id), ordinal INTEGER NOT NULL,
    result_id TEXT NOT NULL REFERENCES search_results(result_id), scene_id CHAR(64) NOT NULL REFERENCES scenes(scene_id),
    value JSONB NOT NULL, PRIMARY KEY(snapshot_id, ordinal), UNIQUE(snapshot_id, result_id)
);
CREATE TABLE IF NOT EXISTS export_jobs (
    job_id TEXT PRIMARY KEY, snapshot_id TEXT NOT NULL REFERENCES export_snapshots(snapshot_id),
    requester_id TEXT NOT NULL, status TEXT NOT NULL CHECK(status IN ('queued','validating','running','completed','failed','expired','revoked')),
    idempotency_key TEXT NOT NULL, request_hash CHAR(64) NOT NULL, value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(), updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(requester_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS ix_export_jobs_status_created ON export_jobs(status, created_at);
CREATE TABLE IF NOT EXISTS export_artifacts (
    artifact_id TEXT PRIMARY KEY, job_id TEXT NOT NULL UNIQUE REFERENCES export_jobs(job_id),
    object_key TEXT NOT NULL, sha256 CHAR(64) NOT NULL, value JSONB NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_events (
    event_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL, action TEXT NOT NULL,
    resource_type TEXT NOT NULL, resource_id TEXT NOT NULL, decision TEXT NOT NULL,
    reason_code TEXT NOT NULL, request_id TEXT NOT NULL, value JSONB NOT NULL,
    occurred_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
