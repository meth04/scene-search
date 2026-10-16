-- Immutable snapshots and transactional outbox use window_records (migration 003).
-- Version-qualified FK prevents a UUID shared across indexes from changing identity.
CREATE TABLE IF NOT EXISTS window_result_refs (
    result_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, index_version TEXT NOT NULL,
    scene_id UUID NOT NULL, rank INTEGER NOT NULL CHECK(rank > 0),
    UNIQUE(run_id, rank), UNIQUE(run_id, index_version, scene_id),
    FOREIGN KEY(index_version, scene_id) REFERENCES scene_windows(index_version, scene_id)
);
CREATE TABLE IF NOT EXISTS window_export_outbox (
    job_id TEXT PRIMARY KEY, workflow_id TEXT NOT NULL UNIQUE,
    dispatched INTEGER NOT NULL DEFAULT 0 CHECK(dispatched IN (0,1))
);
