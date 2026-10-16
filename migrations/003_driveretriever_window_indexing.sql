-- Additive schema: legacy scenes and their references remain readable.
CREATE TABLE IF NOT EXISTS window_records (
    kind TEXT NOT NULL, record_id TEXT NOT NULL, value TEXT NOT NULL,
    PRIMARY KEY(kind, record_id)
);
CREATE TABLE IF NOT EXISTS video_assets (
    asset_id UUID PRIMARY KEY, checksum TEXT NOT NULL, value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS window_catalog (
    index_version TEXT PRIMARY KEY, collection_name TEXT NOT NULL UNIQUE,
    contract_hash TEXT NOT NULL, state TEXT NOT NULL CHECK(state IN ('staging','ready','published','retired')),
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS scene_windows (
    index_version TEXT NOT NULL REFERENCES window_catalog(index_version), scene_id UUID NOT NULL,
    asset_id UUID NOT NULL REFERENCES video_assets(asset_id), window_index INTEGER NOT NULL CHECK(window_index >= 0),
    start_ms DOUBLE PRECISION NOT NULL CHECK(start_ms >= 0), end_ms DOUBLE PRECISION NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('staged','checkpointed')), value TEXT NOT NULL,
    PRIMARY KEY(index_version, scene_id), CHECK(end_ms > start_ms),
    UNIQUE(index_version, asset_id, window_index)
);
CREATE TABLE IF NOT EXISTS source_frames (
    frame_id UUID PRIMARY KEY, asset_id UUID NOT NULL REFERENCES video_assets(asset_id),
    pts BIGINT NOT NULL, time_base_num INTEGER NOT NULL CHECK(time_base_num > 0),
    time_base_den INTEGER NOT NULL CHECK(time_base_den > 0), value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS window_samples (
    index_version TEXT NOT NULL, scene_id UUID NOT NULL,
    sample_index INTEGER NOT NULL CHECK(sample_index BETWEEN 0 AND 11),
    frame_id UUID NOT NULL REFERENCES source_frames(frame_id),
    target_ms DOUBLE PRECISION NOT NULL, pts_ms DOUBLE PRECISION NOT NULL,
    PRIMARY KEY(index_version, scene_id, sample_index),
    FOREIGN KEY(index_version, scene_id) REFERENCES scene_windows(index_version, scene_id)
);
CREATE TABLE IF NOT EXISTS window_checkpoints (
    job_id TEXT NOT NULL, index_version TEXT NOT NULL, scene_id UUID NOT NULL,
    stage TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0, error TEXT,
    PRIMARY KEY(job_id, scene_id),
    FOREIGN KEY(index_version, scene_id) REFERENCES scene_windows(index_version, scene_id)
);
CREATE TABLE IF NOT EXISTS active_index (
    scope TEXT PRIMARY KEY, index_version TEXT REFERENCES window_catalog(index_version),
    previous_version TEXT REFERENCES window_catalog(index_version), generation INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS ix_scene_windows_asset ON scene_windows(index_version, asset_id);
