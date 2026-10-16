CREATE TABLE IF NOT EXISTS index_jobs (
    job_id TEXT PRIMARY KEY,
    index_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    pipeline_config_hash CHAR(64) NOT NULL,
    config_snapshot JSONB NOT NULL,
    status TEXT NOT NULL,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_index_jobs_version ON index_jobs(index_version);

CREATE TABLE IF NOT EXISTS job_items (
    job_id TEXT NOT NULL REFERENCES index_jobs(job_id),
    video_id TEXT NOT NULL,
    attempt INTEGER NOT NULL DEFAULT 0,
    checkpoint JSONB,
    error_code TEXT,
    PRIMARY KEY(job_id, video_id)
);

CREATE TABLE IF NOT EXISTS scenes (
    scene_id CHAR(64) PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES index_jobs(job_id),
    index_version TEXT NOT NULL,
    dataset_version TEXT NOT NULL,
    video_id TEXT NOT NULL,
    start_ms BIGINT NOT NULL,
    end_ms BIGINT NOT NULL,
    privacy_status TEXT NOT NULL,
    preview_uri TEXT,
    preview_checksum CHAR(64),
    value JSONB NOT NULL,
    CHECK (end_ms > start_ms)
);
CREATE INDEX IF NOT EXISTS ix_scenes_index_version ON scenes(index_version);

CREATE TABLE IF NOT EXISTS scene_models (
    scene_id CHAR(64) NOT NULL REFERENCES scenes(scene_id),
    model_role TEXT NOT NULL,
    descriptor JSONB NOT NULL,
    PRIMARY KEY(scene_id, model_role)
);

CREATE TABLE IF NOT EXISTS checkpoints (
    job_id TEXT NOT NULL REFERENCES index_jobs(job_id),
    scene_id CHAR(64) NOT NULL,
    committed_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY(job_id, scene_id)
);

CREATE TABLE IF NOT EXISTS validation_reports (
    index_version TEXT PRIMARY KEY,
    value JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS index_catalog (
    index_version TEXT PRIMARY KEY,
    collection_name TEXT NOT NULL UNIQUE,
    alias_name TEXT,
    embedding_contract JSONB NOT NULL,
    state TEXT NOT NULL CHECK (state IN ('staging', 'published', 'retired')),
    published_at TIMESTAMPTZ
);
