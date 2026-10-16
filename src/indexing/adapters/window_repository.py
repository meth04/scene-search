"""Transactional local SQLite/PostgreSQL persistence for the new window flow.

JSON records retain complete contracts/snapshots. Normalized frame/window tables
enforce source and version references. All multi-record writes share a transaction.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from src.indexing.domain.entities import IndexJob, canonical_json
from src.indexing.domain.windows import VideoAsset, WindowResult


class WindowTransaction:
    def __init__(self, connection, postgres):
        self.connection, self.postgres = connection, postgres

    def execute(self, sql, params=()):
        return self.connection.execute(sql if self.postgres else sql.replace("%s", "?"), params)

    def get(self, kind, key):
        row = self.execute("SELECT value FROM window_records WHERE kind=%s AND record_id=%s", (kind, key)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, kind, key, value):
        self.execute("""INSERT INTO window_records(kind,record_id,value) VALUES(%s,%s,%s)
                     ON CONFLICT(kind,record_id) DO UPDATE SET value=EXCLUDED.value""",
                     (kind, key, canonical_json(value)))

    def list(self, kind):
        return [json.loads(r[0]) for r in self.execute("SELECT value FROM window_records WHERE kind=%s ORDER BY record_id", (kind,)).fetchall()]


class WindowRepository:
    def __init__(self, root, database_url=None):
        self.root = Path(root).resolve()
        self.database_url = database_url
        if not database_url:
            self.root.mkdir(parents=True, exist_ok=True)
            db_file = self.root / "windows.sqlite3"
            init_schema = not db_file.exists() or db_file.stat().st_size == 0
            conn = sqlite3.connect(db_file, timeout=30)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                if init_schema:
                    for migration in ("003_driveretriever_window_indexing.sql", "004_driveretriever_retrieval.sql"):
                        conn.executescript((Path(__file__).resolve().parents[3] / "migrations" / migration).read_text(encoding="utf-8"))
            finally:
                conn.close()

    @contextmanager
    def transaction(self):
        if self.database_url:
            import psycopg
            connection = psycopg.connect(self.database_url)
        else:
            connection = sqlite3.connect(self.root / "windows.sqlite3", timeout=30)
            connection.execute("PRAGMA foreign_keys=ON")
        try:
            with connection:
                if self.database_url:
                    # Serialize compare-and-swap writes across API/worker processes.
                    connection.execute("SELECT pg_advisory_xact_lock(6719951)")
                else:
                    connection.execute("BEGIN IMMEDIATE")
                yield WindowTransaction(connection, bool(self.database_url))
        finally:
            connection.close()

    def get(self, kind, key):
        with self.transaction() as tx:
            return tx.get(kind, key)

    def put(self, kind, key, value):
        with self.transaction() as tx:
            tx.put(kind, key, value)

    def list(self, kind):
        with self.transaction() as tx:
            return tx.list(kind)

    def get_job(self, job_id):
        raw = self.get("job", job_id)
        return IndexJob.model_validate(raw) if raw else None

    def asset(self, asset_id):
        raw = self.get("asset", asset_id)
        return VideoAsset.model_validate(raw) if raw else None

    def save_asset(self, asset, tx=None):
        if tx is None:
            with self.transaction() as transaction:
                return self.save_asset(asset, transaction)
        previous = tx.get("asset", asset.asset_id)
        if previous and previous != asset.model_dump(mode="json"):
            raise ValueError("immutable source asset collision")
        tx.execute("INSERT INTO video_assets(asset_id,checksum,value) VALUES(%s,%s,%s) ON CONFLICT(asset_id) DO NOTHING",
                   (asset.asset_id, asset.checksum, asset.model_dump_json()))
        tx.put("asset", asset.asset_id, asset)

    def get_window(self, version, scene_id):
        with self.transaction() as tx:
            row = tx.execute("SELECT value FROM scene_windows WHERE index_version=%s AND scene_id=%s", (version, scene_id)).fetchone()
        return WindowResult.model_validate_json(row[0]) if row else None

    def windows(self, version):
        with self.transaction() as tx:
            rows = tx.execute("SELECT value FROM scene_windows WHERE index_version=%s ORDER BY asset_id,window_index", (version,)).fetchall()
        return [WindowResult.model_validate_json(r[0]) for r in rows]

    def save_window(self, job_id, window, *, complete=False):
        with self.transaction() as tx:
            catalog = tx.get("catalog", window.index_version)
            if not catalog or catalog["state"] != "staging":
                raise ValueError("INDEX_FROZEN: window writes require staging catalog")
            tx.execute("""INSERT INTO scene_windows(index_version,scene_id,asset_id,window_index,start_ms,end_ms,status,value)
                        VALUES(%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(index_version,scene_id)
                        DO UPDATE SET status=EXCLUDED.status,value=EXCLUDED.value""",
                       (window.index_version, window.scene_id, window.asset_id, window.window_index,
                        window.start_ms, window.end_ms, window.status, window.model_dump_json()))
            for sample in window.samples:
                f = sample.frame
                tx.execute("""INSERT INTO source_frames(frame_id,asset_id,pts,time_base_num,time_base_den,value)
                            VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(frame_id) DO NOTHING""",
                           (f.frame_id, f.asset_id, f.pts, f.time_base_num, f.time_base_den, f.model_dump_json()))
                tx.execute("""INSERT INTO window_samples(index_version,scene_id,sample_index,frame_id,target_ms,pts_ms)
                            VALUES(%s,%s,%s,%s,%s,%s) ON CONFLICT(index_version,scene_id,sample_index)
                            DO UPDATE SET frame_id=EXCLUDED.frame_id,target_ms=EXCLUDED.target_ms,pts_ms=EXCLUDED.pts_ms""",
                           (window.index_version, window.scene_id, sample.sample_index, f.frame_id, sample.target_ms, f.timestamp_ms))
            tx.execute("""INSERT INTO window_checkpoints(job_id,index_version,scene_id,stage) VALUES(%s,%s,%s,%s)
                        ON CONFLICT(job_id,scene_id) DO UPDATE SET stage=EXCLUDED.stage""",
                       (job_id, window.index_version, window.scene_id, "checkpointed" if complete else "staged"))

    def save_catalog(self, catalog, tx=None):
        if tx is None:
            with self.transaction() as transaction:
                return self.save_catalog(catalog, transaction)
        tx.execute("""INSERT INTO window_catalog(index_version,collection_name,contract_hash,state,value) VALUES(%s,%s,%s,%s,%s)
                    ON CONFLICT(index_version) DO UPDATE SET state=EXCLUDED.state,value=EXCLUDED.value""",
                   (catalog["index_version"], catalog["collection_name"], catalog["embedding_contract_hash"], catalog["state"], canonical_json(catalog)))
        tx.put("catalog", catalog["index_version"], catalog)

    def active(self, scope):
        with self.transaction() as tx:
            row = tx.execute("SELECT index_version,previous_version,generation FROM active_index WHERE scope=%s", (scope,)).fetchone()
            return {"index_version": row[0], "previous_version": row[1], "generation": row[2]} if row else None
