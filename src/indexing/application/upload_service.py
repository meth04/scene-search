"""Durable local folder uploads. Receiving files never creates an indexing job."""

from __future__ import annotations

import hashlib
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from uuid import uuid4

from fastapi import HTTPException

from src.indexing.adapters.window_repository import WindowRepository
from src.indexing.application.commands import RunIndexCommand
from src.indexing.application.folder_import import VIDEO_EXTENSIONS
from src.indexing.domain.entities import DatasetManifest, VideoManifestItem, canonical_json, stable_hash
from src.indexing.models.driveretriever_adapter import file_checksum
from src.indexing.video.source_reader import SourceReader


def safe_relative_path(value):
    if not value or "\\" in value or ":" in value or "\x00" in value:
        raise HTTPException(422, "INVALID_RELATIVE_PATH")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise HTTPException(422, "INVALID_RELATIVE_PATH")
    parts = value.split("/")
    if any(p in {"", ".", ".."} or p.endswith((".", " ")) or PureWindowsPath(p).is_reserved() for p in parts):
        raise HTTPException(422, "INVALID_RELATIVE_PATH")
    return PurePosixPath(value)


class UploadService:
    def __init__(self, config):
        self.config = config
        self.repository = WindowRepository(config.storage.root_dir)
        self.root = Path(config.uploads.root_dir).resolve()
        self.reader = SourceReader(config.driveretriever.source_roots)
        if not any(self.root.is_relative_to(p) for p in self.reader.roots):
            raise HTTPException(409, "UPLOAD_ROOT_NOT_ALLOWED")

    def create(self, folder_name, files, actor, key=None):
        if not actor.access_scope_ids:
            raise HTTPException(403, "SOURCE_FORBIDDEN")
        if len(files) > self.config.uploads.max_files:
            raise HTTPException(422, "TOO_MANY_FILES")
        paths, records = set(), []
        for item in files:
            relative = safe_relative_path(item.relative_path)
            if relative.suffix.lower() not in VIDEO_EXTENSIONS:
                raise HTTPException(422, "UNSUPPORTED_VIDEO_FORMAT")
            if item.size_bytes > self.config.uploads.max_file_bytes:
                raise HTTPException(413, "UPLOAD_TOO_LARGE")
            # Case-insensitive uniqueness also protects Windows deployments.
            canonical = str(relative).casefold()
            if canonical in paths:
                raise HTTPException(422, "DUPLICATE_RELATIVE_PATH")
            paths.add(canonical)
            records.append(
                {
                    "file_id": uuid4().hex,
                    "relative_path": str(relative),
                    "size_bytes": item.size_bytes,
                    "status": "pending",
                    "error": None,
                }
            )
        request_id = stable_hash({"actor": actor.model_dump(mode="json"), "key": key}) if key else None
        request_hash = stable_hash({"folder": folder_name, "files": [i.model_dump() for i in files]})
        upload_id = uuid4().hex
        record = {
            "upload_id": upload_id,
            "folder_name": folder_name,
            "actor_id": actor.actor_id,
            "tenant_id": actor.tenant_id,
            "access_scope_id": actor.access_scope_ids[0],
            "status": "receiving",
            "files": records,
            "created_at": datetime.now(UTC).isoformat(),
            "job_id": None,
            "index_version": None,
            "preview_status": "pending",
            "max_file_bytes": self.config.uploads.max_file_bytes,
        }
        with self.repository.transaction() as tx:
            if request_id:
                previous = tx.get("upload_request", request_id)
                if previous:
                    if previous["request_hash"] != request_hash:
                        raise HTTPException(409, "IDEMPOTENCY_KEY_REUSED")
                    return self.public(self._authorize(tx.get("demo_upload", previous["upload_id"]), actor))
                tx.put("upload_request", request_id, {"request_hash": request_hash, "upload_id": upload_id})
            tx.put("demo_upload", upload_id, record)
        return self.public(record)

    @staticmethod
    def _authorize(record, actor):
        if not record:
            raise HTTPException(404, "UPLOAD_NOT_FOUND")
        if (
            record["actor_id"] != actor.actor_id
            or record["tenant_id"] != actor.tenant_id
            or record["access_scope_id"] not in actor.access_scope_ids
        ):
            raise HTTPException(403, "SOURCE_FORBIDDEN")
        if record.get("status") == "deleted":
            raise HTTPException(404, "UPLOAD_NOT_FOUND")
        return record

    def get(self, upload_id, actor):
        return self._authorize(self.repository.get("demo_upload", upload_id), actor)

    def delete(self, upload_id, actor):
        self.get(upload_id, actor)
        target = (self.root / upload_id).resolve()
        if (
            len(upload_id) != 32
            or any(c not in "0123456789abcdef" for c in upload_id)
            or target.parent != self.root
            or target.name != upload_id
        ):
            raise HTTPException(422, "INVALID_RELATIVE_PATH")
        with self.repository.transaction() as tx:
            record = self._authorize(tx.get("demo_upload", upload_id), actor)
            job_id, version = record.get("job_id"), record.get("index_version")
            job = tx.get("job", job_id) if job_id else None
            if job and job["status"] not in {"ready", "published", "failed", "cancelled"}:
                raise HTTPException(409, "DATASET_BUSY")
            inputs = tx.get("job_input", job_id) if job_id else None
            for item in (inputs or {}).get("verified", {}).get("videos", []):
                asset = tx.get("asset", item["asset_id"])
                if asset:
                    asset["available"] = False
                    tx.put("asset", item["asset_id"], asset)
                    tx.execute(
                        "UPDATE video_assets SET value=%s WHERE asset_id=%s", (canonical_json(asset), item["asset_id"])
                    )
            if version:
                catalog = tx.get("catalog", version)
                if catalog:
                    catalog["state"] = "retired"
                    self.repository.save_catalog(catalog, tx)
                tx.execute(
                    "UPDATE active_index SET index_version=NULL,previous_version=NULL,generation=generation+1 "
                    "WHERE index_version=%s",
                    (version,),
                )
                tx.execute("UPDATE active_index SET previous_version=NULL WHERE previous_version=%s", (version,))
            if job:
                job["status"] = "cancelled"
                tx.put("job", job_id, job)
            record.update(status="deleted", deleted_at=datetime.now(UTC).isoformat())
            tx.put("demo_upload", upload_id, record)
        # Retain audit/index references, revoke their source, then remove this upload's files.
        if target.is_dir():
            shutil.rmtree(target)
        return {"status": "success", "message": f"Dataset {upload_id} deleted"}

    @staticmethod
    def public(record):
        return {k: v for k, v in record.items() if k not in {"manifest", "actor_id", "tenant_id", "access_scope_id"}}

    def path(self, upload_id, relative_path):
        safe_relative_path(relative_path)
        base = (self.root / upload_id).resolve()
        target = (base / relative_path).resolve()
        if not base.is_relative_to(self.root) or not target.is_relative_to(base):
            raise HTTPException(422, "INVALID_RELATIVE_PATH")
        return target

    def update_file(self, upload_id, file_id, **changes):
        with self.repository.transaction() as tx:
            record = tx.get("demo_upload", upload_id)
            file = next(f for f in record["files"] if f["file_id"] == file_id)
            file.update(changes)
            tx.put("demo_upload", upload_id, record)
        return file

    @staticmethod
    def readable(path):
        import av

        try:
            with av.open(str(path)) as container:
                if not container.streams.video or next(container.decode(video=0), None) is None:
                    raise ValueError("missing readable video frames")
        except Exception as exc:
            raise HTTPException(422, "INVALID_VIDEO") from exc

    async def receive(self, upload_id, file_id, request, actor):
        import asyncio

        record = self.get(upload_id, actor)
        file = next((f for f in record["files"] if f["file_id"] == file_id), None)
        if not file:
            raise HTTPException(404, "UPLOAD_FILE_NOT_FOUND")
        if file["status"] == "received":
            return file
        if record["status"] != "receiving":
            raise HTTPException(409, "UPLOAD_ALREADY_COMPLETE")
        target = self.path(upload_id, file["relative_path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.parent / (uuid4().hex + ".upload")
        size, digest = 0, hashlib.sha256()
        try:
            with temporary.open("xb") as output:
                async for block in request.stream():
                    size += len(block)
                    if size > self.config.uploads.max_file_bytes:
                        raise HTTPException(413, "UPLOAD_TOO_LARGE")
                    if size > file["size_bytes"]:
                        raise HTTPException(422, "UPLOAD_SIZE_MISMATCH")
                    digest.update(block)
                    output.write(block)
            if size != file["size_bytes"]:
                raise HTTPException(422, "UPLOAD_SIZE_MISMATCH")
            await asyncio.to_thread(self.readable, temporary)
            try:
                os.link(temporary, target)
                checksum = digest.hexdigest()
            except FileExistsError:
                # Recover a committed file after a lost response or server restart.
                if target.stat().st_size != file["size_bytes"]:
                    raise HTTPException(409, "SOURCE_ALREADY_EXISTS") from None
                checksum = await asyncio.to_thread(file_checksum, target)
                await asyncio.to_thread(self.readable, target)
            return self.update_file(upload_id, file_id, status="received", error=None, sha256=checksum)
        except BaseException as exc:
            # A concurrent successful PUT wins over the failed request's status.
            with self.repository.transaction() as tx:
                latest = tx.get("demo_upload", upload_id)
                current = next(f for f in latest["files"] if f["file_id"] == file_id)
                if current["status"] != "received":
                    current.update(status="failed", error=getattr(exc, "detail", "UPLOAD_INTERRUPTED"))
                    tx.put("demo_upload", upload_id, latest)
            raise
        finally:
            temporary.unlink(missing_ok=True)

    def complete(self, upload_id, actor):
        record = self.get(upload_id, actor)
        if record["status"] != "receiving":
            return self.public(record)
        if any(f["status"] != "received" for f in record["files"]):
            raise HTTPException(409, "UPLOAD_INCOMPLETE")
        videos = []
        for file in record["files"]:
            item = VideoManifestItem(
                video_id=file["relative_path"],
                uri=str(self.path(upload_id, file["relative_path"])),
                sha256=file["sha256"],
                metadata={
                    "filename": Path(file["relative_path"]).name,
                    "tenant_id": record["tenant_id"],
                    "access_scope_id": record["access_scope_id"],
                },
            )
            try:
                self.reader.inspect("demo-upload-" + upload_id, item, actor)
            except Exception as exc:
                raise HTTPException(422, {"code": "INVALID_VIDEO", "relative_path": file["relative_path"]}) from exc
            videos.append(item)
        manifest = DatasetManifest(source_id="demo-upload-" + upload_id, dataset_version=upload_id, videos=videos)
        with self.repository.transaction() as tx:
            latest = tx.get("demo_upload", upload_id)
            if latest["status"] == "receiving":
                latest.update(status="uploaded", manifest=manifest.model_dump(mode="json"))
                tx.put("demo_upload", upload_id, latest)
        return self.public(latest)

    def start(self, upload_id, key, actor, orchestrator):
        record = self.get(upload_id, actor)
        if record["status"] == "receiving" or not record.get("manifest"):
            raise HTTPException(409, "UPLOAD_INCOMPLETE")
        if record["job_id"]:
            return orchestrator.jobs.get_job(record["job_id"])
        command = RunIndexCommand(manifest=DatasetManifest.model_validate(record["manifest"]), config=self.config)
        request_id = stable_hash({"actor": actor.actor_id, "demo_start": key})
        try:
            job = orchestrator.prepare(command, actor, request_id=request_id, request_hash=stable_hash(command))
        except ValueError as exc:
            raise HTTPException(409 if str(exc) == "IDEMPOTENCY_KEY_REUSED" else 422, str(exc)) from exc
        with self.repository.transaction() as tx:
            latest = tx.get("demo_upload", upload_id)
            latest.update(status="indexing", job_id=job.job_id, index_version=job.index_version)
            tx.put("demo_upload", upload_id, latest)
        return job
