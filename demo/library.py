"""Canonical local video library: immutable EG2/V6 indexes and private media."""

from __future__ import annotations

import csv
import io
import json
import logging
import re
import shutil
import subprocess
import threading
import time
from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response

from demo.product_index import Integrity, build, load_index
from demo.product_privacy import configuration
from demo.product_search import ProductSearchEngine
from demo.product_worker import ProductWorker
from hybrid_v2.embeddinggemma2 import ROOT, identity, worker_python_path
from hybrid_v2.embeddinggemma2_index import FROZEN_CONFIG
from src.api.dependencies import get_actor
from src.api.upload_routes import UploadInput
from src.hybrid import demo_previews
from src.indexing.application.upload_service import UploadService
from src.indexing.config import DriveRetrieverConfig, PipelineConfig, StorageConfig, UploadConfig
from src.privacy.render import ffmpeg_binary

LIBRARY_ROOT = ROOT / "outputs/product/library"
logger = logging.getLogger(__name__)


class VideoLibrary:
    def __init__(self, service, root=LIBRARY_ROOT, worker_factory=ProductWorker):
        self.service, self.root, self.worker_factory = service, Path(root).resolve(), worker_factory
        self.config = json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["config"]
        self.uploads = UploadService(
            PipelineConfig(
                driveretriever=DriveRetrieverConfig(source_roots=[str(self.root / "videos")]),
                storage=StorageConfig(root_dir=str(self.root / "records")),
                uploads=UploadConfig(root_dir=str(self.root / "videos"), max_file_bytes=200 * 1024**2, max_files=20),
            )
        )
        self.worker, self.warmed, self.engines = None, False, {}
        self.integrity, self._assets = Integrity(), None
        self.search_slots = threading.BoundedSemaphore(4)
        self.runtime_error, self.startup_seconds = None, None
        for record in self.uploads.repository.list("demo_upload"):
            if record["status"] == "indexing":
                self.update(record["upload_id"], status="failed", stage="interrupted", error="PROCESSING_INTERRUPTED")
            elif record["status"] == "deleted":
                # Recover physical cleanup after a prior process interruption.
                uid = record["upload_id"]
                target = (self.uploads.root / uid).resolve()
                if (
                    re.fullmatch(r"[0-9a-f]{32}", uid)
                    and target.parent == self.uploads.root.resolve()
                    and target.exists()
                ):
                    try:
                        shutil.rmtree(target)
                    except OSError:
                        logger.warning("Deleted upload cleanup pending: %s", uid)

    def update(self, upload_id, **changes):
        with self.uploads.repository.transaction() as tx:
            record = tx.get("demo_upload", upload_id)
            record.update(changes)
            tx.put("demo_upload", upload_id, record)
        return self.uploads.public(record)

    def authorize(self, actor):
        self.service.authorize(actor, engineer=True)

    def catalog(self, actor):
        self.authorize(actor)
        return [
            self.uploads.public(r)
            for r in sorted(self.uploads.repository.list("demo_upload"), key=lambda r: r["created_at"], reverse=True)
            if r["status"] != "deleted"
            and r["actor_id"] == actor.actor_id
            and r["tenant_id"] == actor.tenant_id
            and r["access_scope_id"] in actor.access_scope_ids
        ]

    def complete(self, upload_id, actor):
        record = self.uploads.get(upload_id, actor)
        if record["status"] != "receiving":
            return self.uploads.public(record)
        if not all(f["status"] == "received" for f in record["files"]):
            raise HTTPException(409, "UPLOAD_INCOMPLETE")
        return self.update(upload_id, status="uploaded", progress=0, error=None)

    def start(self, upload_id, actor, tasks):
        self.authorize(actor)
        with self.service.lock:
            record = self.uploads.get(upload_id, actor)
            if record["status"] in {"indexing", "ready"}:
                return self.uploads.public(record)
            if record["status"] not in {"uploaded", "failed"}:
                raise HTTPException(409, "UPLOAD_INCOMPLETE")
            if not ffmpeg_binary():
                raise HTTPException(503, "VIDEO_ENCODER_UNAVAILABLE")
            if not self.service.compute.acquire(blocking=False):
                raise HTTPException(429, "HYBRID_BUSY")
            try:
                result = self.update(upload_id, status="indexing", stage="prepare", progress=0, error=None)
                tasks.add_task(self.run, upload_id, actor)
                return result
            except BaseException:
                self.service.compute.release()
                raise

    def asset_identity(self):
        from demo.product_doctor import verify_assets

        if self._assets is None:
            self._assets = verify_assets()
        return self._assets

    @staticmethod
    def privacy_identity():
        return identity(asdict(configuration()))

    @staticmethod
    def ffmpeg():
        return ffmpeg_binary()

    @staticmethod
    def convert(command):
        subprocess.run(
            command,
            check=True,
            capture_output=True,
            timeout=180,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )

    def offline(self, plan):
        rows = json.loads(plan.read_text(encoding="utf-8"))["rows"]
        upload_id = plan.parent.parent.name
        with (plan.parent / "offline.log").open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [str(worker_python_path()), "-B", "-m", "demo.product_offline", str(plan)],
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=log,
                text=True,
                encoding="utf-8",
                errors="replace",
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            # A watchdog bounds a stalled offline phase without blocking status
            # requests. GPU ownership is released by run's finally clause.
            watchdog = threading.Timer(7200, process.kill)
            watchdog.start()
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    try:
                        event = json.loads(line)
                        if event.get("stage") in {"offline_complete", "offline_reused"}:
                            self.update(
                                upload_id,
                                progress=0.35 * (event["row"] + 1) / len(rows),
                                offline_completed=event["row"] + 1,
                            )
                    except (ValueError, TypeError, KeyError):
                        pass
                if process.wait() != 0:
                    raise ValueError("OFFLINE_PROCESSING_FAILED")
            finally:
                watchdog.cancel()
                if process.poll() is None:
                    process.kill()
                    process.wait()
                process.stdout.close()

    def encoder(self):
        if self.worker is None:
            self.asset_identity()
            worker = self.worker_factory(self.config, timeout=600)
            try:
                worker.load()
            except BaseException:
                worker.close()
                raise
            self.worker = worker
        return self.worker

    def warmup(self):
        started = time.perf_counter()
        try:
            self.asset_identity()
            ProductSearchEngine._validate_vector(self.encoder().encode_text("a road scene with vehicles")[0])
            for record in self.uploads.repository.list("demo_upload"):
                if record["status"] == "ready":
                    try:
                        load_index(
                            self.uploads.path(record["upload_id"], "_product/published"),
                            integrity=self.integrity,
                            expected_meta=record.get("index_meta_sha256"),
                        )
                    except (OSError, ValueError, KeyError, TypeError):
                        self.update(record["upload_id"], status="failed", stage="integrity", error="INDEX_CORRUPT")
            self.warmed, self.runtime_error = True, None
        except Exception as exc:
            self.runtime_error = getattr(exc, "code", "LOCAL_RUNTIME_UNAVAILABLE")
            self.close()
        self.startup_seconds = time.perf_counter() - started

    def close(self):
        if self.worker is not None:
            self.worker.close()
        self.worker, self.warmed = None, False
        for engine in self.engines.values():
            engine.worker = None

    def prepare_builtin(self):
        if not self.service.compute.acquire(blocking=False):
            raise HTTPException(429, "HYBRID_BUSY")
        try:
            self.close()
        finally:
            self.service.compute.release()

    def run(self, upload_id, actor):
        started = time.perf_counter()
        try:
            self.update(upload_id, **build(self, upload_id, actor), index_version=upload_id)
            self.warmup()
        except Exception as exc:
            logger.exception("Product processing failed for %s", upload_id)
            code = getattr(exc, "code", None) or (
                str(exc) if str(exc).isupper() and len(str(exc)) < 80 else "PROCESSING_FAILED"
            )
            self.update(upload_id, status="failed", stage="failed", error=code)
            self.close()
        finally:
            self.root.mkdir(parents=True, exist_ok=True)
            with (self.root / "processing_metrics.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"upload_id": upload_id, "wall_seconds": time.perf_counter() - started}) + "\n")
            self.service.compute.release()

    def snapshot(self, upload_id, actor):
        record = self.uploads.get(upload_id, actor)
        if record["status"] != "ready" or record.get("product_schema") != "driveevidence-v6-product-v1":
            raise HTTPException(409, "INDEX_NOT_READY")
        folder = self.uploads.path(upload_id, "_product/published")
        try:
            loaded = load_index(folder, integrity=self.integrity, expected_meta=record["index_meta_sha256"])
            return record, folder, loaded["rows"]
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "INDEX_CORRUPT") from None

    def search(self, upload_id, query, top_k, actor, *, mode="eg2_only", filters=None):
        from demo.backend import SearchFilters

        if not query.strip():
            raise HTTPException(422, "INVALID_QUERY")
        if not self.search_slots.acquire(blocking=False):
            raise HTTPException(429, "HYBRID_BUSY")
        if not self.service.compute.acquire(timeout=1.0 if self.warmed else 0):
            self.search_slots.release()
            raise HTTPException(429, "HYBRID_BUSY")
        started = time.perf_counter()
        try:
            record, folder, rows = self.snapshot(upload_id, actor)
            loaded = load_index(folder, integrity=self.integrity, expected_meta=record["index_meta_sha256"])
            engine = self.engines.setdefault(upload_id, ProductSearchEngine(folder, self))
            payload = engine.filtered_search(loaded, query, mode, filters or SearchFilters())
            hits = [
                {
                    **hit,
                    "global_score": hit["score"],
                    "mode": mode,
                    "source": "uploaded",
                    "metadata": {
                        k: rows[hit["row"]][k]
                        for k in ("filename", "source_id", "source_start_sec", "source_end_sec", "privacy")
                    },
                }
                for hit in payload["candidates"][:top_k]
            ]
            payload.pop("candidates")
            payload["retrieval"]["end_to_end_seconds"] = time.perf_counter() - started
            return {"index_id": upload_id, "results": hits, **payload}
        except HTTPException:
            raise
        except (ValueError, RuntimeError, OSError):
            logger.exception("Product search failed")
            raise HTTPException(503, "SEARCH_UNAVAILABLE") from None
        finally:
            self.service.compute.release()
            self.search_slots.release()

    def media(self, upload_id, row, actor, asset="mp4"):
        _, folder, rows = self.snapshot(upload_id, actor)
        if row < 0 or row >= len(rows):
            raise HTTPException(404, "PREVIEW_UNAVAILABLE")
        return folder / "media" / str(row) / ("safe.mp4" if asset == "mp4" else "poster.jpg"), rows[row]

    def readiness(self, upload_id, actor):
        try:
            _, folder, _ = self.snapshot(upload_id, actor)
            loaded = load_index(folder, integrity=self.integrity)
            degraded = bool(loaded["evidence_fallback"])
            return {
                "ready": self.warmed and not self.service.compute.locked(),
                "service_state": "DEGRADED_EG2_ONLY" if degraded else "READY_EG2_ONLY",
                "eg2_model_loaded": self.worker is not None,
                "warmup_complete": self.warmed,
                "global_index_loaded": True,
                "v6_index_loaded": not degraded,
                "source_hashes_valid": True,
                "privacy_ready": True,
                "schema_valid": True,
                "policy_hash_valid": True,
                "representation_hash_valid": True,
                "fallback_reason": loaded["evidence_fallback"],
                "index_version": loaded["meta"]["publication_identity"],
            }
        except HTTPException as exc:
            if exc.status_code in (403, 404):
                raise
            return {
                "ready": False,
                "service_state": "NOT_READY",
                "privacy_ready": False,
                "source_hashes_valid": False,
                "error": exc.detail,
            }

    def delete(self, upload_id, actor):
        with self.service.lock:
            if self.uploads.get(upload_id, actor)["status"] == "indexing":
                raise HTTPException(409, "DATASET_BUSY")
            if not self.service.compute.acquire(blocking=False):
                raise HTTPException(429, "HYBRID_BUSY")
            try:
                engine = self.engines.pop(upload_id, None)
                if engine:
                    engine.close()
                self.integrity.drop(self.uploads.root / upload_id)
                # Windows cannot unlink the last decoded array still mmapped by
                # the frozen worker. Stop that child before deleting its upload.
                self.close()
                try:
                    result = self.uploads.delete(upload_id, actor)
                except OSError:
                    # UploadService commits revocation first. A playback handle
                    # may delay physical cleanup; startup retries only this owned
                    # UUID directory. The deleted item is never accessible.
                    result = {"upload_id": upload_id, "status": "deleted", "cleanup_pending": True}
                self.warmup()
                return result
            finally:
                self.service.compute.release()

    def retry(self, upload_id, actor, tasks):
        self.authorize(actor)
        with self.service.lock:
            record = self.uploads.get(upload_id, actor)
            if record["status"] == "ready":
                try:
                    self.snapshot(upload_id, actor)
                    return self.uploads.public(record)
                except HTTPException:
                    self.update(upload_id, status="failed", stage="repair", error="INDEX_CORRUPT")
            return self.start(upload_id, actor, tasks)


def library_router(library, search_request):
    router = APIRouter(prefix="/library", tags=["video library"])

    @router.get("")
    def catalog(actor=Depends(get_actor)):
        return {"uploads": library.catalog(actor)}

    @router.get("/config")
    def config(actor=Depends(get_actor)):
        library.authorize(actor)
        return {
            "enabled": bool(ffmpeg_binary()),
            "max_file_bytes": library.uploads.config.uploads.max_file_bytes,
            "max_files": library.uploads.config.uploads.max_files,
            "video_extensions": [".mp4"],
            "max_duration_seconds": 1800,
            "privacy": "privacy-v2.1",
            "default_mode": "eg2_only",
            "modes": ["eg2_only", "shadow", "active"],
            "v6_experimental": True,
            "filters": ["source_ids", "start_sec", "end_sec"],
            "scientific_status": "HOLD",
        }

    @router.post("/uploads", status_code=201)
    def create(body: UploadInput, actor=Depends(get_actor), key: str | None = Header(None, alias="Idempotency-Key")):
        library.authorize(actor)
        if any(
            f.relative_path.split("/")[0].startswith("_") or Path(f.relative_path).suffix.lower() != ".mp4"
            for f in body.files
        ):
            raise HTTPException(422, "INVALID_RELATIVE_PATH")
        return library.uploads.create(body.folder_name, body.files, actor, key)

    @router.put("/uploads/{upload_id}/files/{file_id}")
    async def receive(upload_id: str, file_id: str, request: Request, actor=Depends(get_actor)):
        library.authorize(actor)
        return await library.uploads.receive(upload_id, file_id, request, actor)

    @router.post("/uploads/{upload_id}/complete")
    def complete(upload_id: str, actor=Depends(get_actor)):
        library.authorize(actor)
        return library.complete(upload_id, actor)

    @router.post("/uploads/{upload_id}/start", status_code=202)
    def start(upload_id: str, tasks: BackgroundTasks, actor=Depends(get_actor)):
        return library.start(upload_id, actor, tasks)

    @router.get("/uploads/{upload_id}")
    def state(upload_id: str, actor=Depends(get_actor)):
        library.authorize(actor)
        return library.uploads.public(library.uploads.get(upload_id, actor))

    @router.post("/uploads/{upload_id}/retry", status_code=202)
    def retry(upload_id: str, tasks: BackgroundTasks, actor=Depends(get_actor)):
        return library.retry(upload_id, actor, tasks)

    @router.delete("/uploads/{upload_id}")
    def delete(upload_id: str, actor=Depends(get_actor)):
        library.authorize(actor)
        return library.delete(upload_id, actor)

    def search(upload_id: str, body, actor=Depends(get_actor)):
        library.authorize(actor)
        return library.search(upload_id, body.query, body.top_k, actor, mode=body.mode, filters=body.filters)

    search.__annotations__["body"] = search_request
    router.add_api_route("/{upload_id}/search", search, methods=["POST"])

    @router.get("/{upload_id}/clips")
    def clips(upload_id: str, actor=Depends(get_actor)):
        _, _, rows = library.snapshot(upload_id, actor)
        return {"clips": rows}

    @router.get("/{upload_id}/readiness")
    def readiness(upload_id: str, actor=Depends(get_actor)):
        from fastapi.responses import JSONResponse

        state = library.readiness(upload_id, actor)
        return JSONResponse(state, status_code=200 if state["ready"] else 503)

    @router.post("/{upload_id}/export")
    def export(upload_id: str, body: dict, actor=Depends(get_actor)):
        _, _, rows = library.snapshot(upload_id, actor)
        selected = body.get("rows", [])
        if (
            not isinstance(selected, list)
            or not selected
            or len(selected) > len(rows)
            or any(type(i) is not int or i < 0 or i >= len(rows) for i in selected)
        ):
            raise HTTPException(422, "INVALID_EXPORT_SELECTION")
        values = [
            {k: rows[i][k] for k in ("row", "clip_id", "source_id", "filename", "start_sec", "end_sec", "privacy")}
            for i in selected
        ]
        search_body = body.get("search")
        if search_body is not None:
            try:
                request = search_request.model_validate(search_body)
            except Exception:
                raise HTTPException(422, "INVALID_EXPORT_SEARCH") from None
            payload = library.search(upload_id, request.query, 20, actor, mode=request.mode, filters=request.filters)
            hits = {hit["row"]: hit for hit in payload["results"]}
            if any(i not in hits for i in selected):
                raise HTTPException(422, "EXPORT_ROW_OUTSIDE_RESULTS")
            for value, i in zip(values, selected):
                value.update({k: hits[i][k] for k in ("rank", "global_score", "final_score", "mode")})
        else:
            for rank, value in enumerate(values, 1):
                value.update(rank=rank, global_score=None, final_score=None, mode="library_selection")
        if body.get("format", "json") == "csv":
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(values[0]))
            writer.writeheader()
            for value in values:
                writer.writerow(
                    {
                        k: "'" + v if isinstance(v, str) and v.lstrip().startswith(("=", "+", "-", "@")) else v
                        for k, v in value.items()
                    }
                )
            return Response(
                stream.getvalue(),
                media_type="text/csv",
                headers={"Content-Disposition": 'attachment; filename="clips.csv"'},
            )
        return {"index_id": upload_id, "clips": values}

    @router.post("/{upload_id}/previews/{row}/url")
    def grant(upload_id: str, row: int, request: Request, actor=Depends(get_actor)):
        _, clip = library.media(upload_id, row, actor)
        token, expires = demo_previews.grant(upload_id, row, actor)
        base = str(request.base_url).rstrip("/") + f"/api/v1/library/{upload_id}/previews/{row}"
        return {
            "url": f"{base}/media?token={token}",
            "poster_url": f"{base}/poster?token={token}",
            "expires_in_seconds": demo_previews.TTL,
            "expires_at": expires,
            "start_sec": 0,
            "end_sec": clip["end_sec"] - clip["start_sec"],
            "source_id": clip["source_id"],
            "source_start_sec": clip["start_sec"],
            "source_end_sec": clip["end_sec"],
            "privacy": "privacy-v2.1",
        }

    @router.get("/{upload_id}/previews/{row}/media")
    def media(upload_id: str, row: int, token: str = Query(max_length=4096)):
        actor = demo_previews.granted_actor(token, upload_id, row)
        path, _ = library.media(upload_id, row, actor)
        return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})

    @router.get("/{upload_id}/previews/{row}/poster")
    def poster(upload_id: str, row: int, token: str = Query(max_length=4096)):
        actor = demo_previews.granted_actor(token, upload_id, row)
        path, _ = library.media(upload_id, row, actor, "jpg")
        return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})

    return router
