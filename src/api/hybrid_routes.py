"""Local Hybrid V2 indexes use their own embedding and ranking contract."""

import os
from functools import lru_cache
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_actor, production_mode
from src.hybrid.config import load_hybrid_config
from src.hybrid.service import HybridService

router = APIRouter(prefix="/hybrid", tags=["hybrid-v2"])


@lru_cache(maxsize=4)
def configured_service(path):
    return HybridService(load_hybrid_config(path))


def get_hybrid_service():
    if production_mode():
        raise HTTPException(403, "HYBRID_LOCAL_ONLY")
    path = os.getenv("HYBRID_CONFIG")
    if not path:
        raise HTTPException(503, "HYBRID_CONFIG_REQUIRED")
    return configured_service(path)


class HybridSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    top_k: int = Field(default=10, ge=1, le=100)
    ranker: Literal["caption", "stage3", "base", "stage5", "dense", "dense-entity", "embeddinggemma2", "driveevidence", "driveevidence_v2", "driveevidence_v3", "driveevidence_v4", "driveevidence_v5", "driveevidence_v6"] | None = None
    decision_provider: Literal["none", "laya"] | None = None
    deployment_mode: Literal["eg2_only", "shadow", "active"] | None = None
    debug: bool = False


class HybridIndexRequest(BaseModel):
    video: str = Field(min_length=1)
    max_windows: int | None = Field(default=None, ge=1)


@router.get("/health")
def health():
    return {"status": "ok", "qwen_required_for_search": False}


@router.get("/readiness")
def readiness(index_id: str | None = None, operation: Literal["search", "index"] = "search", actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    try:
        return service.readiness(actor, index_id, operation)
    except (ValueError, RuntimeError, OSError, KeyError):
        raise HTTPException(409, {"code": "INDEX_NOT_READY"}) from None


@router.get("/indexes/{index_id}/clip")
def private_clip(index_id: str, start: float, end: float, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    return FileResponse(service.private_clip(index_id, actor, start, end), media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@router.get("/indexes")
def indexes(actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    return {"indexes": service.list_indexes(actor)}


@router.get("/indexes/{index_id}/media")
def private_media(index_id: str, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    path = service.private_media(index_id, actor)
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": "private, no-store"})


@router.post("/indexes/{index_id}/search")
def search(index_id: str, body: HybridSearchRequest, request: Request, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    import logging
    import time
    from uuid import uuid4
    request_id = uuid4().hex
    supplied = request.headers.get("X-Correlation-ID", "")
    correlation_id = supplied if supplied.isalnum() and len(supplied) <= 80 else request_id
    started = time.perf_counter()
    code, status = None, "SUCCESS"
    try:
        options = {"top_k": body.top_k, "ranker": body.ranker}
        if body.decision_provider is not None:
            options["decision_provider"] = body.decision_provider
        if body.deployment_mode is not None:
            options["deployment_mode"] = body.deployment_mode
        if body.debug:
            options["debug"] = True
        result = service.search(index_id, body.query, actor, **options)
        result.update(request_id=result.get("request_id", request_id), correlation_id=correlation_id)
        return result
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        logging.getLogger(__name__).exception("Search failed: request_id=%s", request_id)
        code = "INVALID_QUERY" if str(exc) == "INVALID_QUERY" else "INDEX_NOT_READY" if str(exc) == "INDEX_NOT_READY" else "INDEX_CORRUPT"
        status = "FAILED"
        raise HTTPException(422, {"code": code, "request_id": request_id, "correlation_id": correlation_id}) from None
    except HTTPException as exc:
        status, code = "FAILED", exc.detail
        if isinstance(exc.detail, dict):
            raise HTTPException(exc.status_code, {**exc.detail, "request_id": request_id, "correlation_id": correlation_id}, headers=exc.headers) from None
        raise
    finally:
        logging.getLogger(__name__).info(__import__("json").dumps(dict(request_id=request_id, correlation_id=correlation_id, index_id=index_id,
            operation="search", duration_ms=1000*(time.perf_counter()-started), status=status, error_code=code)))


@router.post("/jobs", status_code=202)
def start_job(
    body: HybridIndexRequest, tasks: BackgroundTasks, actor=Depends(get_actor), service=Depends(get_hybrid_service)
):
    job, command = service.prepare_job(body.video, actor, max_windows=body.max_windows)
    tasks.add_task(service.run_job, job, command)
    return job


@router.get("/jobs/{job_id}")
def job(job_id: str, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    return service.get_job(job_id, actor)
