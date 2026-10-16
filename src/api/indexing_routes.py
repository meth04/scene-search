import os
from functools import lru_cache

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from pydantic import BaseModel

from src.api.dependencies import get_actor, production_mode
from src.indexing.application.commands import RunIndexCommand
from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator
from src.indexing.config import PipelineConfig, load_pipeline_config
from src.indexing.domain.entities import DatasetManifest, stable_hash
from src.retrieval.domain.entities import ActorContext

router = APIRouter(prefix="/indexing", tags=["indexing"])


class RunRequest(BaseModel):
    manifest: DatasetManifest
    config: PipelineConfig | None = None


class VersionRequest(BaseModel):
    index_version: str
    alias: str = "scenes_production"
    config: PipelineConfig | None = None
    expected_generation: int | None = None


class ValidateRequest(BaseModel):
    index_version: str
    config: PipelineConfig | None = None


class RollbackRequest(VersionRequest):
    index_version: str | None = None


def get_orchestrator(config: PipelineConfig | None = None) -> WindowIndexingOrchestrator:
    """Enforce server configuration and construct the supported window indexer.

    Args:
        config: Validated configuration; saved snapshots retain their original serialized fields.

    Returns:
        Window orchestrator for the approved configuration.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    config = config or server_indexing_config()
    if production_mode() and config.config_hash != server_indexing_config().config_hash:
        raise HTTPException(status_code=403, detail="SERVER_CONFIG_REQUIRED")
    from src.bootstrap import build_indexing

    return build_indexing(config)


@lru_cache
def server_indexing_config():
    path = os.getenv("INDEXING_CONFIG")
    return load_pipeline_config(path) if path else PipelineConfig()


def require_engineer(actor):
    if not set(actor.roles) & {"Engineer", "Admin"}:
        raise HTTPException(status_code=403, detail="INDEX_ADMIN_REQUIRED")


def check_window_job_scope(orchestrator, job_id, actor):
    """Recheck tenant, source scope and live registration for a stored job.

    Args:
        orchestrator: Indexer owning the repository and trusted source reader.
        job_id: Persisted indexing job identifier.
        actor: Authenticated caller and current source access scopes.

    Raises:
        HTTPException: If the original or current sources are outside caller access.
    """
    inputs = orchestrator.jobs.get("job_input", job_id)
    if inputs:
        for asset in inputs["verified"]["videos"]:
            if asset["tenant_id"] != actor.tenant_id or asset["access_scope_id"] not in actor.access_scope_ids:
                raise HTTPException(status_code=403, detail="SOURCE_FORBIDDEN")
            current = orchestrator.jobs.asset(asset["asset_id"])
            if (
                not current
                or current.tenant_id != actor.tenant_id
                or current.access_scope_id not in actor.access_scope_ids
            ):
                raise HTTPException(status_code=403, detail="SOURCE_FORBIDDEN")
            try:
                orchestrator.reader.check_current_registration(current, actor)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail="SOURCE_FORBIDDEN") from exc


@router.post("/jobs")
def run_job(
    request: RunRequest,
    response: Response,
    actor: ActorContext = Depends(get_actor),
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
):
    """Authorize and queue an idempotent indexing request for the worker.

    Args:
        request: Parsed API request or streaming request body.
        response: HTTP response whose status may be updated.
        actor: Authenticated actor, tenant and current source access scopes.
        idempotency_key: Caller key used to reuse an identical request safely.

    Returns:
        Persisted job, including its final or paused status.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    config = request.config or server_indexing_config()
    if production_mode() and config.config_hash != server_indexing_config().config_hash:
        raise HTTPException(status_code=403, detail="SERVER_CONFIG_REQUIRED")
    try:
        orchestrator = get_orchestrator(config)
        command = RunIndexCommand(manifest=request.manifest, config=config)
        if config.config_hash != server_indexing_config().config_hash:
            raise HTTPException(status_code=409, detail="SERVER_CONFIG_REQUIRED")
        if not idempotency_key:
            raise HTTPException(status_code=422, detail="IDEMPOTENCY_KEY_REQUIRED")
        request_id = stable_hash({"actor": actor.actor_id, "key": idempotency_key})
        request_hash = stable_hash(command)
        previous = orchestrator.jobs.get("index_request", request_id)
        if previous:
            if previous["request_hash"] != request_hash:
                raise HTTPException(status_code=409, detail="IDEMPOTENCY_KEY_REUSED")
            response.status_code = 202
            check_window_job_scope(orchestrator, previous["job_id"], actor)
            return orchestrator.jobs.get_job(previous["job_id"])
        try:
            job = orchestrator.prepare(command, actor, request_id=request_id, request_hash=request_hash)
        except ValueError as exc:
            if str(exc) == "IDEMPOTENCY_KEY_REUSED":
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            raise
        response.status_code = 202
        return job
    except HTTPException:
        raise
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="SOURCE_FORBIDDEN") from exc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/jobs/{job_id}")
def job_status(job_id: str, actor: ActorContext = Depends(get_actor)):
    require_engineer(actor)
    orchestrator = get_orchestrator()
    job = orchestrator.jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job not found")
    check_window_job_scope(orchestrator, job_id, actor)
    return job


@router.post("/jobs/{job_id}/retry")
def retry_job(job_id: str, request: RunRequest | None = None, actor: ActorContext = Depends(get_actor)):
    """Validate job ownership and request identity before queuing a retry.

    Args:
        job_id: Persisted job identifier.
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Updated indexing job.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    orchestrator = get_orchestrator(request.config if request else None)
    existing = orchestrator.jobs.get_job(job_id)
    if not existing:
        raise HTTPException(status_code=404, detail="job not found")
    check_window_job_scope(orchestrator, job_id, actor)
    try:
        if request:
            stored = orchestrator.jobs.get("job_input", job_id)
            if (
                request.config
                and request.config.config_hash != existing.pipeline_config_hash
                or request.manifest.model_dump(mode="json") != stored["manifest"]
            ):
                raise HTTPException(status_code=409, detail="JOB_CONFIG_CHANGED")
        return orchestrator.control(job_id, "retry")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/jobs/{job_id}/{action}")
def control_job(job_id: str, action: str, actor: ActorContext = Depends(get_actor)):
    """Check actor scope and apply the requested indexing job action.

    Args:
        job_id: Persisted job identifier.
        action: Requested job control action.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Updated indexing job.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    if action not in {"pause", "resume", "cancel"}:
        raise HTTPException(status_code=404, detail="unknown job action")
    orchestrator = get_orchestrator()
    check_window_job_scope(orchestrator, job_id, actor)
    try:
        return orchestrator.control(job_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/validate")
def validate_index(request: ValidateRequest, actor: ActorContext = Depends(get_actor)):
    """Authorize the target index and return its validation gates.

    Args:
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Validation report including the aggregate pass flag.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    try:
        config = request.config or server_indexing_config()
        orchestrator = get_orchestrator(config)
        for job in orchestrator.jobs.list("job"):
            if job["index_version"] == request.index_version:
                check_window_job_scope(orchestrator, job["job_id"], actor)
        report = orchestrator.validate(request.index_version, config)
        return {**report.model_dump(mode="json"), "passed": report.passed}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/publish")
def publish_index(request: VersionRequest, actor: ActorContext = Depends(get_actor)):
    """Authorize target and active indexes before version-checked publication.

    Args:
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Published and previously active index versions.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    try:
        orchestrator = get_orchestrator(request.config)
        active = orchestrator.jobs.active(request.alias)
        for job in orchestrator.jobs.list("job"):
            if (
                job["index_version"] == request.index_version
                or active
                and job["index_version"] == active["index_version"]
            ):
                check_window_job_scope(orchestrator, job["job_id"], actor)
        previous = orchestrator.publish(
            request.index_version,
            request.alias,
            actor_id=actor.actor_id,
            expected_generation=request.expected_generation,
        )
        return {"published": request.index_version, "previous_index_version": previous}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/rollback")
def rollback_index(request: RollbackRequest, actor: ActorContext = Depends(get_actor)):
    """Authorize target and active indexes before version-checked rollback.

    Args:
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Active and previously active index versions.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    try:
        orchestrator = get_orchestrator(request.config)
        active = orchestrator.jobs.active(request.alias)
        for job in orchestrator.jobs.list("job"):
            if (
                job["index_version"] == request.index_version
                or active
                and job["index_version"] == active["index_version"]
            ):
                check_window_job_scope(orchestrator, job["job_id"], actor)
        previous = orchestrator.rollback(
            request.index_version,
            request.alias,
            actor_id=actor.actor_id,
            expected_generation=request.expected_generation,
        )
        return {"active": request.index_version, "previous_index_version": previous}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class FolderImportRequest(BaseModel):
    folder_path: str = "data/videos"
    source_id: str = "local_folder"
    dataset_version: str = "v1.0"


class VideoSearchRequest(BaseModel):
    video_path: str = "data/videos/sample.mp4"
    query_text: str
    top_k: int = 5


@router.post("/import-folder")
def import_folder(request: FolderImportRequest, actor: ActorContext = Depends(get_actor)):
    """Translate folder-import application errors into the existing HTTP contract.

    Args:
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Empty-folder warning or queued indexing job response.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    from src.indexing.application.folder_import import import_api_folder
    from src.indexing.application.request_errors import IndexingRequestError

    try:
        return import_api_folder(request, actor, server_indexing_config(), get_orchestrator)
    except IndexingRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc


@router.post("/video-search")
def search_single_video_context(request: VideoSearchRequest, actor: ActorContext = Depends(get_actor)):
    raise HTTPException(status_code=410, detail="Use search-sessions and runs against a validated index")


@router.post("/upload-video")
async def upload_video(
    request: Request, filename: str = Query("uploaded_video.mp4"), actor: ActorContext = Depends(get_actor)
):
    """Translate legacy upload application errors into the existing HTTP contract.

    Args:
        request: Parsed API request or streaming request body.
        filename: Requested source filename, validated before writing.
        actor: Authenticated actor, tenant and current source access scopes.

    Returns:
        Saved source and queued indexing job response.

    Raises:
        HTTPException: If authorization, request validation or application processing rejects the request.
    """
    require_engineer(actor)
    from src.indexing.application.legacy_upload_service import upload_legacy_video
    from src.indexing.application.request_errors import IndexingRequestError

    try:
        return await upload_legacy_video(
            request, filename, actor, server_indexing_config, get_orchestrator, production_mode()
        )
    except IndexingRequestError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail) from exc
