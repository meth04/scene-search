from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, Header, Query
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_actor, get_retrieval_services
from src.retrieval.application.commands import (
    CreateResultSetCommand,
    CreateSessionCommand,
    ExportCommand,
    ResultSetCommand,
    ReviewCommand,
    SearchCommand,
)
from src.retrieval.domain.entities import ActorContext, ExportConfig
from src.retrieval.domain.states import ReviewStatus

router = APIRouter(tags=["retrieval"])


class SessionRequest(BaseModel):
    title: str = ""
    index_id: str | None = None


class SearchRequest(BaseModel):
    index_id: str = "scenes_production"
    request_key: str = Field(default_factory=lambda: str(uuid4()))
    query_text: str = Field(min_length=1, max_length=1000)
    filters: dict = Field(default_factory=dict)
    page_size: int = Field(default=20, ge=1, le=100)


class ReviewRequest(BaseModel):
    status: ReviewStatus
    expected_version: int = Field(gt=0)
    note: str | None = Field(default=None, max_length=2000)
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    subject_version: str | None = None
    start_ms: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_ms: float | None = Field(default=None, gt=0, allow_inf_nan=False)


class ResultSetRequest(BaseModel):
    name: str


class MutationRequest(BaseModel):
    expected_version: int = Field(gt=0)


class ExportRequest(BaseModel):
    result_set_id: str
    expected_version: int = Field(gt=0)
    config: ExportConfig = Field(default_factory=ExportConfig)
    request_id: str = Field(default_factory=lambda: str(uuid4()))


@router.post("/search-sessions", status_code=201)
def create_session(
    body: SessionRequest, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)
):
    return services.search.create_session(CreateSessionCommand(**body.model_dump()), actor)


@router.get("/search-sessions/{session_id}")
def get_session(session_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Return the current actor's pinned search session.

    Args:
        session_id: Persisted session identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Public authorized session fields.
    """
    return services.search.session(session_id, actor)


@router.get("/source-media/{grant_id}")
def source_media(
    grant_id: str, token: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)
):
    """Authorize the source grant and return a response supporting byte ranges.

    Args:
        grant_id: Persisted signed-media grant identifier.
        token: Signed, expiring grant or cursor token.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Private, non-cacheable file response.
    """
    path = services.search.media_path(grant_id, token, actor)
    # Starlette FileResponse handles byte Range requests for browser seeking.
    return FileResponse(
        path, media_type=services.search.media_type(grant_id), headers={"Cache-Control": "private, no-store"}
    )


@router.get("/export-media/{job_id}")
def export_media(
    job_id: str, token: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)
):
    """Authorize an export download and close its stream after transmission.

    Args:
        job_id: Persisted job identifier.
        token: Signed, expiring grant or cursor token.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Streaming manifest response with attachment headers.
    """
    stream, extension = services.exports.download(job_id, token, actor)

    def chunks():
        try:
            while block := stream.read(65536):
                yield block
        finally:
            stream.close()

    return StreamingResponse(
        chunks(),
        media_type="text/csv" if extension == "csv" else "application/x-ndjson",
        headers={
            "Content-Disposition": f'attachment; filename="manifest.{extension}"',
            "Cache-Control": "private, no-store",
        },
    )


@router.post("/search-sessions/{session_id}/runs")
def create_run(
    session_id: str,
    body: SearchRequest,
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Submit a search against the session's pinned index.

    Args:
        session_id: Persisted session identifier.
        body: Validated HTTP request body.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Search run and its initial page.
    """
    return services.search.execute(SearchCommand(session_id=session_id, **body.model_dump()), actor)


@router.get("/search-runs/{run_id}/results")
def page_results(
    run_id: str,
    cursor: str | None = None,
    limit: int = Query(20, ge=1, le=100),
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Read a signed page of currently authorized search results.

    Args:
        run_id: Persisted search-run identifier.
        cursor: Signed pagination cursor, or None for the initial page.
        limit: Maximum number of result positions scanned for this page.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Result page and optional next cursor.
    """
    return services.search.page(run_id, cursor, actor, limit)


@router.post("/search-results/{result_id}/preview-url")
def preview(result_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Create a signed preview grant for an authorized result.

    Args:
        result_id: Persisted search-result identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Preview URL and representative interval.
    """
    return services.search.create_preview_url(result_id, actor)


@router.put("/search-results/{result_id}/review")
def review(
    result_id: str,
    body: ReviewRequest,
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Apply a versioned review for an authorized result.

    Args:
        result_id: Persisted search-result identifier.
        body: Validated HTTP request body.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Persisted review response.
    """
    return services.review.update(ReviewCommand(result_id=result_id, **body.model_dump()), actor)


@router.post("/result-sets", status_code=201)
def create_result_set(
    body: ResultSetRequest, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)
):
    return services.result_sets.create(CreateResultSetCommand(**body.model_dump()), actor)


@router.get("/result-sets/{result_set_id}")
def get_result_set(
    result_set_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)
):
    """Read the current actor's result set.

    Args:
        result_set_id: Persisted result-set identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Owned result-set record.
    """
    return services.result_sets.get(result_set_id, actor)


@router.put("/result-sets/{result_set_id}/items/{result_id}")
def add_item(
    result_set_id: str,
    result_id: str,
    body: MutationRequest,
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Apply the requested result-set mutation under a version check.

    Args:
        result_set_id: Persisted result-set identifier.
        result_id: Persisted search-result identifier.
        body: Validated HTTP request body.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Current or updated result-set record.
    """
    return services.result_sets.add_item(
        ResultSetCommand(
            result_set_id=result_set_id, result_id=result_id, operation="add", expected_version=body.expected_version
        ),
        actor,
    )


@router.delete("/result-sets/{result_set_id}/items/{result_id}")
def remove_item(
    result_set_id: str,
    result_id: str,
    body: MutationRequest,
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Apply the requested result-set removal under a version check.

    Args:
        result_set_id: Persisted result-set identifier.
        result_id: Persisted search-result identifier.
        body: Validated HTTP request body.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Current or updated result-set record.
    """
    return services.result_sets.remove_item(
        ResultSetCommand(
            result_set_id=result_set_id, result_id=result_id, operation="remove", expected_version=body.expected_version
        ),
        actor,
    )


@router.post("/exports", status_code=202)
def export(
    body: ExportRequest,
    idempotency_key: str = Header(alias="Idempotency-Key"),
    actor: ActorContext = Depends(get_actor),
    services=Depends(get_retrieval_services),
):
    """Submit an idempotent manifest export from a versioned result set.

    Args:
        body: Validated HTTP request body.
        idempotency_key: Caller key used to reuse an identical request safely.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Queued or previously created export job.
    """
    return services.exports.request(ExportCommand(idempotency_key=idempotency_key, **body.model_dump()), actor)


@router.get("/exports/{job_id}")
def export_status(job_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Read the current actor's export status.

    Args:
        job_id: Persisted job identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Public export job fields.
    """
    return services.exports.status(job_id, actor)


@router.post("/exports/{job_id}/download")
def download(job_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Authorize the export and issue a signed, expiring download URL.

    Args:
        job_id: Persisted job identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Download URL and its lifetime in seconds.
    """
    return services.exports.create_download_url(job_id, actor)


@router.post("/exports/{job_id}/revoke")
def revoke(job_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Atomically revoke an owned export and append its audit event.

    Args:
        job_id: Persisted job identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Public revoked job fields.
    """
    return services.exports.revoke(job_id, actor)


@router.post("/exports/{job_id}/retry")
def retry_export(job_id: str, actor: ActorContext = Depends(get_actor), services=Depends(get_retrieval_services)):
    """Requeue a failed export while preserving its snapshot.

    Args:
        job_id: Persisted job identifier.
        actor: Authenticated actor, tenant and current source access scopes.
        services: Configured retrieval services supplied by the API.

    Returns:
        Public queued export job.
    """
    return services.exports.retry(job_id, actor)
