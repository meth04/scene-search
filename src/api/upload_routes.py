"""Local demo uploads retain the existing actor and engineer access checks."""

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from src.api.dependencies import get_actor, production_mode
from src.api.indexing_routes import get_orchestrator, require_engineer, server_indexing_config
from src.indexing.application.folder_import import VIDEO_EXTENSIONS
from src.indexing.application.upload_service import UploadService

router = APIRouter(prefix="/indexing/uploads", tags=["demo uploads"])


class UploadFileInput(BaseModel):
    relative_path: str = Field(min_length=1, max_length=1024)
    size_bytes: int = Field(gt=0)


class UploadInput(BaseModel):
    folder_name: str = Field(min_length=1, max_length=255)
    files: list[UploadFileInput] = Field(min_length=1, max_length=10000)


def service(actor=Depends(get_actor)):
    require_engineer(actor)
    if production_mode():
        raise HTTPException(409, "REGISTER_SOURCE_BEFORE_INDEXING")
    config = server_indexing_config()
    if config.model_family != "driveretriever":
        raise HTTPException(409, "DEMO_CONFIG_REQUIRED")
    return UploadService(config)


@router.get("/datasets")
def list_datasets(actor=Depends(get_actor)):
    """List ready uploads owned by the actor within their tenant and scopes.

    Args:
        actor: Authenticated identity and current source permissions.

    Returns:
        Owned ready datasets, deduplicated by index version.
    """
    config = server_indexing_config()
    from src.indexing.adapters.window_repository import WindowRepository

    repo = WindowRepository(config.storage.root_dir)
    uploads = repo.list("demo_upload")
    results = []
    seen_indexes = set()
    for u in sorted(uploads, key=lambda x: str(x.get("created_at", "")), reverse=True):
        if (
            u.get("status") == "deleted"
            or u.get("actor_id") != actor.actor_id
            or u.get("tenant_id") != actor.tenant_id
            or u.get("access_scope_id") not in actor.access_scope_ids
        ):
            continue
        job_id = u.get("job_id")
        job = repo.get("job", job_id) if job_id else None
        if job and job.get("status") in {"ready", "published"} and u.get("index_version"):
            idx = u["index_version"]
            if idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            results.append(
                {
                    "upload_id": u["upload_id"],
                    "folder_name": u.get("folder_name", "Video dataset"),
                    "index_version": idx,
                    "file_count": len(u.get("files", [])),
                    "created_at": u.get("created_at", ""),
                    "status": "ready",
                }
            )
    return results


@router.delete("/datasets")
def delete_all_datasets(actor=Depends(get_actor)):
    uploads = service(actor)
    records = [
        u
        for u in uploads.repository.list("demo_upload")
        if u.get("status") != "deleted"
        and u.get("actor_id") == actor.actor_id
        and u.get("tenant_id") == actor.tenant_id
        and u.get("access_scope_id") in actor.access_scope_ids
    ]
    for record in records:
        uploads.delete(record["upload_id"], actor)
    return {"status": "success", "message": "All your datasets deleted", "deleted_count": len(records)}


@router.delete("/datasets/{upload_id}")
def delete_dataset(upload_id: str, actor=Depends(get_actor)):
    return service(actor).delete(upload_id, actor)


@router.get("/config")
def upload_config(actor=Depends(get_actor)):
    require_engineer(actor)
    config = server_indexing_config()
    return {
        "max_file_bytes": config.uploads.max_file_bytes,
        "max_files": config.uploads.max_files,
        "video_extensions": sorted(VIDEO_EXTENSIONS),
        "enabled": config.model_family == "driveretriever" and not production_mode(),
    }


@router.post("", status_code=201)
def create_upload(
    body: UploadInput,
    actor=Depends(get_actor),
    uploads=Depends(service),
    key: str | None = Header(default=None, alias="Idempotency-Key", max_length=200),
):
    return uploads.create(body.folder_name, body.files, actor, key)


@router.get("/{upload_id}")
def get_upload(upload_id: str, actor=Depends(get_actor), uploads=Depends(service)):
    record = uploads.get(upload_id, actor)
    result = uploads.public(record)
    if record["job_id"]:
        job = uploads.repository.get("job", record["job_id"])
        result["job"] = job
        result["job_error"] = uploads.repository.get("job_error", record["job_id"])
        inputs = uploads.repository.get("job_input", record["job_id"])
        media = [uploads.repository.get("browser_media", a["asset_id"]) for a in inputs["verified"]["videos"]]
        result["preview_status"] = "ready" if all(media) else "pending"
    return result


@router.put("/{upload_id}/files/{file_id}")
async def put_file(upload_id: str, file_id: str, request: Request, actor=Depends(get_actor), uploads=Depends(service)):
    return await uploads.receive(upload_id, file_id, request, actor)


@router.post("/{upload_id}/complete")
def complete_upload(upload_id: str, actor=Depends(get_actor), uploads=Depends(service)):
    return uploads.complete(upload_id, actor)


@router.post("/{upload_id}/start", status_code=202)
def start_upload(
    upload_id: str,
    actor=Depends(get_actor),
    uploads=Depends(service),
    key: str = Header(alias="Idempotency-Key", min_length=1, max_length=200),
):
    return uploads.start(upload_id, key, actor, get_orchestrator())
