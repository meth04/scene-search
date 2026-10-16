import os
from pathlib import Path

from src.indexing.application.commands import RunIndexCommand
from src.indexing.application.request_errors import IndexingRequestError
from src.indexing.domain.entities import DatasetManifest, VideoManifestItem


async def upload_legacy_video(request, filename, actor, server_indexing_config, get_orchestrator, is_production):
    """Stream and atomically publish a legacy upload before creating its index job.

    Args:
        request: Parsed API request or streaming request body.
        filename: Requested source filename, validated before writing.
        actor: Authenticated actor, tenant and current source access scopes.
        server_indexing_config: Provider of cached server indexing settings.
        get_orchestrator: Factory preserving the endpoint's server configuration checks.
        is_production: Whether the endpoint must require registered sources instead of local upload.

    Returns:
        Original upload response with saved path and queued job.

    Raises:
        IndexingRequestError: If source selection, upload limits or job creation rejects the request.
    """
    _validate_upload(filename, is_production)
    file_path, size = await _save_source(request, filename, server_indexing_config)
    return _queue_uploaded_source(file_path, filename, size, actor, server_indexing_config, get_orchestrator)


def _validate_upload(filename, is_production):
    """Require a plain filename and local upload mode before writing files.

    Args:
        filename: Requested source filename.
        is_production: Whether uploads must use registered source manifests.

    Raises:
        IndexingRequestError: For an unsafe filename or production upload.
    """
    if (
        Path(filename).name != filename
        or filename in {"", ".", ".."}
        or "/" in filename
        or "\\" in filename
        or ":" in filename
    ):
        raise IndexingRequestError(status_code=422, detail="INVALID_FILENAME")
    if is_production:
        raise IndexingRequestError(status_code=409, detail="REGISTER_SOURCE_BEFORE_INDEXING")


async def _save_source(request, filename, server_indexing_config):
    """Stream to a temporary file and publish without replacing another upload.

    Args:
        request: Request exposing an asynchronous stream of byte blocks.
        filename: Previously validated source filename.
        server_indexing_config: Settings provider queried for each block's limit.

    Returns:
        Published source path and byte count. Temporary files are removed.

    Raises:
        IndexingRequestError: If the destination exists or the size limit is exceeded.
        OSError: If streaming or file publication fails.
    """
    save_dir = Path("data/videos")
    save_dir.mkdir(parents=True, exist_ok=True)

    file_path = save_dir / filename
    if file_path.exists():
        raise IndexingRequestError(status_code=409, detail="SOURCE_ALREADY_EXISTS")
    from uuid import uuid4

    temporary = save_dir / (uuid4().hex + ".upload")
    size = 0
    try:
        with temporary.open("xb") as stream:
            async for block in request.stream():
                size += len(block)
                if size > server_indexing_config().uploads.max_file_bytes:
                    raise IndexingRequestError(status_code=413, detail="UPLOAD_TOO_LARGE")
                stream.write(block)
        # Atomic publication without replacing a source uploaded concurrently.
        os.link(temporary, file_path)
    except FileExistsError as exc:
        raise IndexingRequestError(status_code=409, detail="SOURCE_ALREADY_EXISTS") from exc
    finally:
        if temporary.is_file():
            temporary.unlink()
    return file_path, size


def _queue_uploaded_source(file_path, filename, size, actor, server_indexing_config, get_orchestrator):
    """Queue the saved source while retaining it if job creation fails.

    Args:
        file_path: Atomically published source path.
        filename: Original source filename.
        size: Number of uploaded bytes.
        actor: Authenticated tenant and source scopes.
        server_indexing_config: Provider of current server settings.
        get_orchestrator: Factory enforcing the endpoint's configuration checks.

    Returns:
        Existing upload response including the queued job and saved path.

    Raises:
        IndexingRequestError: If service or job creation fails after saving.
    """

    item = VideoManifestItem(
        video_id=file_path.stem,
        uri=str(file_path.resolve()),
        metadata={
            "size_bytes": size,
            "filename": filename,
            "tenant_id": actor.tenant_id,
            "access_scope_id": actor.access_scope_ids[0],
        },
    )

    manifest = DatasetManifest(
        source_id="local_upload",
        dataset_version="v1.0",
        videos=[item],
    )

    try:
        config = server_indexing_config()
        orchestrator = get_orchestrator(config)
        command = RunIndexCommand(manifest=manifest, config=config)
        job = orchestrator.prepare(command, actor)
        return {
            "status": job.status.value,
            "message": f"Đã lưu '{filename}' và tạo tác vụ indexing.",
            "file_name": filename,
            "saved_path": str(file_path.resolve()),
            "job": job.model_dump(mode="json"),
        }
    except Exception as exc:
        raise IndexingRequestError(
            status_code=422, detail=f"Source saved; indexing job creation failed: {exc}"
        ) from exc
