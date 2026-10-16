"""Build indexing input from local videos without renaming or moving sources."""

from pathlib import Path

from src.indexing.application.commands import RunIndexCommand
from src.indexing.application.request_errors import IndexingRequestError
from src.indexing.domain.entities import DatasetManifest, VideoManifestItem, stable_hash
from src.indexing.models.driveretriever_adapter import file_checksum
from src.indexing.video.source_reader import SourceReader

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".m4v", ".mpeg", ".mpg", ".ts"}


def scan_videos(folder, config):
    """Return sorted CLI video paths within the configured source roots.

    Create the folder if missing; resolve each video through the source reader.

    Args:
        folder: Local folder to scan recursively.
        config: Pipeline settings containing allowed source roots.

    Returns:
        Authorized paths matching the CLI's video extensions.

    Raises:
        ValueError: If the folder or a video resolves outside the source roots.
    """
    folder = Path(folder).resolve()
    reader = SourceReader(config.driveretriever.source_roots)
    if not any(folder.is_relative_to(root) for root in reader.roots):
        raise ValueError("SOURCE_NOT_ALLOWED")
    folder.mkdir(parents=True, exist_ok=True)
    return [
        reader.path(str(path))
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]


def folder_command(folder, config, *, tenant_id="default", access_scope_id="default"):
    """Build a content-addressed CLI manifest without moving source videos.

    Args:
        folder: Local folder to scan recursively.
        config: Pipeline settings and allowed source roots.
        tenant_id: Tenant attached to every source item.
        access_scope_id: Source access scope attached to every item.

    Returns:
        Indexing command, or None if no supported videos were found.

    Raises:
        ValueError: If a source resolves outside the configured roots.
        OSError: If scanning or reading source checksums fails.
    """
    folder = Path(folder).resolve()
    items = [
        VideoManifestItem(
            video_id=path.relative_to(folder).as_posix(),
            uri=str(path),
            sha256=file_checksum(path),
            metadata={"filename": path.name, "tenant_id": tenant_id, "access_scope_id": access_scope_id},
        )
        for path in scan_videos(folder, config)
    ]
    if not items:
        return None
    manifest = DatasetManifest(
        source_id="folder-" + stable_hash(str(folder))[:16],
        dataset_version=stable_hash([item.model_dump(mode="json") for item in items]),
        videos=items,
    )
    return RunIndexCommand(manifest=manifest, config=config)


def import_api_folder(request, actor, config, get_orchestrator):
    """Scan a local API folder using its original extensions, ordering and naming.

    Args:
        request: Parsed API request or streaming request body.
        actor: Authenticated actor, tenant and current source access scopes.
        config: Validated configuration; saved snapshots retain their original serialized fields.
        get_orchestrator: Factory preserving the endpoint's server configuration checks.

    Returns:
        Original empty-folder warning or queued-job response.

    Raises:
        IndexingRequestError: If source selection, upload limits or job creation rejects the request.
    """
    if config.storage.provider == "production":
        raise IndexingRequestError(status_code=409, detail="SUBMIT_REGISTERED_SOURCE_MANIFEST")
    reader = SourceReader(config.driveretriever.source_roots)
    folder = Path(request.folder_path).resolve()
    if not any(folder.is_relative_to(root) for root in reader.roots):
        raise IndexingRequestError(status_code=403, detail="SOURCE_NOT_ALLOWED")
    if not folder.is_dir():
        raise IndexingRequestError(status_code=404, detail="source folder was not found")

    items = _api_folder_items(folder, actor)
    if not items:
        return {
            "status": "warning",
            "message": f"Scanned '{folder.resolve()}'. No video files found. Place video files (.mp4, .mkv) in the directory.",
            "scanned_path": str(folder.resolve()),
            "imported_count": 0,
            "videos": [],
        }
    return _queue_api_folder(request, actor, config, get_orchestrator, items)


def _api_folder_items(folder, actor):
    """Build API source items in filesystem order using the API extension set.

    Keep API naming and metadata distinct from the content-addressed CLI import.

    Args:
        folder: Resolved source folder accepted by the endpoint.
        actor: Authenticated tenant and source scopes.

    Returns:
        Manifest items with relative video IDs and resolved source paths.

    Raises:
        OSError: If source enumeration or file metadata cannot be read.
    """
    video_extensions = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
    video_files = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in video_extensions]
    return [
        VideoManifestItem(
            video_id=path.relative_to(folder).as_posix(),
            uri=str(path.resolve()),
            metadata={
                "size_bytes": path.stat().st_size,
                "filename": path.name,
                "tenant_id": actor.tenant_id,
                "access_scope_id": actor.access_scope_ids[0],
            },
        )
        for path in video_files
    ]


def _queue_api_folder(request, actor, config, get_orchestrator, items):
    """Queue an API folder manifest and preserve its job-error response.

    Args:
        request: API request supplying source ID and dataset version.
        actor: Authenticated source owner.
        config: Validated pipeline settings.
        get_orchestrator: Factory enforcing server configuration checks.
        items: Source items in the original scan order.

    Returns:
        Queued-job response with the number of imported videos.

    Raises:
        IndexingRequestError: If service or job creation fails.
    """
    manifest = DatasetManifest(
        source_id=request.source_id,
        dataset_version=request.dataset_version,
        videos=items,
    )

    try:
        orchestrator = get_orchestrator(config)
        command = RunIndexCommand(manifest=manifest, config=config)
        job = orchestrator.prepare(command, actor)
        return {
            "status": job.status.value,
            "message": f"Indexing job created for {len(items)} videos.",
            "imported_count": len(items),
            "job": job.model_dump(mode="json"),
        }
    except Exception as exc:
        raise IndexingRequestError(status_code=422, detail=str(exc)) from exc
