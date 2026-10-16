from __future__ import annotations

from pathlib import Path

from src.indexing.domain.entities import stable_hash
from src.indexing.domain.windows import VideoAsset, source_uuid
from src.indexing.models.driveretriever_adapter import file_checksum


class SourceReader:
    def __init__(self, roots: list[str], registry_path=None):
        self.roots = [Path(p).resolve() for p in roots]
        self.registry_path = registry_path

    def registration(self, source_id, video_id, actor=None):
        if not self.registry_path:
            return None
        import json
        registry = json.loads(Path(self.registry_path).read_text(encoding="utf-8"))
        entry = registry.get(source_id+"/"+video_id)
        if not entry or not entry.get("available", True):
            raise PermissionError("SOURCE_NOT_REGISTERED")
        metadata = entry["metadata"]
        if actor and (metadata["tenant_id"] != actor.tenant_id or metadata["access_scope_id"] not in actor.access_scope_ids):
            raise PermissionError("SOURCE_FORBIDDEN")
        if actor and actor.actor_id not in entry.get("allowed_actor_ids", []):
            raise PermissionError("SOURCE_FORBIDDEN: actor access was revoked in the source registry")
        return entry

    def check_current_registration(self, asset, actor):
        entry = self.registration(asset.source_id, asset.video_id, actor)
        if entry and (entry["sha256"].lower() != asset.checksum or entry["uri"] != (asset.origin_uri or asset.uri)):
            raise PermissionError("SOURCE_VERSION_CHANGED")

    def materialize(self, uri, checksum):
        import os
        import tempfile
        import urllib.request
        from urllib.parse import urlparse

        parsed = urlparse(uri)
        if parsed.scheme not in {"https", "s3"} or not checksum or not self.registry_path:
            raise ValueError("SOURCE_NOT_ALLOWED: remote sources require registration and checksum")
        if not self.roots:
            raise ValueError("SOURCE_NOT_ALLOWED")
        cache = (self.roots[0] / ".source-cache").resolve()
        if not cache.is_relative_to(self.roots[0]):
            raise ValueError("SOURCE_NOT_ALLOWED: cache is outside the source root")
        cache.mkdir(parents=True, exist_ok=True)
        target = cache / (checksum.lower()+".video")
        if target.is_file() and file_checksum(target) == checksum.lower():
            return target
        temporary = None
        response = None
        try:
            if parsed.scheme == "https":
                class NoRedirect(urllib.request.HTTPRedirectHandler):
                    def redirect_request(self, req, fp, code, msg, headers, newurl):
                        raise ValueError("SOURCE_REDIRECT_NOT_ALLOWED")
                response = urllib.request.build_opener(NoRedirect).open(uri, timeout=60)
            else:
                from minio import Minio
                client = Minio(os.environ["S3_ENDPOINT"], access_key=os.environ["S3_ACCESS_KEY"],
                               secret_key=os.environ["S3_SECRET_KEY"], secure=os.getenv("S3_SECURE", "true").lower() == "true")
                response = client.get_object(parsed.netloc, parsed.path.lstrip("/"))
            with tempfile.NamedTemporaryFile(dir=cache, delete=False) as stream:
                temporary = Path(stream.name)
                while block := response.read(1024*1024):
                    stream.write(block)
            if file_checksum(temporary) != checksum.lower():
                raise ValueError("SOURCE_CHECKSUM_MISMATCH")
            os.replace(temporary, target)
            temporary = None
            return target
        finally:
            if response:
                response.close()
                if hasattr(response, "release_conn"):
                    response.release_conn()
            if temporary and temporary.is_file():
                temporary.unlink()

    def path(self, uri: str) -> Path:
        # A remote source must first be materialized by its trusted storage adapter.
        # Arbitrary network locations never get decoded as client supplied URLs.
        import os
        from urllib.parse import unquote, urlparse

        if "://" in uri:
            parsed = urlparse(uri)
            if parsed.scheme != "file" or parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
                raise ValueError("SOURCE_NOT_ALLOWED: materialize remote sources in an allowed source root")
            uri = unquote(parsed.path)
            if os.name == "nt":
                uri = uri.lstrip("/")
        path = Path(uri).resolve(strict=True)
        if not path.is_file() or not any(path.is_relative_to(root) for root in self.roots):
            raise ValueError("SOURCE_NOT_ALLOWED: path is outside configured source roots")
        return path

    def inspect(self, source_id, item, actor=None) -> VideoAsset:
        import av

        registration = self.registration(source_id, item.video_id, actor)
        if registration and (registration["uri"] != item.uri or registration["sha256"].lower() != (item.sha256 or "").lower()
                             or registration["metadata"] != item.metadata):
            raise PermissionError("SOURCE_REGISTRY_MISMATCH")
        tenant = str(item.metadata.get("tenant_id", "default"))
        scope = str(item.metadata.get("access_scope_id", "default"))
        if actor and (actor.tenant_id != tenant or scope not in actor.access_scope_ids):
            raise PermissionError("SOURCE_FORBIDDEN")
        path = self.materialize(item.uri, item.sha256) if item.uri.startswith(("https://", "s3://")) else self.path(item.uri)
        checksum = file_checksum(path)
        if item.sha256 and item.sha256.lower() != checksum:
            raise ValueError("SOURCE_CHECKSUM_MISMATCH")
        # ffprobe verifies the container independently of the decoder.
        import json
        import subprocess
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
                               check=True, capture_output=True, text=True, encoding="utf-8", timeout=60)
        metadata = json.loads(probe.stdout)
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise ValueError("SOURCE_INVALID: video stream is missing")
            stream = container.streams.video[0]
            if not stream.time_base:
                raise ValueError("SOURCE_INVALID: stream time base is missing")
            first = next((f for f in container.decode(stream) if f.pts is not None), None)
            if first is None:
                raise ValueError("SOURCE_INVALID: readable PTS frames are missing")
            origin = stream.start_time if stream.start_time is not None else first.pts
            duration = (float(stream.duration*stream.time_base)*1000 if stream.duration is not None
                        else float(metadata["format"]["duration"])*1000)
            return VideoAsset(
                asset_id=source_uuid("asset", source_id, item.video_id, checksum, item.uri,
                                     stable_hash(item.metadata), path.stat().st_size, path.stat().st_mtime_ns),
                source_id=source_id, video_id=item.video_id, checksum=checksum, uri=str(path),
                origin_uri=item.uri, size_bytes=path.stat().st_size, mtime_ns=path.stat().st_mtime_ns,
                video_metadata={"streams": metadata.get("streams", []), "format": metadata.get("format", {})},
                duration_ms=duration, stream_index=stream.index, timeline_origin_pts=origin,
                time_base_num=stream.time_base.numerator, time_base_den=stream.time_base.denominator,
                tenant_id=tenant, access_scope_id=scope,
                camera=item.metadata.get("camera", item.metadata.get("camera_id")),
                captured_at=item.metadata.get("captured_at"), region=item.metadata.get("region"),
            )
