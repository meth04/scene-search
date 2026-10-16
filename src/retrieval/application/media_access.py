from __future__ import annotations

from uuid import uuid4

from src.indexing.domain.entities import stable_hash
from src.indexing.models.driveretriever_adapter import file_checksum
from src.retrieval.application.window_common import require
from src.retrieval.domain.errors import (
    ConflictError,
    ForbiddenError,
    NotFoundError,
    RetrievalError,
)
from src.retrieval.ranking.pagination import SignedCursorService


class MediaAccess:
    """Check source and result access on every media or review operation."""

    def __init__(self, config, repository, reader, cursor):
        """Bind existing dependencies; do not open connections or load a model.

        Args:
            config: Retrieval settings for signing and preview expiry.
            repository: Window records and source assets.
            reader: Trusted source resolver and registration checker.
            cursor: Existing signature verifier.
        """
        self.config = config
        self.repository = repository
        self.reader = reader
        self.cursor = cursor

    def session(self, session_id, actor):
        """Load a session and enforce its actor and tenant ownership.

        Args:
            session_id: Persisted session identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Authorized session record.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        session = require(self.repository.get("session", session_id))
        if session["owner_id"] != actor.actor_id or session["tenant_id"] != actor.tenant_id:
            raise ForbiddenError("session is outside actor scope")
        return session

    def asset(self, asset_id, actor, *, verify=False):
        """Resolve an authorized current source and optionally verify its checksum.

        Args:
            asset_id: Source asset identifier.
            actor: Authenticated actor, tenant and current source access scopes.
            verify: Whether to read source bytes and verify the checksum as well as file metadata.

        Returns:
            Source asset and resolved local path.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
            NotFoundError: If the requested record or source is unavailable.
        """
        asset = require(self.repository.asset(asset_id), "PREVIEW_UNAVAILABLE")
        if (
            not asset.available
            or asset.tenant_id != actor.tenant_id
            or asset.access_scope_id not in actor.access_scope_ids
        ):
            raise ForbiddenError("SOURCE_FORBIDDEN")
        try:
            self.reader.check_current_registration(asset, actor)
            path = self.reader.path(asset.uri)
            stat = path.stat()
            if (asset.size_bytes and asset.size_bytes != stat.st_size) or (
                asset.mtime_ns and asset.mtime_ns != stat.st_mtime_ns
            ):
                raise ForbiddenError("SOURCE_VERSION_CHANGED")
            if verify and file_checksum(path) != asset.checksum:
                raise ForbiddenError("SOURCE_VERSION_CHANGED")
        except PermissionError as exc:
            raise ForbiddenError("SOURCE_FORBIDDEN") from exc
        except (OSError, ValueError) as exc:
            raise NotFoundError("PREVIEW_UNAVAILABLE") from exc
        return asset, path

    def result(self, result_id, actor):
        """Authorize a result and verify that its source and review subject remain current.

        Args:
            result_id: Persisted search-result identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Result record and representative window.

        Raises:
            ConflictError: If versions, request identity or the current state prevent the operation.
        """
        result = require(self.repository.get("result", result_id))
        run = require(self.repository.get("run", result["run_id"]))
        self.session(run["session_id"], actor)
        window = require(self.repository.get_window(result["index_version"], result["scene_id"]))
        self.asset(window.asset_id, actor, verify=True)
        if result.get("base_subject_version", result["subject_version"]) != self.subject_version(window):
            raise ConflictError("REVIEW_SUBJECT_CHANGED")
        return result, window

    @staticmethod
    def subject_version(window):
        """Hash the representative window identity, interval and media policy.

        Args:
            window: Representative source window.

        Returns:
            Stable review-subject hash.
        """
        return stable_hash(
            {
                "index_version": window.index_version,
                "scene_id": window.scene_id,
                "asset_id": window.asset_id,
                "range": [window.start_ms, window.end_ms],
                "media_policy": window.media_policy_version,
            }
        )

    def create_preview_url(self, result_id, actor):
        """Authorize a result and persist a signed, time-limited media grant.

        Args:
            result_id: Persisted search-result identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Preview URL, interval, anchor, privacy status and grant lifetime.
        """
        result, window = self.result(result_id, actor)
        grant_id = str(uuid4())
        self.repository.put(
            "media_grant", grant_id, {"result_id": result_id, "actor_id": actor.actor_id, "tenant_id": actor.tenant_id}
        )
        signer = SignedCursorService(self.config.cursor_signing_key, self.config.preview.url_ttl_seconds)
        token = signer.sign(grant_id, window.index_version, 0, result["subject_version"])
        return {
            "url": f"/api/v1/source-media/{grant_id}?token={token}",
            "anchor_ms": result["anchor_ms"],
            "start_ms": result["start_ms"],
            "end_ms": result["end_ms"],
            "privacy_status": "cached_required",
            "expires_in_seconds": self.config.preview.url_ttl_seconds,
        }

    def media_path(self, grant_id, token, actor):
        """Verify a media grant and recheck live source access before serving bytes.

        Args:
            grant_id: Persisted signed-media grant identifier.
            token: Signed, expiring grant or cursor token.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Prepared private H.264 media path with a verified privacy cache.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
            NotFoundError: If the requested record or source is unavailable.
        """
        payload = self.cursor.verify(token)
        grant = require(self.repository.get("media_grant", grant_id))
        if payload["run_id"] != grant_id:
            raise ForbiddenError("MEDIA_GRANT_INVALID")
        if actor.actor_id == "local-user" and grant.get("actor_id"):
            from src.retrieval.domain.entities import ActorContext

            actor = ActorContext(
                actor_id=grant["actor_id"],
                tenant_id=grant.get("tenant_id", actor.tenant_id),
                access_scope_ids=actor.access_scope_ids,
                roles=actor.roles,
            )
        elif grant["actor_id"] != actor.actor_id or grant["tenant_id"] != actor.tenant_id:
            raise ForbiddenError("MEDIA_GRANT_INVALID")
        result, window = self.result(grant["result_id"], actor)
        if payload["index_version"] != window.index_version or payload["result_hash"] != result["subject_version"]:
            raise ForbiddenError("MEDIA_GRANT_INVALID")
        asset, source = self.asset(window.asset_id, actor, verify=True)
        from src.privacy.media import find_privacy_cache, private_full_media

        try:
            cache = find_privacy_cache(source, asset.checksum)
            return private_full_media(source, cache)
        except (ValueError, RuntimeError, OSError) as exc:
            raise RetrievalError("PRIVACY_PREVIEW_UNAVAILABLE: " + str(exc)) from exc

    def media_type(self, grant_id):
        """Return the media type of the verified private H.264 derivative.

        Args:
            grant_id: Persisted signed-media grant identifier.

        Returns:
            HTTP media type string.
        """
        return "video/mp4"
