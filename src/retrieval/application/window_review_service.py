from __future__ import annotations

from src.indexing.domain.entities import stable_hash
from src.indexing.domain.windows import source_uuid
from src.retrieval.application.window_common import utc_now
from src.retrieval.domain.errors import (
    ConflictError,
    ForbiddenError,
    RetrievalError,
)


class WindowReviewService:
    """Record versioned reviews and invalidate result-set membership when needed."""

    def __init__(self, search):
        """Bind service dependencies and initialize local coordination state.

        Args:
            search: Search service sharing its repository, media access and configuration.
        """
        self.search, self.repository = search, search.repository

    def update(self, command, actor):
        """Record an idempotent review and invalidate verdicts after interval changes.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Persisted review response with its updated version and subject hash.

        Raises:
            RetrievalError: If the query, filter or export request violates its supported contract.
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
            ConflictError: If versions, request identity or the current state prevent the operation.
        """
        result, window = self.search.media.result(command.result_id, actor)
        start, end = getattr(command, "start_ms", None), getattr(command, "end_ms", None)
        if start is None or end is None or not window.start_ms <= start < end <= window.end_ms:
            raise RetrievalError("REVIEW_RANGE_INVALID: confirmation must lie within the representative window")
        event_id = source_uuid("review", actor.actor_id, command.result_id, command.request_id)
        request_hash = stable_hash(command)
        with self.repository.transaction() as tx:
            previous = tx.get("review_event", event_id)
            if previous:
                if previous["request_hash"] != request_hash:
                    raise ConflictError("IDEMPOTENCY_KEY_REUSED")
                return previous["response"]
            current = tx.get("result", command.result_id)
            asset = tx.get("asset", window.asset_id)
            if (
                not asset
                or not asset["available"]
                or asset["tenant_id"] != actor.tenant_id
                or asset["access_scope_id"] not in actor.access_scope_ids
            ):
                raise ForbiddenError("SOURCE_FORBIDDEN")
            if command.subject_version != current["subject_version"]:
                raise ConflictError("REVIEW_SUBJECT_CHANGED")
            if current["version"] != command.expected_version:
                raise ConflictError(
                    "REVIEW_CONFLICT",
                    details={"current_version": current["version"], "current_status": current["review_status"]},
                )
            old_range = current["review_range"] or {"start_ms": window.start_ms, "end_ms": window.end_ms}
            new_range = {"start_ms": start, "end_ms": end}
            range_changed = old_range != new_range
            verdict = (
                "unreviewed" if range_changed and current["review_status"] != "unreviewed" else command.status.value
            )
            if range_changed:
                current["subject_version"] = stable_hash({"base": current["base_subject_version"], "range": new_range})
            current.update(
                review_status=verdict,
                review_note=command.note,
                review_range={"start_ms": start, "end_ms": end},
                version=current["version"] + 1,
                updated_at=utc_now(),
                review_event_id=event_id,
            )
            tx.put("result", current["result_id"], current)
            tx.put(
                "review_event",
                event_id,
                {
                    "event_id": event_id,
                    "actor_id": actor.actor_id,
                    "request_hash": request_hash,
                    "response": current,
                    "at": utc_now(),
                },
            )
            tx.put(
                "retrieval_audit",
                event_id,
                {
                    "action": "review_updated",
                    "actor_id": actor.actor_id,
                    "result_id": current["result_id"],
                    "at": utc_now(),
                },
            )
            if current["review_status"] != "relevant":
                for result_set in tx.list("set"):
                    ids = [i for i in result_set["items"] if i["result_id"] != current["result_id"]]
                    if len(ids) != len(result_set["items"]):
                        result_set.update(items=ids, version=result_set["version"] + 1)
                        tx.put("set", result_set["result_set_id"], result_set)
            return current
