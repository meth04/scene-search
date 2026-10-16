from __future__ import annotations

from uuid import uuid4

from src.retrieval.application.window_common import require, utc_now
from src.retrieval.domain.errors import (
    ConflictError,
    ForbiddenError,
    RetrievalError,
)


class WindowResultSetService:
    """Manage owned, versioned collections of confirmed search results."""

    def __init__(self, search):
        """Bind service dependencies and initialize local coordination state.

        Args:
            search: Search service sharing its repository, media access and configuration.
        """
        self.search, self.repository = search, search.repository

    def create(self, command, actor):
        """Persist an empty result set owned by the current actor and tenant.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            New result-set record.
        """
        result_set = {
            "result_set_id": str(uuid4()),
            "owner_id": actor.actor_id,
            "tenant_id": actor.tenant_id,
            "name": command.name,
            "version": 1,
            "index_version": None,
            "items": [],
            "created_at": utc_now(),
        }
        self.repository.put("set", result_set["result_set_id"], result_set)
        return result_set

    def get(self, result_set_id, actor):
        """Read a result set after checking ownership.

        Args:
            result_set_id: Persisted result-set identifier.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Authorized result-set record.
        """
        value = require(self.repository.get("set", result_set_id))
        self.check_owner(value, actor)
        return value

    @staticmethod
    def check_owner(value, actor):
        """Require result-set actor and tenant ownership.

        Args:
            value: Record or request value to inspect.
            actor: Authenticated actor, tenant and current source access scopes.

        Raises:
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
        """
        if value["owner_id"] != actor.actor_id or value["tenant_id"] != actor.tenant_id:
            raise ForbiddenError("result set is outside actor scope")

    def add_item(self, command, actor):
        """Apply the requested result-set mutation under a version check.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Current or updated result-set record.
        """
        return self._mutate(command, actor)

    def remove_item(self, command, actor):
        """Apply the requested result-set removal under a version check.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Current or updated result-set record.
        """
        return self._mutate(command, actor)

    def _mutate(self, command, actor):
        """Authorize the result and atomically update a versioned result set.

        Args:
            command: Validated operation inputs, including any required version and request identity.
            actor: Authenticated actor, tenant and current source access scopes.

        Returns:
            Current or updated result-set record.

        Raises:
            ConflictError: If versions, request identity or the current state prevent the operation.
            ForbiddenError: If ownership, source authorization or the signed grant is invalid.
            RetrievalError: If the query, filter or export request violates its supported contract.
        """
        result, _ = self.search.media.result(command.result_id, actor)
        with self.repository.transaction() as tx:
            value = require(tx.get("set", command.result_set_id))
            self.check_owner(value, actor)
            if value["version"] != command.expected_version:
                raise ConflictError("RESULT_SET_CONFLICT", details={"current_version": value["version"]})
            current = tx.get("result", command.result_id)
            asset = tx.get("asset", current["asset_id"])
            if (
                not asset
                or not asset["available"]
                or asset["tenant_id"] != actor.tenant_id
                or asset["access_scope_id"] not in actor.access_scope_ids
            ):
                raise ForbiddenError("SOURCE_FORBIDDEN")
            existing = any(i["result_id"] == command.result_id for i in value["items"])
            changed = False
            if command.operation == "add":
                if current["review_status"] != "relevant" or not current["review_range"]:
                    raise RetrievalError("RESULT_NOT_RELEVANT")
                if value["index_version"] and value["index_version"] != result["index_version"]:
                    raise ConflictError("RESULT_SET_INDEX_CONFLICT")
                if not existing:
                    value["items"].append(
                        {
                            "result_id": result["result_id"],
                            "scene_id": result["scene_id"],
                            "index_version": result["index_version"],
                            "added_by": actor.actor_id,
                        }
                    )
                    value["index_version"] = result["index_version"]
                    changed = True
            elif existing:
                value["items"] = [i for i in value["items"] if i["result_id"] != command.result_id]
                changed = True
            if changed:
                value["version"] += 1
                tx.put("set", value["result_set_id"], value)
            return value
