import os
from functools import lru_cache

from fastapi import Header, Request

from src.retrieval.adapters.factory import build_services
from src.retrieval.config import RetrievalConfig, load_retrieval_config
from src.retrieval.domain.entities import ActorContext


def production_mode():
    from src.config import get_settings
    from src.indexing.config import load_structured

    if get_settings().app_env == "production":
        return True
    retrieval = os.getenv("RETRIEVAL_CONFIG")
    indexing = os.getenv("INDEXING_CONFIG")
    return bool(retrieval and load_structured(retrieval).get("provider") == "production") or bool(
        indexing and load_structured(indexing).get("storage", {}).get("provider") == "production"
    )


@lru_cache
def get_retrieval_services():
    path = os.getenv("RETRIEVAL_CONFIG")
    return build_services(load_retrieval_config(path) if path else RetrievalConfig())


def get_actor(
    request: Request,
    x_actor_id: str = Header(default="local-user"),
    x_tenant_id: str = Header(default="default"),
    x_access_scopes: str = Header(default="default"),
    x_roles: str = Header(default="Engineer"),
) -> ActorContext:
    if production_mode():
        # A verified authentication middleware must inject the actor. Client headers
        # are never treated as trusted identity/tenant/scope in production.
        actor = getattr(request.state, "verified_actor", None)
        if not isinstance(actor, ActorContext):
            from src.api.authentication import verified_bearer

            return verified_bearer(request)
        return actor
    return ActorContext(
        actor_id=x_actor_id,
        tenant_id=x_tenant_id,
        access_scope_ids=[value.strip() for value in x_access_scopes.split(",") if value.strip()],
        roles=[value.strip() for value in x_roles.split(",") if value.strip()],
    )
