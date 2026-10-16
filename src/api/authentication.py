"""Verify issuer-signed bearer tokens; tenant and resource scopes come from claims."""
import os
from functools import lru_cache

from fastapi import HTTPException

from src.retrieval.domain.entities import ActorContext


@lru_cache(maxsize=4)
def jwks_client(url):
    from jwt import PyJWKClient
    return PyJWKClient(url, cache_keys=True, lifespan=60, timeout=10)


def verified_bearer(request):
    import jwt

    issuer, audience, jwks = (os.getenv(k) for k in ("AUTH_ISSUER", "AUTH_AUDIENCE", "AUTH_JWKS_URL"))
    if not all((issuer, audience, jwks)):
        raise HTTPException(status_code=503, detail="AUTH_PROVIDER_NOT_CONFIGURED")
    if not jwks.startswith("https://"):
        raise HTTPException(status_code=503, detail="AUTH_JWKS_REQUIRES_HTTPS")
    authorization = request.headers.get("authorization", "")
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="BEARER_TOKEN_REQUIRED")
    token = authorization[7:]
    try:
        key = jwks_client(jwks).get_signing_key_from_jwt(token)
        claims = jwt.decode(token, key.key, algorithms=["RS256", "ES256"], audience=audience, issuer=issuer,
                            options={"require": ["exp", "iat", "sub", "iss", "aud", "tenant_id", "access_scope_ids"]})
        scopes = claims["access_scope_ids"]
        roles = claims.get("roles", [])
        if not isinstance(scopes, list) or not all(isinstance(s, str) and s for s in scopes):
            raise ValueError("invalid access scope claims")
        if not isinstance(roles, list) or not all(isinstance(r, str) for r in roles):
            raise ValueError("invalid role claims")
        return ActorContext(actor_id=claims["sub"], tenant_id=claims["tenant_id"], access_scope_ids=scopes, roles=roles)
    except (jwt.PyJWKClientConnectionError, jwt.PyJWKClientError) as exc:
        raise HTTPException(status_code=503, detail="AUTH_PROVIDER_UNAVAILABLE") from exc
    except (jwt.InvalidTokenError, ValueError, TypeError) as exc:
        raise HTTPException(status_code=401, detail="TOKEN_INVALID") from exc
