"""Cache-only local preview API; every media request revalidates its source."""
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import FileResponse

from src.api.dependencies import get_actor
from src.api.hybrid_routes import get_hybrid_service
from src.hybrid import demo_previews

router = APIRouter(prefix='/hybrid/indexes', tags=['demo-previews'])


def create_app():
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    app = FastAPI(title='DriveEvidence V6 demo previews')
    app.add_middleware(CORSMiddleware, allow_origins=[
        'http://127.0.0.1:8000', 'http://localhost:8000',
        'http://127.0.0.1:8080', 'http://localhost:8080',
    ],
        allow_methods=['GET', 'POST'], allow_headers=['X-Roles', 'X-Actor-Id', 'X-Tenant-Id', 'X-Access-Scopes', 'Content-Type'])
    app.include_router(router, prefix='/api/v1')

    @app.get('/health')
    def health():
        return {'status': 'ok', 'service': 'driveevidence-v6-demo-previews'}

    return app


@router.get('/{index_id}/previews')
def previews(index_id: str, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    try:
        return demo_previews.available(service, index_id, actor)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(409, 'PREVIEW_UNAVAILABLE') from None


@router.post('/{index_id}/previews/{row}/url')
def preview_url(index_id: str, row: int, request: Request, actor=Depends(get_actor), service=Depends(get_hybrid_service)):
    try:
        _, receipt = demo_previews.resolve(service, index_id, row, actor)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(409, 'PREVIEW_UNAVAILABLE') from None
    token, expires = demo_previews.grant(index_id, row, actor)
    base = str(request.base_url).rstrip('/') + f'/api/v1/hybrid/indexes/{index_id}/previews/{row}'
    return dict(url=f'{base}/media?token={token}', poster_url=f'{base}/poster?token={token}',
        expires_at=datetime.fromtimestamp(expires, UTC).isoformat(), expires_in_seconds=demo_previews.TTL,
        start_sec=receipt['preview_start_sec'], end_sec=receipt['preview_end_sec'])


def media_response(index_id, row, token, service, asset):
    actor = demo_previews.granted_actor(token, index_id, row)
    try:
        path, _ = demo_previews.resolve(service, index_id, row, actor, asset)
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(409, 'PREVIEW_UNAVAILABLE') from None
    return FileResponse(path, media_type='video/mp4' if asset.endswith('.mp4') else 'image/jpeg',
        headers={'Cache-Control': 'private, no-store'})


@router.get('/{index_id}/previews/{row}/media')
def media(index_id: str, row: int, token: str = Query(max_length=4096), service=Depends(get_hybrid_service)):
    return media_response(index_id, row, token, service, 'preview.mp4')


@router.get('/{index_id}/previews/{row}/poster')
def poster(index_id: str, row: int, token: str = Query(max_length=4096), service=Depends(get_hybrid_service)):
    return media_response(index_id, row, token, service, 'poster.jpg')
