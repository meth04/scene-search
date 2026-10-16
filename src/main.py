from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from src.api.hybrid_routes import router as hybrid_router
from src.api.indexing_routes import router as indexing_router
from src.api.retrieval_routes import router as retrieval_router
from src.api.routes import router
from src.api.upload_routes import router as upload_router
from src.config import get_settings
from src.retrieval.domain.errors import RetrievalError


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the primary encoder without applying optional Hybrid configuration.

    Args:
        app: Application whose lifespan is being managed.

    Yields:
        Control to the server after primary encoder initialization.
    """
    from src.indexing.telemetry import configure_tracing
    configure_tracing()
    from src.api.indexing_routes import server_indexing_config
    config = server_indexing_config()
    if config.model_family == "driveretriever":
        import asyncio

        from src.indexing.models.driveretriever_adapter import build_driveretriever
        await asyncio.to_thread(build_driveretriever, config.driveretriever)
    settings = get_settings()
    print(f"Starting {settings.app_name} in {settings.app_env} mode")
    yield
    print("Shutting down...")


app = FastAPI(
    title="AI20K Agent",
    description="AI Agent built with LangGraph",
    version="1.0.0",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")
app.include_router(indexing_router, prefix="/api/v1")
app.include_router(retrieval_router, prefix="/api/v1")
app.include_router(upload_router, prefix="/api/v1")
app.include_router(hybrid_router, prefix="/api/v1")

# Mount UI/UX Prototype & RBAC Dashboard
app.mount("/ui", StaticFiles(directory="presentation", html=True), name="ui")

# Source video and export artifacts are served by authorized, expiring gateways.


@app.get("/")
async def root():
    from src.api.dependencies import production_mode
    from src.api.indexing_routes import server_indexing_config
    page = "scenesearch_prototype.html"
    if production_mode() and server_indexing_config().model_family == "driveretriever":
        page = "driveretriever.html"
    return RedirectResponse(url="/ui/" + page)


@app.exception_handler(RetrievalError)
async def retrieval_error_handler(request, exc):
    request_id = request.headers.get("x-request-id", "") or uuid4().hex
    return JSONResponse(
        status_code=exc.status_code,
        content={"error_code": exc.code, "message": str(exc), "request_id": request_id, **exc.details},
    )


@app.get("/health")
async def health():
    return {"status": "ok", "env": settings.app_env}


@app.get("/metrics", include_in_schema=False)
def metrics():
    from fastapi import Response
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
