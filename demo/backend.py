"""Scene Search workspace over the published corpus and isolated user uploads."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, model_validator

from demo.library import LIBRARY_ROOT, VideoLibrary, library_router
from src.api.dependencies import get_actor
from src.hybrid.config import load_hybrid_config
from src.hybrid.service import ROOT, HybridService

CONFIG = ROOT / "configs/hybrid/driveevidence_v6.yaml"
PUBLISHED = ROOT / "outputs/driveevidence_v6/index/serve_index"
SERVICE_NAME = "scenesearch-workspace"
SAFE_METADATA = (
    "clip_id",
    "source_clip_id",
    "source_row",
    "sequence",
    "day_night",
    "segment_id",
    "start_sec",
    "end_sec",
    "source_start_sec",
    "source_end_sec",
    "split",
    "fold",
)


class DemoService(HybridService):
    """Restrict discovery before any engine can read an index or load a model."""

    def _indexes(self):
        return [PUBLISHED] if (PUBLISHED / "index_meta.json").is_file() else []

    def train_snapshot(self, path):
        try:
            meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
            manifest_bytes = (path / "manifest.json").read_bytes()
            rows = json.loads(manifest_bytes)["clips"]
            policy = meta["policy"]
            if (
                meta["format"] != "driveevidence-v6"
                or policy.get("enabled") is not False
                or policy.get("state") != "EG2_ONLY"
                or hashlib.sha256(manifest_bytes).hexdigest() != meta["hashes"]["manifest.json"]
                or not rows
                or len(rows) != meta["num_clips"]
                or len({row["row"] for row in rows}) != len(rows)
                or any(row.get("split") != "train" for row in rows)
            ):
                raise ValueError("Published TRAIN index or active policy changed")
            return meta, rows
        except (OSError, ValueError, KeyError, TypeError):
            raise HTTPException(409, "DEMO_TRAIN_INDEX_REQUIRED") from None

    def resolve_index(self, index_id, actor):
        path = super().resolve_index(index_id, actor)
        self.train_snapshot(path)
        return path


class SearchFilters(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_ids: list[str] = Field(default_factory=list, max_length=20)
    start_sec: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    end_sec: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def ordered(self):
        if self.start_sec is not None and self.end_sec is not None and self.end_sec <= self.start_sec:
            raise ValueError("FILTER_TIME_RANGE_INVALID")
        return self


class DemoSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=1, max_length=1000, pattern=r"\S")
    top_k: int = Field(default=5, ge=1, le=20)
    mode: Literal["eg2_only", "shadow", "active"] = "eg2_only"
    filters: SearchFilters = Field(default_factory=SearchFilters)


def create_app(*, library_root=LIBRARY_ROOT, warmup=True):
    import threading

    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse

    from src.retrieval.domain.entities import ActorContext

    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", DRIVEEVIDENCE_V6_DEPLOYMENT_MODE="eg2_only")
    service = DemoService(load_hybrid_config(CONFIG))
    library = VideoLibrary(service, library_root)

    def local_actor(request: Request):
        # Header identities are convenience scopes for one trusted local user.
        # They are not an authentication boundary or a multi-user deployment.
        return ActorContext(
            actor_id=request.headers.get("X-Actor-Id", "local-user"),
            tenant_id=request.headers.get("X-Tenant-Id", "default"),
            access_scope_ids=request.headers.get("X-Access-Scopes", "default").split(","),
            roles=request.headers.get("X-Roles", "Engineer").split(","),
        )

    def initialize():
        with service.compute:
            library.warmup()

    @asynccontextmanager
    async def lifespan(app):
        thread = None
        if warmup:
            thread = threading.Thread(target=initialize, name="product-warmup", daemon=True)
            thread.start()
        yield
        if thread:
            thread.join(timeout=600)
        library.close()

    app = FastAPI(title="Scene Search Local Product", lifespan=lifespan)
    app.state.library = library
    app.dependency_overrides[get_actor] = local_actor
    app.include_router(library_router(library, DemoSearchRequest), prefix="/api/v1")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8501", "http://localhost:8501"],
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["Content-Type", "Idempotency-Key"],
    )

    @app.middleware("http")
    async def local_boundary(request, call_next):
        from urllib.parse import urlparse

        host = request.url.hostname
        peer = request.client.host if request.client else ""
        origin = request.headers.get("origin")
        parsed = urlparse(origin) if origin else None
        if (
            host not in {"127.0.0.1", "localhost", "testserver"}
            or peer not in {"127.0.0.1", "::1", "testclient"}
            or parsed
            and (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"})
        ):
            return JSONResponse({"detail": "LOCAL_ONLY"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        return response

    @app.get("/health")
    def health():
        return {
            "status": "ok",
            "service": SERVICE_NAME,
            "library_enabled": True,
            "user_ranking": "EG2",
            "v6_mode": "eg2_only",
            "scientific_status": "HOLD",
            "ready": library.warmed,
            "engine": {
                "ready": library.warmed,
                "models_loaded": library.worker is not None,
                "warmup_complete": library.warmed,
            },
            "local_only": True,
        }

    @app.get("/ready")
    @app.get("/api/v1/readiness")
    def ready():
        busy = service.compute.locked()
        data = {
            "ready": library.warmed and not busy,
            "service_state": "BUSY" if busy else "READY_EG2_ONLY" if library.warmed else "NOT_READY",
            "eg2_model_loaded": library.worker is not None,
            "warmup_complete": library.warmed,
            "asset_hashes_valid": library._assets is not None,
            "remote_inference": False,
            "startup_seconds": library.startup_seconds,
            "error": library.runtime_error,
            "scientific_status": "HOLD",
            "collections": [
                {
                    "index_id": r["upload_id"],
                    "state": r["status"],
                    "privacy": r.get("privacy_state", "NOT_READY"),
                    "v6": r.get("v6_state", "NOT_READY"),
                }
                for r in library.uploads.repository.list("demo_upload")
                if r["status"] != "deleted"
            ],
        }
        return JSONResponse(data, status_code=200 if data["ready"] else 503)

    @app.get("/api/v1/hybrid/indexes")
    def legacy_catalog(actor=Depends(get_actor)):
        library.authorize(actor)
        return {"indexes": []}

    @app.get("/api/v1/capabilities")
    def capabilities():
        return {
            "product_path": "Streamlit + local FastAPI",
            "default_mode": "eg2_only",
            "modes": ["eg2_only", "shadow", "active"],
            "v6_experimental": True,
            "privacy": "privacy-v2.1",
            "filters": ["source_ids", "start_sec", "end_sec"],
            "precision_recall_ndcg": "NOT_ESTABLISHED",
            "scientific_status": "HOLD",
            "remote_inference": False,
            "raw_media_fallback": False,
            "raw_query_logging": False,
        }

    return app
