"""Expose the existing offline Hybrid pipeline without mixing embedding spaces."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from src.hybrid.config import HybridConfig

ROOT = Path(__file__).resolve().parents[2]


def workspace_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


class HybridService:
    def __init__(self, config: HybridConfig):
        self.config = config
        self.index_root = workspace_path(config.index_root)
        self.job_root = workspace_path(config.job_root)
        self.lock = threading.RLock()
        self.active_jobs = set()
        # Indexing/search share one slot: avoid loading multiple foundation models.
        self.compute = threading.Lock()
        from hybrid_v2.search_engine import SearchEngineCache

        self.engines = SearchEngineCache(config.search_cache_entries)

    def authorize(self, actor, *, engineer=False):
        if actor.tenant_id != self.config.tenant_id or self.config.access_scope_id not in actor.access_scope_ids:
            raise HTTPException(403, "SOURCE_FORBIDDEN")
        if engineer and not set(actor.roles) & {"Engineer", "Admin"}:
            raise HTTPException(403, "INDEX_ADMIN_REQUIRED")

    def source(self, value):
        path = workspace_path(value)
        roots = [workspace_path(root) for root in self.config.source_roots]
        if not any(path.is_relative_to(root) for root in roots):
            raise HTTPException(403, "SOURCE_FORBIDDEN")
        if not path.is_file():
            raise HTTPException(404, "SOURCE_UNAVAILABLE")
        return path

    @staticmethod
    def index_id(path):
        return hashlib.sha256(path.name.encode()).hexdigest()[:24]

    def _indexes(self):
        if not self.index_root.is_dir():
            return []
        return [
            p
            for p in sorted(self.index_root.iterdir())
            if p.is_dir() and p.resolve().is_relative_to(self.index_root) and (p / "index_meta.json").is_file()
        ]

    def resolve_index(self, index_id, actor):
        self.authorize(actor)
        for path in self._indexes():
            if self.index_id(path) == index_id:
                try:
                    meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
                    source = meta["video"]["path"]
                    if not isinstance(source, str):
                        raise ValueError("Invalid index source")
                except (OSError, ValueError, KeyError, TypeError):
                    raise HTTPException(409, {"code": "INDEX_CORRUPT"}) from None
                self.source(source)
                return path.resolve()
        raise HTTPException(404, "INDEX_NOT_FOUND")

    def list_indexes(self, actor):
        self.authorize(actor)
        from hybrid_v2.dense_metadata import DENSE_FORMAT
        from hybrid_v2.embeddinggemma2 import FORMAT as EG2_FORMAT
        from hybrid_v2.local_runtime import FORMAT

        result = []
        for path in self._indexes():
            try:
                meta = json.loads((path / "index_meta.json").read_text(encoding="utf-8"))
                if meta.get("format") not in (
                    FORMAT,
                    DENSE_FORMAT,
                    EG2_FORMAT,
                    "driveevidence-v1",
                    "driveevidence-v2",
                    "driveevidence-v3",
                    "driveevidence-v4",
                    "driveevidence-v5",
                    "driveevidence-v6",
                ):
                    continue
                self.source(meta["video"]["path"])
                result.append(
                    {
                        "index_id": self.index_id(path),
                        "name": path.name,
                        "model_family": "driveevidence_v6"
                        if meta.get("format") == "driveevidence-v6"
                        else "driveevidence_v5"
                        if meta.get("format") == "driveevidence-v5"
                        else "driveevidence_v4"
                        if meta.get("format") == "driveevidence-v4"
                        else "driveevidence_v3"
                        if meta.get("format") == "driveevidence-v3"
                        else "driveevidence_v2"
                        if meta.get("format") == "driveevidence-v2"
                        else "driveevidence"
                        if meta.get("format") == "driveevidence-v1"
                        else "embeddinggemma2"
                        if meta.get("format") == EG2_FORMAT
                        else "hybrid_v2",
                        "windows": meta["num_windows"],
                        "ready": "validate" in meta.get("phases", {}),
                        "caption_fallback_rate": meta.get("caption_stats", {}).get("fallback_rate"),
                    }
                )
            except (OSError, ValueError, KeyError, HTTPException):
                continue
        return result

    def search(self, index_id, query, actor, *, top_k=10, ranker=None, decision_provider=None, debug=False, deployment_mode=None):
        if not query.strip():
            raise HTTPException(422, {"code": "INVALID_QUERY"})
        path = self.resolve_index(index_id, actor)
        from hybrid_v2.local_index import read_meta

        if "validate" not in read_meta(path).get("phases", {}):
            raise HTTPException(409, {"code": "INDEX_NOT_READY"})
        if not self.compute.acquire(blocking=False):
            raise HTTPException(429, "HYBRID_BUSY")
        try:
            options = {"ranker": ranker or self.config.ranker}
            if deployment_mode is not None and options["ranker"] != "driveevidence_v6":
                raise ValueError("V6_DEPLOYMENT_MODE_REQUIRES_V6")
            if options["ranker"] == "driveevidence_v6" and (deployment_mode or self.config.deployment_mode) is not None:
                options["deployment_mode"] = deployment_mode or self.config.deployment_mode
            selected_decision = decision_provider or self.config.decision_provider
            if options["ranker"] in (
                "driveevidence_v2",
                "driveevidence_v3",
                "driveevidence_v4",
                "driveevidence_v5",
                "driveevidence_v6",
            ):
                selected_decision = "none"  # The separate query-only planner is mandatory in this mode.
            if options["ranker"] == "driveevidence":
                if selected_decision not in (None, "none"):
                    raise HTTPException(
                        503, {"code": "DRIVEEVIDENCE_REQUIRES_DECISION_PROVIDER_NONE", "fallback": False}
                    )
                selected_decision = "none"
            if options["ranker"] == "embeddinggemma2":
                from hybrid_v2.embeddinggemma2 import EmbeddingGemma2Error

                if selected_decision not in (None, "none"):
                    raise EmbeddingGemma2Error("EG2_REQUIRES_DECISION_PROVIDER_NONE")
                selected_decision = "none"
            if selected_decision is not None:
                options["decision_provider"] = selected_decision
            if selected_decision == "laya" or os.getenv("DECISION_PROVIDER") == "laya" and selected_decision != "none":
                options.update(
                    laya_device=self.config.laya_device,
                    laya_model_path=self.config.laya_model_path,
                    laya_batch_size=self.config.laya_batch_size,
                    laya_candidate_count=self.config.laya_candidate_count,
                )
            table = self.engines.search_table(path, query, **options)
            results = table["candidates"][:top_k]
            if table.get("decision", {}).get("state") == "NOT_FOUND" and not debug:
                results = []
            if not debug:
                results = [
                    {
                        k: v
                        for k, v in r.items()
                        if k
                        not in (
                            "constraint_result",
                            "laya_decision",
                            "entity_explanation",
                            "features",
                            "standardized_features",
                            "dense_cosines",
                            "exact_detail_hits",
                            "contradiction_reasons",
                            "ranks",
                        )
                    }
                    for r in results
                ]
            return {
                **({"request_id": table["retrieval"]["request_id"]}
                   if options["ranker"] == "driveevidence_v6" and table.get("retrieval", {}).get("request_id") else {}),
                "index_id": index_id,
                "model_family": "driveevidence_v6"
                if options["ranker"] == "driveevidence_v6"
                else "driveevidence_v5"
                if options["ranker"] == "driveevidence_v5"
                else "driveevidence_v4"
                if options["ranker"] == "driveevidence_v4"
                else "driveevidence_v3"
                if options["ranker"] == "driveevidence_v3"
                else "driveevidence_v2"
                if options["ranker"] == "driveevidence_v2"
                else "driveevidence"
                if options["ranker"] == "driveevidence"
                else "hybrid_v2",
                "ranker": results[0]["ranker"] if results else ranker or self.config.ranker,
                "media_url": f"/api/v1/hybrid/indexes/{index_id}/media",
                "results": results,
                **(
                    {"decision": {k: v for k, v in table["decision"].items() if k != "health"}}
                    if "decision" in table
                    else {}
                ),
                **(
                    {"retrieval": table["retrieval"]}
                    if debug
                    and options["ranker"]
                    in (
                        "driveevidence_v2",
                        "driveevidence_v3",
                        "driveevidence_v4",
                        "driveevidence_v5",
                        "driveevidence_v6",
                    )
                    else {}
                ),
            }
        except Exception as exc:
            if options.get("ranker") == "driveevidence_v6" and isinstance(exc, (ValueError, RuntimeError, OSError)):
                raise HTTPException(
                    503, {"code": "DRIVEEVIDENCE_V6_NOT_READY", "detail": str(exc), "fallback": False}
                ) from None
            if options.get("ranker") == "driveevidence_v5" and isinstance(exc, (ValueError, RuntimeError, OSError)):
                raise HTTPException(
                    503, {"code": "DRIVEEVIDENCE_V5_NOT_READY", "detail": str(exc), "fallback": False}
                ) from None
            if options.get("ranker") in ("driveevidence_v3", "driveevidence_v4") and isinstance(
                exc, (ValueError, RuntimeError, OSError)
            ):
                raise HTTPException(
                    503,
                    {
                        "code": "DRIVEEVIDENCE_V4_FAILED"
                        if options.get("ranker") == "driveevidence_v4"
                        else "DRIVEEVIDENCE_V3_FAILED",
                        "detail": str(exc),
                        "fallback": False,
                    },
                ) from None
            if options.get("ranker") == "driveevidence_v2" and isinstance(exc, (ValueError, RuntimeError, OSError)):
                raise HTTPException(
                    503, {"code": "DRIVEEVIDENCE_V2_FAILED", "detail": str(exc), "fallback": False}
                ) from None
            from hybrid_v2.embeddinggemma2 import EmbeddingGemma2Error

            if isinstance(exc, EmbeddingGemma2Error):
                raise HTTPException(
                    503,
                    {
                        "code": exc.code,
                        "ranker": "driveevidence" if options.get("ranker") == "driveevidence" else "embeddinggemma2",
                        "detail": str(exc),
                        "fallback": False,
                    },
                ) from None
            if options.get("ranker") == "driveevidence" and isinstance(exc, (ValueError, RuntimeError, OSError)):
                raise HTTPException(
                    503, {"code": "DRIVEEVIDENCE_FAILED", "detail": str(exc), "fallback": False}
                ) from None
            from hybrid_v2.decision_provider import DecisionError

            if isinstance(exc, DecisionError):
                raise HTTPException(
                    503, {"code": exc.code, "decision_provider": "laya", "degraded": True, "fallback": False}
                ) from None
            raise
        finally:
            self.compute.release()

    def private_media(self, index_id, actor):
        path = self.resolve_index(index_id, actor)
        from hybrid_v2.privacy_cli import video_for_index
        from src.privacy.media import private_full_media

        try:
            with self.lock:
                return private_full_media(video_for_index(path), path / "privacy")
        except (ValueError, RuntimeError, OSError):
            import logging

            logging.getLogger(__name__).exception("Private media unavailable")
            code = (
                "PRIVACY_CACHE_MISSING"
                if not (path / "privacy/privacy_meta.json").exists()
                else "PRIVACY_CACHE_INVALID"
            )
            raise HTTPException(409, {"code": code, "message": "PRIVACY_PREVIEW_UNAVAILABLE"}) from None

    def readiness(self, actor, index_id=None, operation="search"):
        self.authorize(actor)
        if operation == "index":
            from hybrid_v2.qwen_provider import QwenError, create_provider

            provider = create_provider(self.config.model_dump())
            try:
                return {"ready": True, "operation": operation, "qwen": provider.healthcheck()}
            except QwenError as exc:
                raise HTTPException(503, {"code": exc.code, "operation": operation}) from None
            finally:
                provider.close()
        if not index_id:
            if self.config.ranker == "driveevidence_v6":
                indexes = [i for i in self.list_indexes(actor) if i["model_family"] == "driveevidence_v6"]
                if not indexes:
                    return {
                        "ready": False,
                        "service_state": "NOT_READY",
                        "ranker": "driveevidence_v6",
                        "qwen_required": False,
                    }
                return self.readiness(actor, indexes[0]["index_id"], operation)
            if self.config.ranker == "driveevidence_v5":
                indexes = [i for i in self.list_indexes(actor) if i["model_family"] == "driveevidence_v5"]
                if not indexes:
                    return {
                        "ready": False,
                        "service_state": "NOT_READY",
                        "ranker": "driveevidence_v5",
                        "qwen_required": False,
                    }
                return self.readiness(actor, indexes[0]["index_id"], operation)
            if self.config.ranker == "driveevidence_v4":
                from hybrid_v2.embeddinggemma2 import runtime_readiness

                runtime_readiness()
                return {
                    "ready": any(
                        i["ready"] and i["model_family"] == "driveevidence_v4" for i in self.list_indexes(actor)
                    ),
                    "operation": "search",
                    "qwen_required": False,
                    "ranker": "driveevidence_v4",
                    "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
                }
            if self.config.ranker == "driveevidence_v3":
                from hybrid_v2.embeddinggemma2 import runtime_readiness

                runtime_readiness()
                return {
                    "ready": any(
                        i["ready"] and i["model_family"] == "driveevidence_v3" for i in self.list_indexes(actor)
                    ),
                    "operation": "search",
                    "qwen_required": False,
                    "ranker": "driveevidence_v3",
                    "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
                }
            if self.config.ranker == "driveevidence_v2":
                from hybrid_v2.embeddinggemma2 import runtime_readiness

                runtime_readiness()
                return {
                    "ready": any(
                        i["ready"] and i["model_family"] == "driveevidence_v2" for i in self.list_indexes(actor)
                    ),
                    "operation": "search",
                    "qwen_required": False,
                    "ranker": "driveevidence_v2",
                    "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
                }
            if self.config.ranker == "driveevidence":
                from hybrid_v2.embeddinggemma2 import runtime_readiness

                runtime_readiness()
                return {
                    "ready": any(i["ready"] and i["model_family"] == "driveevidence" for i in self.list_indexes(actor)),
                    "operation": "search",
                    "qwen_required": False,
                    "ranker": "driveevidence",
                    "decision": {"provider": "none", "ready": True},
                }
            if self.config.ranker == "embeddinggemma2":
                from hybrid_v2.embeddinggemma2 import EmbeddingGemma2Error, runtime_readiness

                try:
                    runtime_readiness()
                except EmbeddingGemma2Error as exc:
                    raise HTTPException(
                        503, {"code": exc.code, "ranker": "embeddinggemma2", "fallback": False}
                    ) from None
                return {
                    "ready": any(
                        i["ready"] and i["model_family"] == "embeddinggemma2" for i in self.list_indexes(actor)
                    ),
                    "operation": "search",
                    "qwen_required": False,
                    "ranker": "embeddinggemma2",
                    "decision": {"provider": "none", "ready": True},
                }
            if (self.config.decision_provider or os.getenv("DECISION_PROVIDER", "none")) == "laya":
                return {
                    "ready": False,
                    "operation": "search",
                    "qwen_required": False,
                    "decision": {"provider": "laya", "ready": False, "code": "LAYA_INDEX_REQUIRED", "fallback": False},
                }
            return {
                "ready": any(i["ready"] for i in self.list_indexes(actor)),
                "operation": "search",
                "qwen_required": False,
            }
        path = self.resolve_index(index_id, actor)
        from hybrid_v2.embeddinggemma2 import FORMAT as EG2_FORMAT
        from hybrid_v2.local_index import read_meta

        if read_meta(path).get("format") == "driveevidence-v6":
            try:
                self.engines.search_table(
                    path, "a road scene with vehicles", ranker="driveevidence_v6", decision_provider="none",
                    **({"deployment_mode": self.config.deployment_mode} if self.config.deployment_mode else {})
                )
                engine = self.engines.entries[str(path.resolve())][1]
                return {
                    **engine.health,
                    "index_id": index_id,
                    "operation": operation,
                    "ranker": "driveevidence_v6",
                    "qwen_required": False,
                }
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, {"code": "DRIVEEVIDENCE_V6_NOT_READY", "detail": str(exc)}) from None
        if read_meta(path).get("format") == "driveevidence-v5":
            try:
                # Startup loads/warmups complete before readiness or normal traffic.
                self.engines.search_table(
                    path, "a road scene with vehicles", ranker="driveevidence_v5", decision_provider="none"
                )
                engine = self.engines.entries[str(path.resolve())][1]
                return {
                    **engine.health,
                    "index_id": index_id,
                    "operation": operation,
                    "ranker": "driveevidence_v5",
                    "qwen_required": False,
                }
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, {"code": "DRIVEEVIDENCE_V5_NOT_READY", "detail": str(exc)}) from None
        if read_meta(path).get("format") == "driveevidence-v4":
            from hybrid_v2.driveevidence_v4 import load_index
            from hybrid_v2.embeddinggemma2 import runtime_readiness

            try:
                load_index(path)
                runtime_readiness()
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, {"code": "DRIVEEVIDENCE_V4_INDEX_NOT_READY", "detail": str(exc)}) from None
            return {
                "ready": True,
                "operation": operation,
                "index_id": index_id,
                "ranker": "driveevidence_v4",
                "qwen_required": False,
                "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
            }
        if read_meta(path).get("format") == "driveevidence-v3":
            from hybrid_v2.driveevidence_v3 import load_index
            from hybrid_v2.embeddinggemma2 import runtime_readiness

            try:
                load_index(path)
                runtime_readiness()
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, {"code": "DRIVEEVIDENCE_V3_INDEX_NOT_READY", "detail": str(exc)}) from None
            return {
                "ready": True,
                "operation": operation,
                "index_id": index_id,
                "ranker": "driveevidence_v3",
                "qwen_required": False,
                "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
            }
        if read_meta(path).get("format") == "driveevidence-v2":
            from hybrid_v2.driveevidence_v2 import load_index
            from hybrid_v2.embeddinggemma2 import runtime_readiness

            try:
                load_index(path)
                runtime_readiness()
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(503, {"code": "DRIVEEVIDENCE_V2_INDEX_NOT_READY", "detail": str(exc)}) from None
            return {
                "ready": True,
                "operation": operation,
                "index_id": index_id,
                "ranker": "driveevidence_v2",
                "qwen_required": False,
                "decision": {"provider": "laya_query_planner", "fallback": "embeddinggemma2"},
            }
        if read_meta(path).get("format") == "driveevidence-v1":
            from hybrid_v2.driveevidence import load_index
            from hybrid_v2.embeddinggemma2 import runtime_readiness

            try:
                load_index(path)
                runtime_readiness()
            except (ValueError, RuntimeError, OSError) as exc:
                raise HTTPException(
                    503, {"code": "DRIVEEVIDENCE_INDEX_NOT_READY", "detail": str(exc), "fallback": False}
                ) from None
            return {
                "ready": True,
                "operation": operation,
                "index_id": index_id,
                "ranker": "driveevidence",
                "qwen_required": False,
                "decision": {"provider": "none", "ready": True},
            }
        if read_meta(path).get("format") == EG2_FORMAT:
            from hybrid_v2.embeddinggemma2 import EmbeddingGemma2Error, runtime_readiness
            from hybrid_v2.embeddinggemma2_index import validate_index as validate_eg2

            try:
                validate_eg2(path)
                runtime_readiness()
            except EmbeddingGemma2Error as exc:
                raise HTTPException(503, {"code": exc.code, "ranker": "embeddinggemma2", "fallback": False}) from None
            return {
                "ready": True,
                "operation": operation,
                "index_id": index_id,
                "ranker": "embeddinggemma2",
                "qwen_required": False,
                "decision": {"provider": "none", "ready": True},
            }
        from hybrid_v2.local_index import validate_index

        validate_index(path)
        # §25: an incomplete V4.1 pilot index must never be advertised as ready.
        publication = path / "metadata/publication.json"
        if publication.is_file() and json.loads(publication.read_text(encoding="utf-8")).get("status") == "UNPUBLISHED":
            raise HTTPException(503, {"code": "INDEX_UNPUBLISHED", "operation": operation, "fallback": False})
        from hybrid_v2.decision_provider import DecisionConfig, DecisionError

        try:
            config = DecisionConfig.resolve(
                provider=self.config.decision_provider,
                device=self.config.laya_device,
                model_path=self.config.laya_model_path,
                batch_size=self.config.laya_batch_size,
                candidate_count=self.config.laya_candidate_count,
            )
        except (ValueError, TypeError):
            raise HTTPException(503, {"code": "DECISION_CONFIG_INVALID", "fallback": False}) from None
        decision = {"provider": "none", "ready": True}
        if config.provider == "laya":
            if self.config.ranker != "dense-entity":
                raise HTTPException(
                    503,
                    {"code": "LAYA_REQUIRES_METADATA_V4_DENSE_ENTITY", "decision_provider": "laya", "fallback": False},
                )
            if not self.compute.acquire(blocking=False):
                raise HTTPException(429, "HYBRID_BUSY")
            try:
                decision = self.engines.decision_readiness(path, config)
            except DecisionError as exc:
                raise HTTPException(503, {"code": exc.code, "decision_provider": "laya", "fallback": False}) from None
            finally:
                self.compute.release()
        return {
            "ready": decision["ready"],
            "operation": "search",
            "index_id": index_id,
            "qwen_required": False,
            "decision": decision,
        }

    def private_clip(self, index_id, actor, start, end):
        import math

        path = self.resolve_index(index_id, actor)
        from hybrid_v2.local_index import read_meta

        meta = read_meta(path)
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= meta["video"]["duration_sec"]:
            raise HTTPException(422, {"code": "INVALID_RANGE"})
        from src.privacy.render import render_private_clip

        try:
            target = self.job_root / "private_exports" / (uuid4().hex + ".mp4")
            render_private_clip(meta["video"]["path"], path / "privacy", target, start, end)
            return target
        except (ValueError, RuntimeError, OSError):
            code = (
                "PRIVACY_CACHE_MISSING"
                if not (path / "privacy/privacy_meta.json").exists()
                else "PRIVACY_CACHE_INVALID"
            )
            raise HTTPException(409, {"code": code}) from None

    def _write_job(self, job):
        self.job_root.mkdir(parents=True, exist_ok=True)
        path = self.job_root / (job["job_id"] + ".json")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(job, indent=2), encoding="utf-8")
        temporary.replace(path)

    def get_job(self, job_id, actor):
        self.authorize(actor, engineer=True)
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise HTTPException(404, "JOB_NOT_FOUND")
        with self.lock:
            path = self.job_root / (job_id + ".json")
            if not path.is_file():
                raise HTTPException(404, "JOB_NOT_FOUND")
            job = json.loads(path.read_text(encoding="utf-8"))
        if job["actor_id"] != actor.actor_id and "Admin" not in actor.roles:
            raise HTTPException(403, "SOURCE_FORBIDDEN")
        with self.lock:
            if job["status"] in {"queued", "running"} and job_id not in self.active_jobs:
                job = {**job, "status": "failed", "error": "API restarted before indexing finished; submit a new job."}
                self._write_job(job)
        metadata = self.index_root / ("api_" + job_id) / "index_meta.json"
        if metadata.is_file():
            meta = json.loads(metadata.read_text(encoding="utf-8"))
            job["completed_phases"] = list(meta.get("phases", {}))
        return job

    def prepare_job(self, video, actor, *, max_windows=None):
        self.authorize(actor, engineer=True)
        source = self.source(video)
        if not self.compute.acquire(blocking=False):
            raise HTTPException(429, "HYBRID_BUSY")
        try:
            job_id = uuid4().hex
            target = self.index_root / ("api_" + job_id)
            job = {
                "job_id": job_id,
                "actor_id": actor.actor_id,
                "status": "queued",
                "index_id": self.index_id(target),
                "model_family": "hybrid_v2",
            }
            with self.lock:
                self._write_job(job)
                self.active_jobs.add(job_id)
            return job, self.command(source, target, max_windows)
        except Exception:
            self.compute.release()
            raise

    def command(self, video, output, max_windows=None):
        command = [
            sys.executable,
            "-B",
            "-m",
            "hybrid_v2.laptop_cli",
            "index",
            "--video",
            str(video),
            "--output",
            str(output),
            "--local-files-only",
        ]
        for name in ("stage2b_checkpoint", "stage3a_checkpoint", "stage5_checkpoint", "hf_cache_dir", "detector"):
            command += ["--" + name.replace("_", "-"), str(workspace_path(getattr(self.config, name)))]
        for name in (
            "device",
            "precision",
            "siglip_revision",
            "bge_model",
            "bge_revision",
            "qwen_config",
        ):
            value = getattr(self.config, name)
            if value is not None:
                command += ["--" + name.replace("_", "-"), str(value)]
        from hybrid_v2.qwen_provider import resolve_provider_config

        qwen = resolve_provider_config(self.config.model_dump())
        command += ["--qwen-mode", qwen.mode, "--qwen-base-url", qwen.base_url]
        if qwen.model:
            command += ["--vlm-model", qwen.model]
        command += [
            "--visual-batch-size",
            "1",
            "--vision-chunk-size",
            "4",
            "--yolo-device",
            "cpu",
            "--bge-device",
            "cpu",
        ]
        if max_windows is not None:
            command += ["--max-windows", str(max_windows)]
        if self.config.privacy_on_ingest:
            command += ["--privacy", "--privacy-profile", self.config.privacy_profile]
        return command

    def run_job(self, job, command):
        try:
            self.engines.release_embeddinggemma2()
            with self.lock:
                job = {**job, "status": "running"}
                self._write_job(job)
            environment = os.environ.copy()
            environment["OMP_NUM_THREADS"] = str(self.config.cpu_threads)
            environment["MKL_NUM_THREADS"] = str(self.config.cpu_threads)
            with (self.job_root / (job["job_id"] + ".log")).open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
            job["status"] = "ready" if completed.returncode == 0 else "failed"
            if completed.returncode:
                job["error"] = "Hybrid indexing failed; inspect the job log on the server."
        except Exception:
            job["status"] = "failed"
            job["error"] = "INDEXING_FAILED"
        finally:
            try:
                with self.lock:
                    self._write_job(job)
                    self.active_jobs.discard(job["job_id"])
            finally:
                self.compute.release()
