"""Persistent serving of the exact best measured V6 flow; safe EG2 default."""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from pathlib import Path

import numpy as np

from .driveevidence_v6 import candidate_evidence, complexity, stable_order
from .driveevidence_v6_rerank import decide, local_score
from .driveevidence_v6_selected import DEPLOYMENT_MODES, POLICY_HASH, REPRESENTATION_HASH, load_index, selected_config
from .driveevidence_v6_worker import DriveEvidenceV6Worker
from .embeddinggemma2 import DIMENSION, REVISION, ROOT, digest, identity


class DriftMonitor:
    def __init__(self, profile=None):
        self.profile = profile or {}
        self.records = deque(maxlen=self.profile.get("window", 32))
        self.alarm, self.reason = False, None

    def observe(self, record):
        self.records.append(record)
        if len(self.records) < self.records.maxlen:
            return self.alarm
        for name, reference in self.profile.get("features", {}).items():
            values = [r[name] for r in self.records if r.get(name) is not None]
            if len(values) < self.records.maxlen // 2:
                continue
            fraction = sum(v < reference["min"] - reference.get("padding", .1) or
                           v > reference["max"] + reference.get("padding", .1) for v in values) / len(values)
            if fraction > self.profile.get("outside_fraction", .5):
                self.alarm, self.reason = True, name
        return self.alarm


def resolve_deployment_mode(value=None):
    mode = value if value is not None else os.getenv("DRIVEEVIDENCE_V6_DEPLOYMENT_MODE", "eg2_only")
    if mode not in DEPLOYMENT_MODES:
        raise ValueError("INVALID_V6_DEPLOYMENT_MODE")
    return mode


class DriveEvidenceV6SearchEngine:
    def __init__(self, index_path, worker_factory=None, planner=None, *, deployment_mode=None,
                 telemetry_path=None, privacy_guard=None):
        self.index_path = Path(index_path)
        self.worker_factory = worker_factory or DriveEvidenceV6Worker
        self.planner = planner  # Compatibility only; selected serving never loads/calls Laya.
        self.mode = resolve_deployment_mode(deployment_mode)
        self.telemetry_path = Path(telemetry_path) if telemetry_path else None
        self.privacy_guard = privacy_guard
        self.loaded = self.worker = None
        self.lock = threading.RLock()
        self.query_cache = OrderedDict()
        self.load_count = 0
        self.shadow_config = selected_config()
        self.health = {k: False for k in ("ready", "eg2_model_loaded", "global_index_loaded", "v6_index_loaded",
            "source_hashes_valid", "schema_valid", "policy_hash_valid", "representation_hash_valid",
            "privacy_ready", "warmup_complete", "index_loaded", "models_loaded", "hashes_valid")}
        self.health.update(deployment_mode=self.mode, service_state="NOT_READY")

    def _health(self, mode, fallback=False):
        self.health.update(deployment_mode=mode, service_state="DEGRADED_EG2_ONLY" if fallback else
            {"eg2_only": "READY_EG2_ONLY", "shadow": "READY_V6_SHADOW", "active": "READY_V6_ACTIVE"}[mode])

    def load(self):
        with self.lock:
            if self.loaded is not None:
                return self
            started = time.perf_counter()
            try:
                self.loaded = load_index(self.index_path)
                for name, expected in self.loaded["meta"].get("model_hashes", {}).items():
                    if "laya" in name.casefold():
                        continue
                    path = (ROOT / name).resolve()
                    if not path.is_relative_to(ROOT) or digest(path) != expected:
                        raise ValueError("V6_MODEL_HASH_MISMATCH")
                self.worker = self.worker_factory(self.loaded["meta"]["config"])
                self.worker.load()
                from src.privacy.media import private_full_media
                privacy_ok = self.privacy_guard() if self.privacy_guard is not None else callable(private_full_media)
                if not privacy_ok:
                    raise ValueError("V6_PRIVACY_GUARD_UNAVAILABLE")
                self.load_count += 1
                warmup = time.perf_counter()
                vector, _ = self.worker.encode_text("a road scene with vehicles")
                self._validate_vector(vector)
                self.warmup_seconds = time.perf_counter() - warmup
                self.startup_seconds = time.perf_counter() - started
                self.drift = DriftMonitor(self.loaded["meta"].get("drift_profile"))
                valid = not self.loaded["evidence_fallback"]
                self.health.update(ready=True, eg2_model_loaded=True, global_index_loaded=True,
                    v6_index_loaded=valid, source_hashes_valid=True, schema_valid=valid,
                    policy_hash_valid=valid, representation_hash_valid=valid, privacy_ready=True,
                    warmup_complete=True, index_loaded=True, models_loaded=True, hashes_valid=valid)
                self._health(self.mode, not valid)
                return self
            except Exception:
                self.close()
                raise

    @staticmethod
    def _validate_vector(vector):
        v = np.asarray(vector, np.float32)
        if v.shape != (DIMENSION,) or not np.isfinite(v).all() or not np.isclose(np.linalg.norm(v), 1., atol=.003):
            raise ValueError("V6_INVALID_EG2_TEXT_VECTOR")
        # Frozen encoder output is already normalized; do not divide it a second time.
        return v

    def cache_identity(self, query, deployment_mode=None):
        mode = resolve_deployment_mode(deployment_mode or self.mode)
        meta = self.loaded["meta"]
        return identity({"literal_query": query, "EG2_revision": REVISION,
            "EG2_config": meta["config"], "representation_hash": meta.get("representation_sha256", REPRESENTATION_HASH),
            "scorer_config_hash": POLICY_HASH, "experimental_policy": meta.get("experimental_policy"),
            "serving_manifest": self.loaded.get("serving_manifest"), "deployment_mode": mode})

    def _query(self, query, mode, timing):
        begun = time.perf_counter()
        key = self.cache_identity(query, mode)
        saved = self.query_cache.get(key)
        timing["cache"] = time.perf_counter() - begun
        timing.update(cache_hit=saved is not None, Laya_calls=0, Laya=0., factor_text=0.,
                      global_text=0., EG2_text_encode=0., peak_EG2_VRAM_bytes=0)
        if saved is not None:
            self.query_cache.move_to_end(key)
            return saved
        begun = time.perf_counter()
        vector, metrics = self.worker.encode_text(query)
        timing["global_text"] = timing["EG2_text_encode"] = time.perf_counter() - begun
        begun = time.perf_counter()
        vector = self._validate_vector(vector)
        timing["normalization"] += time.perf_counter() - begun
        saved = {"global_vector": vector, "vectors": {query: vector},
                 "atoms": [{"type": "GLOBAL", "text": query, "weight": 1.}], "route": complexity(query)}
        timing["peak_EG2_VRAM_bytes"] = metrics.get("peak_VRAM_bytes", 0)
        self.query_cache[key] = saved
        while len(self.query_cache) > 256:
            self.query_cache.popitem(last=False)
        return saved

    def search_table(self, query, *, ranker="driveevidence_v6", decision_provider=None,
                     deployment_mode=None, **unused):
        started = time.perf_counter()
        mode = resolve_deployment_mode(deployment_mode or self.mode)
        if ranker != "driveevidence_v6" or decision_provider not in (None, "none", "laya") or not isinstance(query, str) or not query.strip() or len(query) > 1000:
            raise ValueError("INVALID_V6_QUERY")
        timing = {"normalization": time.perf_counter() - started, "evidence_fetch": 0.,
                  "late_interaction": 0., "pairwise_protection": 0.}
        with self.lock:
            self.load()
            data = self._query(query, mode, timing)
            begun = time.perf_counter()
            scores = self.loaded["global_vectors"] @ data["global_vector"]
            if not np.isfinite(scores).all():
                self.health.update(ready=False, service_state="NOT_READY")
                raise ValueError("V6_INVALID_EG2_GLOBAL_SCORES")
            base = stable_order(scores)
            timing["global_search"] = timing["candidate_search"] = time.perf_counter() - begun
            order, proposed = base, base
            diagnostics, fallback_reason = [], None
            decision = {"state": "KEEP_EG2", "reason_code": "EG2_ONLY", "swap": False, "proposal": None}
            if self.loaded["evidence_fallback"]:
                fallback_reason = "INDEX_EVIDENCE_MISSING"
            elif self.drift.alarm:
                fallback_reason = "DRIFT_FALLBACK"
            elif mode != "eg2_only":
                try:
                    experimental = self.loaded["meta"].get("experimental_policy")
                    if experimental and (experimental.get("qualified") is not False or
                            experimental.get("config") != selected_config() or experimental.get("config_sha256") != POLICY_HASH):
                        raise ValueError("V6_SELECTED_POLICY_HASH_MISMATCH")
                    if self.shadow_config != selected_config():
                        raise ValueError("V6_SELECTED_CONFIG_CHANGED")
                    local = {}
                    for row in base[:5]:
                        row = int(row)
                        begun = time.perf_counter()
                        evidence = candidate_evidence(self.loaded, row)
                        timing["evidence_fetch"] += time.perf_counter() - begun
                        begun = time.perf_counter()
                        result = local_score(evidence, data["atoms"], data["vectors"], self.shadow_config)
                        timing["late_interaction"] += time.perf_counter() - begun
                        if not result["valid"] or not np.isfinite(result["score"]):
                            raise ValueError("V6_INVALID_LOCAL_SCORE")
                        local[row] = result
                        diagnostics.append({"candidate_row": row, **result})
                    begun = time.perf_counter()
                    proposed, decision = decide(scores, local, self.shadow_config)
                    timing["pairwise_protection"] = time.perf_counter() - begun
                    if (len(proposed) != len(base) or set(proposed[:5]) != set(base[:5]) or
                            not np.array_equal(proposed[5:], base[5:])):
                        raise ValueError("V6_INVALID_BOUNDED_PROPOSAL")
                    if mode == "active":
                        order = proposed
                except Exception as exc:
                    fallback_reason = "V6_RUNTIME_FAILURE:" + type(exc).__name__
                    order = proposed = base
                    decision = {"state": "KEEP_EG2", "reason_code": fallback_reason, "swap": False, "proposal": None}
            margin = float(scores[base[0]] - scores[base[1]]) if len(base) > 1 else None
            proposal = decision.get("proposal")
            late_margin = proposal["late_advantage"] if proposal else None
            try:
                if self.drift.observe({"EG2_margin": margin, "late_margin": late_margin,
                        "complex_route": int(data["route"]["route"] == "complex"),
                        "rerank_rate": int(decision["swap"]), "latency": time.perf_counter() - started}):
                    fallback_reason = "DRIFT_FALLBACK"
                    order = proposed = base
            except Exception:
                fallback_reason = "V6_DRIFT_MONITOR_FAILURE"
                order = proposed = base
            timing["evidence_late_interaction"] = timing["evidence_fetch"] + timing["late_interaction"] + timing["pairwise_protection"]
            request_id = str(uuid.uuid4())
            meta = self.loaded["meta"]
            index_version = identity(self.loaded.get("serving_manifest", meta.get("hashes", {})))
            def ids(ranking):
                return [self.loaded["rows"][int(i)]["clip_id"] for i in ranking[:5]]
            fallback = fallback_reason is not None
            reranked = mode == "active" and not fallback and bool(decision["swap"])
            telemetry = {"request_id": request_id, "query_hash": identity(query), "deployment_mode": mode,
                "index_version": index_version, "policy_version": POLICY_HASH, "eg2_top5": ids(base),
                "v6_top5": ids(proposed), "top1_would_change": bool(proposed[0] != base[0]),
                "returned_top1": self.loaded["rows"][int(order[0])]["clip_id"], "changed_top1": reranked,
                "promotion": {"rank_from": int(np.flatnonzero(base == proposed[0])[0]) + 1, "rank_to": 1} if proposed[0] != base[0] else None,
                "global_margin": margin, "late_margin": late_margin, "pairwise_decision": decision,
                "fallback": fallback, "fallback_reason": fallback_reason}
            if self.telemetry_path is not None:
                try:
                    self.telemetry_path.parent.mkdir(parents=True, exist_ok=True)
                    with self.telemetry_path.open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps({**telemetry, "timing": timing,
                            "latency": time.perf_counter() - started}) + "\n")
                except Exception:
                    fallback_reason = "V6_TELEMETRY_FAILURE"
                    fallback, reranked, order = True, False, base
            begun = time.perf_counter()
            candidates = []
            for rank, row in enumerate(order, 1):
                clip = self.loaded["rows"][int(row)]
                candidates.append({**{k: clip[k] for k in ("clip_id", "source_row", "source_clip_id", "segment_id", "start_sec", "end_sec", "day_night")},
                    "row": int(row), "rank": rank, "ranker": ranker, "score": float(scores[row]),
                    "final_score": decision.get("scored_candidates", {}).get(int(row), {}).get("total", float(scores[row])) if reranked else float(scores[row])})
            timing["response_assembly"] = timing["serialization"] = time.perf_counter() - begun
            timing["total"] = time.perf_counter() - started
            self._health(mode, fallback)
            return {"query": query, "format": "driveevidence-v6-candidates", "index": str(self.index_path), "candidates": candidates,
                "decision": {"state": "SAFE_RERANK" if reranked else "KEEP_EG2", "reason_code": fallback_reason or decision["reason_code"],
                    "user_ranking": "V6" if mode == "active" and not fallback else "EG2", "deployment_mode": mode,
                    "not_found_enabled": False, "provider": "none", "fallback": fallback, "fallback_reason": fallback_reason},
                "retrieval": {"request_id": request_id, "retrieval_mode": ranker, "deployment_mode": mode,
                    "reranked": reranked, "eg2_top5": ids(base), "v6_top5": ids(proposed), "fallback": fallback,
                    "fallback_reason": fallback_reason, "index_version": index_version, "policy_version": POLICY_HASH,
                    "representation_version": REPRESENTATION_HASH, "route": data["route"],
                    "planner": {"Laya_calls": 0, "cache_hit": timing["cache_hit"]}, "diagnostic": diagnostics,
                    "shadow_decision": decision, "timing": timing, "warm_seconds": timing["total"],
                    "engine_load_count": self.load_count, "Qwen_calls": 0, "Laya_calls": 0,
                    "candidate_level_Laya_calls": 0, "external_model_API_calls": 0,
                    "detector_calls": 0, "tracker_calls": 0, "video_encoder_calls": 0, "image_encoder_calls": 0}}

    def close(self):
        if self.worker is not None:
            self.worker.close()
        self.worker = self.loaded = None
        self.query_cache.clear()
        self.health.update({k: False for k in self.health if isinstance(self.health[k], bool)})
        self.health["service_state"] = "NOT_READY"
