"""Persistent, bounded, Qwen-independent search. Validate immutable inputs once at load."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from .dense_metadata import DENSE_FORMAT, FIELDS
from .dense_search import query_texts, score_dense
from .entity_search import rerank_entity_candidates
from .intent_parser import parse_intent, parse_query
from .local_index import ARTIFACTS, array, load_stage5, read_meta, rows, validate_index
from .local_video_search import rank_candidates, score_candidates


class SearchEngine:
    def __init__(self, index_path, *, bge_factory=None, siglip_factory=None, validator=validate_index):
        self.index_path = Path(index_path).resolve()
        self.bge_factory, self.siglip_factory, self.validator = bge_factory, siglip_factory, validator
        self.lock = threading.RLock()
        self.loaded = None
        self.load_count = self.query_count = 0
        self.cold_load_seconds = None
        self.decision_providers = {}

    def load(self):
        with self.lock:
            if self.loaded is not None:
                return self
            started = time.perf_counter()
            self.validation = self.validator(self.index_path)
            meta = read_meta(self.index_path)
            arrays = {
                name: array(self.index_path, name)
                for phase in ("visual", "motion", "fuse", "frame-project", "text-index")
                for name in ARTIFACTS[phase]
            }
            records = []
            if meta["format"] == DENSE_FORMAT:
                records = rows(self.index_path / "metadata/dense_metadata.jsonl")
                arrays.update({f"text/dense_{f}.npy": array(self.index_path, f"text/dense_{f}.npy") for f in FIELDS})
            from .entity_constraints import normalize_entities

            entities = [normalize_entities(r) for r in records]
            v4_features = None
            if meta.get("v4_native") or meta.get("v4_1_native") or meta.get("v4_2_native"):
                import json

                from .v4_matcher import prepare_features

                store = "metadata/v4" if meta.get("v4_native") else "metadata/v4_2/final" if meta.get("v4_2_native") else "metadata/v4_1/final"
                # A partial V4.1 index declares non-searchable placeholder rows in
                # metadata/unpublished.json; substitute an inert schema-shaped
                # record for those rows (they never matched FINAL_SUCCESS), so the
                # matcher still consumes a full row-aligned feature list without
                # inventing facts. Strictly absent/undeclared rows still fail loud.
                unpublished = set()
                manifest_path = self.index_path / "metadata/unpublished.json"
                if (meta.get("v4_1_native") or meta.get("v4_2_native")) and manifest_path.is_file():
                    unpublished = set(json.loads(manifest_path.read_text())["rows"])
                v4_features = []
                for r in records:
                    row_file = self.index_path / f"{store}/rows/{r['row']:06d}.json"
                    if row_file.is_file():
                        value = json.loads(row_file.read_text())
                    elif f"{r['row']:06d}" in unpublished:
                        from .metadata_v4_1 import placeholder_final

                        value = placeholder_final()
                    else:
                        raise ValueError("V4.1 metadata row missing: " + f"{r['row']:06d}")
                    v4_features.append(prepare_features(value))
            from .query_encoder import SigLIPQueryEncoder
            from .text_encoder import BGEEncoder

            model_started = time.perf_counter()
            bge = (self.bge_factory or BGEEncoder)(meta["models"]["bge"]["path"], "cpu", local_files_only=True)
            bge_loaded = time.perf_counter()
            try:
                sig = (self.siglip_factory or SigLIPQueryEncoder)(
                    meta["checkpoints"]["stage2b"]["path"],
                    "cpu",
                    local_files_only=True,
                    model_path=meta["models"]["siglip"]["path"],
                    text_only=True,
                )
                sig_loaded = time.perf_counter()
                ck = load_stage5(meta["checkpoints"]["stage5"]["path"])
                from .stage5_model import Stage5HybridReranker

                cfg = ck["config"]
                stage5 = Stage5HybridReranker(
                    18, int(cfg["hidden"]), float(cfg["dropout"]), float(cfg["residual_scale"])
                )
                stage5.load_state_dict(ck["model_state_dict"], strict=True)
                stage5.eval()
                loaded = {
                    "meta": meta,
                    "arrays": arrays,
                    "records": records,
                    "entities": entities,
                    "v4_features": v4_features,
                    "manifest": rows(self.index_path / "manifest.jsonl"),
                    "captions": rows(self.index_path / "captions/captions.jsonl"),
                    "stage5_checkpoint": ck,
                    "stage5_model": stage5,
                }
            except Exception:
                del bge
                raise
            self.bge, self.sig, self.loaded = bge, sig, loaded
            self.model_load_timing = {
                "bge_load_seconds": bge_loaded - model_started,
                "siglip_load_seconds": sig_loaded - bge_loaded,
            }
            self.load_count += 1
            self.cold_load_seconds = time.perf_counter() - started
            return self

    @torch.inference_mode()
    def search_table(
        self,
        query,
        *,
        ranker="dense-entity",
        caption_k=100,
        stage3_k=100,
        rerank_k=50,
        candidate_k=50,
        temporal_iou_threshold=0.5,
        no_temporal_dedup=False,
        intent_mode="rule",
        decision_provider=None,
        laya_device=None,
        laya_model_path=None,
        laya_batch_size=None,
        laya_candidate_count=None,
    ):
        if (
            not isinstance(query, str)
            or not query.strip()
            or len(query) > 1000
            or min(caption_k, stage3_k, rerank_k, candidate_k) <= 0
        ):
            raise ValueError("INVALID_QUERY")
        if ranker not in {"caption", "stage3", "base", "stage5", "dense", "dense-entity"}:
            raise ValueError("INVALID_QUERY")
        if intent_mode not in ("rule", "none", "qwen"):
            raise ValueError("INVALID_QUERY")
        # Historical qwen intent flags are now deterministic too; offline retrieval has no transport dependency.
        with self.lock:
            self.load()
            started = time.perf_counter()
            from .decision_provider import DecisionConfig, DecisionError, create_decision_provider

            try:
                decision_config = DecisionConfig.resolve(
                    provider=decision_provider,
                    device=laya_device,
                    model_path=laya_model_path,
                    batch_size=laya_batch_size,
                    candidate_count=laya_candidate_count,
                )
            except (ValueError, TypeError) as exc:
                raise DecisionError("DECISION_CONFIG_INVALID") from exc
            if decision_config.provider == "laya" and (
                not self.loaded["meta"].get("v4_native") or ranker != "dense-entity"
            ):
                raise DecisionError("LAYA_REQUIRES_METADATA_V4_DENSE_ENTITY")
            intent = parse_intent(query) if intent_mode != "none" else dict(action="", relation="", context="")
            if ranker in ("dense", "dense-entity") and self.loaded["meta"]["format"] != DENSE_FORMAT:
                if ranker == "dense-entity":
                    raise ValueError("dense structured metadata is required for ranker dense-entity")
                ranker = "caption"
            if ranker in ("dense", "dense-entity"):
                texts = query_texts(query, parse_query(query))
                encoding_started = time.perf_counter()
                qfields = self.bge.encode([texts[f] for f in FIELDS], batch_size=8)
                bge_encoded = time.perf_counter()
                _, qproj = self.sig.encode([query], batch_size=1)
                sig_encoded = time.perf_counter()
                table = score_dense(
                    self.index_path,
                    query,
                    qfields,
                    qproj[0],
                    caption_k,
                    stage3_k,
                    candidate_k if ranker == "dense-entity" else rerank_k,
                    loaded=self.loaded,
                )
                retrieval_seconds = time.perf_counter() - sig_encoded
                if ranker == "dense-entity":
                    if self.loaded["meta"].get("v4_native") or self.loaded["meta"].get("v4_1_native") or self.loaded["meta"].get("v4_2_native"):
                        from .v4_matcher import rerank

                        table = rerank(table, self.loaded["v4_features"])
                        if decision_config.provider == "laya":
                            from .v4_matcher import WEIGHTS

                            provider = self.decision_providers.get(decision_config)
                            if provider is None:
                                self.decision_providers.clear()
                                provider = create_decision_provider(decision_config)
                                provider.load()
                                # One optional local model per engine, including device/config changes.
                                self.decision_providers[decision_config] = provider
                            table["pre_laya_candidates"] = [dict(r) for r in table["candidates"]]
                            shortlist = [dict(r) for r in table["candidates"][:15]]
                            judge_started = time.perf_counter()
                            decisions = provider.judge(
                                query,
                                shortlist[: decision_config.candidate_count],
                                self.loaded["v4_features"],
                                table["v4_constraints"],
                            )
                            for r, decision in zip(shortlist, decisions):
                                r["laya_decision"] = decision
                                r["final_score"] += WEIGHTS["laya_term"] * (2 * decision["score"] - 1)
                                r["ranker"] = "dense-entity-v4-laya"
                            shortlist.sort(key=lambda r: (-r["final_score"], r["row"]))
                            table["candidates"] = shortlist + table["candidates"][15:]
                            for i, r in enumerate(table["candidates"], 1):
                                r["rank"] = i
                            best = table["candidates"][0]
                            constraints = best["constraint_result"]
                            table["decision"] = {
                                "provider": "laya",
                                "degraded": False,
                                "state": "MATCH"
                                if constraints["query_constraints"] > 0
                                and constraints["matched"] == constraints["query_constraints"]
                                and not constraints["contradicted"]
                                and best.get("laya_decision", {}).get("decision") == "MATCH"
                                else "NOT_FOUND"
                                if decisions and all(d["decision"] == "NOT_FOUND" for d in decisions)
                                else "LOW_CONFIDENCE",
                                "threshold_status": "provisional_unfitted",
                                "candidate_count": len(decisions),
                                "local": True,
                                "latency_seconds": time.perf_counter() - judge_started,
                                "health": provider.health(),
                            }
                    else:
                        table = rerank_entity_candidates(
                            table,
                            self.loaded["records"],
                            temporal_iou_threshold=temporal_iou_threshold,
                            temporal_dedup=not no_temporal_dedup,
                            normalized_entities=self.loaded.get("entities"),
                        )
                table.setdefault("timing", {}).update(
                    bge_encode_seconds=bge_encoded - encoding_started,
                    siglip_encode_seconds=sig_encoded - bge_encoded,
                    dense_candidate_retrieval_seconds=retrieval_seconds,
                    **getattr(self, "model_load_timing", {}),
                )
            else:
                qfields = self.bge.encode(
                    [query, intent["action"] or "none", intent["relation"] or "none", intent["context"] or "none"],
                    batch_size=4,
                )
                qraw, qproj = self.sig.encode([query], batch_size=1)
                table = score_candidates(
                    self.index_path,
                    query,
                    intent,
                    qfields,
                    qraw[0],
                    qproj[0],
                    caption_k,
                    stage3_k,
                    rerank_k,
                    loaded=self.loaded,
                )
                table["candidates"] = rank_candidates(table, ranker, top_k=len(table["candidates"]), debug_scores=True)
            table.setdefault("timing", {}).update(
                total_search_seconds=time.perf_counter() - started,
                cold_load_seconds=self.cold_load_seconds,
                engine_load_count=self.load_count,
                query_time_qwen_calls=0,
            )
            self.query_count += 1
            return table

    def search(self, query, *, top_k=10, debug=False, **kwargs):
        if top_k <= 0:
            raise ValueError("INVALID_QUERY")
        table = self.search_table(query, **kwargs)
        if table.get("decision", {}).get("state") == "NOT_FOUND" and not debug:
            return []
        return table["candidates"][:top_k]

    def close(self):
        with self.lock:
            self.loaded = None
            self.decision_providers.clear()
            for name in ("bge", "sig"):
                if hasattr(self, name):
                    delattr(self, name)


class SearchEngineCache:
    def __init__(self, max_entries=2, factory=SearchEngine):
        if not 1 <= max_entries <= 8:
            raise ValueError("Search engine cache must be bounded (1..8)")
        self.max_entries, self.factory = max_entries, factory
        self.entries = OrderedDict()
        self.lock = threading.RLock()

    def search_table(self, index, query, **kwargs):
        # Serialize use/eviction so an engine can never be closed during a request.
        with self.lock:
            key = str(Path(index).resolve())
            # Small file stat identity avoids stale residency after index publication; no per-query array hashes.
            meta = Path(key) / "index_meta.json"
            identity = (meta.stat().st_mtime_ns, meta.stat().st_size)
            eg2 = kwargs.get("ranker") == "embeddinggemma2"
            driveevidence = kwargs.get("ranker") == "driveevidence"
            driveevidence_v2 = kwargs.get("ranker") == "driveevidence_v2"
            driveevidence_v3 = kwargs.get("ranker") == "driveevidence_v3"
            driveevidence_v4 = kwargs.get("ranker") == "driveevidence_v4"
            driveevidence_v5 = kwargs.get("ranker") == "driveevidence_v5"
            driveevidence_v6 = kwargs.get("ranker") == "driveevidence_v6"
            if driveevidence_v6:
                from .driveevidence_v6_service import DriveEvidenceV6SearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, Path(key)/"serving_manifest.json", *[Path(key)/name for name in info.get("hashes", {})]]
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (None, None) for p in files)
                if any(not isinstance(engine, DriveEvidenceV6SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif driveevidence_v5:
                from .driveevidence_v5_service import DriveEvidenceV5SearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, *[Path(key)/name for name in info.get("hashes", {})]]
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (None, None) for p in files)
                if any(not isinstance(engine, DriveEvidenceV5SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif driveevidence_v4:
                from .driveevidence_v4 import DriveEvidenceV4SearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, *[Path(key)/name for name in info.get("hashes", {})]]
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (None, None) for p in files)
                if any(not isinstance(engine, DriveEvidenceV4SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif driveevidence_v3:
                from .driveevidence_v3 import DriveEvidenceV3SearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, *[Path(key)/name for name in info.get("hashes", {})]]
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (None, None) for p in files)
                if any(not isinstance(engine, DriveEvidenceV3SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif driveevidence_v2:
                from .driveevidence_v2 import DriveEvidenceV2SearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, *[Path(key)/name for name in info.get("hashes", {})]]
                # Missing optional evidence must reach V2's validated EG2 fallback.
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) if p.is_file() else (None, None) for p in files)
                if any(not isinstance(engine, DriveEvidenceV2SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif driveevidence:
                from .driveevidence import DriveEvidenceSearchEngine
                import json
                info = json.loads(meta.read_text(encoding="utf-8"))
                files = [meta, *[Path(key)/name for name in info.get("hashes", {})]]
                identity = tuple((p.stat().st_mtime_ns, p.stat().st_size) for p in files)
                if any(not isinstance(engine, DriveEvidenceSearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            elif eg2:
                # EG2 owns the GPU exclusively. Historical cache behavior stays unchanged otherwise.
                from .embeddinggemma2_index import EmbeddingGemma2SearchEngine, index_identity
                identity = index_identity(key)
                if any(not isinstance(engine, EmbeddingGemma2SearchEngine) or name != key
                       for name, (_, engine) in self.entries.items()):
                    self.clear()
            else:
                for name, (_, engine) in list(self.entries.items()):
                    if getattr(engine, "worker_factory", None) is not None:
                        self.entries.pop(name)[1].close()
            saved = self.entries.get(key)
            if saved is not None and saved[0] != identity:
                self.entries.pop(key)[1].close()
                saved = None
            if saved is None:
                while len(self.entries) >= self.max_entries:
                    self.entries.popitem(last=False)[1][1].close()
                if driveevidence_v6:
                    import os
                    engine = DriveEvidenceV6SearchEngine(key, deployment_mode=os.getenv("DRIVEEVIDENCE_V6_DEPLOYMENT_MODE", "eg2_only"),
                        telemetry_path=Path(os.getenv("DRIVEEVIDENCE_V6_TELEMETRY_PATH", "outputs/driveevidence_v6/integration/telemetry.jsonl")))
                elif driveevidence_v5:
                    import os
                    engine = DriveEvidenceV5SearchEngine(key, deployment_mode=os.getenv("DRIVEEVIDENCE_V5_DEPLOYMENT_MODE", "eg2_only"))
                else:
                    engine = (DriveEvidenceV4SearchEngine(key) if driveevidence_v4 else
                          DriveEvidenceV3SearchEngine(key) if driveevidence_v3 else
                          DriveEvidenceV2SearchEngine(key) if driveevidence_v2 else
                          DriveEvidenceSearchEngine(key) if driveevidence else
                          EmbeddingGemma2SearchEngine(key) if eg2 else self.factory(key))
                engine.load()
                saved = (identity, engine)
                self.entries[key] = saved
            self.entries.move_to_end(key)
            return saved[1].search_table(query, **kwargs)

    def clear(self):
        with self.lock:
            for _, engine in self.entries.values():
                engine.close()
            self.entries.clear()

    def release_embeddinggemma2(self):
        """Release only the optional GPU worker before a different GPU operation."""
        with self.lock:
            for name, (_, engine) in list(self.entries.items()):
                if getattr(engine, "worker_factory", None) is not None:
                    self.entries.pop(name)[1].close()

    def decision_readiness(self, index, config):
        from .decision_provider import DecisionError, create_decision_provider

        with self.lock:
            self.release_embeddinggemma2()
            key = str(Path(index).resolve())
            meta = Path(key) / "index_meta.json"
            identity = (meta.stat().st_mtime_ns, meta.stat().st_size)
            saved = self.entries.get(key)
            if saved is not None and saved[0] != identity:
                self.entries.pop(key)[1].close()
                saved = None
            if saved is None:
                while len(self.entries) >= self.max_entries:
                    self.entries.popitem(last=False)[1][1].close()
                engine = self.factory(key)
                engine.load()
                saved = (identity, engine)
                self.entries[key] = saved
            engine = saved[1]
            if not engine.loaded["meta"].get("v4_native"):
                raise DecisionError("LAYA_REQUIRES_METADATA_V4_DENSE_ENTITY")
            provider = engine.decision_providers.get(config)
            if provider is None:
                engine.decision_providers.clear()
                provider = create_decision_provider(config)
                provider.load()
                engine.decision_providers[config] = provider
            self.entries.move_to_end(key)
            return provider.health()


DEFAULT_CACHE = SearchEngineCache(max_entries=2)
