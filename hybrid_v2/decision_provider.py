"""Optional local decision provider. No hosted calls or silent deterministic fallback."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import logging
import math
import os
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path

import torch

MODEL = "convaiinnovations/laya"
REVISION = "7b928d828b7b0e022f929d9bd2e44165aa270148"
LOCAL_FILES = (
    "config.json",
    "model.safetensors",
    "rl_agent_config.json",
    "encoder/config.json",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
)
PROMPT_VERSION = "scene-decision-v1"
STATE_VERSION = "scene-actor-state-v1"
INSTRUCTION = "Does the candidate satisfy the query? Required subject, attributes, action and relations must belong to the same relevant actor. Explicit contradictions mean no. Unknown or missing required evidence means no. Use only candidate facts; do not invent facts."
QUESTIONS = {
    "match": {
        "type": "choice",
        "instructions": INSTRUCTION,
        "criteria": {
            "A": "yes, required scene and same actor facts are supported",
            "B": "no, required facts are missing, uncertain or contradicted",
        },
    }
}
LOGGER = logging.getLogger(__name__)


def _package_version(name):
    """Distribution version, or None when the optional package is absent.

    The Laya runtime is an optional dependency. An injected loader (tests,
    alternate backends) must not fail merely because the package metadata is
    unavailable, so record the version as unknown instead of raising.
    """
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


class DecisionError(RuntimeError):
    def __init__(self, code):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class DecisionConfig:
    provider: str = "none"
    device: str = "auto"
    model_path: str = "outputs/metadata_v4_laya/laya/weights/" + REVISION
    revision: str = REVISION
    batch_size: int = 1
    candidate_count: int = 10
    cache_entries: int = 512

    def __post_init__(self):
        if self.provider not in ("none", "laya") or self.device not in ("auto", "cpu", "cuda"):
            raise ValueError("Invalid decision provider/device")
        if not 1 <= self.batch_size <= 15 or not 1 <= self.candidate_count <= 15 or not 0 <= self.cache_entries <= 4096:
            raise ValueError("Decision inference/cache bounds exceeded")

    @classmethod
    def resolve(
        cls, *, provider=None, device=None, model_path=None, batch_size=None, candidate_count=None, environ=None
    ):
        env = os.environ if environ is None else environ
        defaults = cls()

        def setting(explicit, key, default):
            return explicit if explicit is not None else env.get(key, default)

        return cls(
            provider=setting(provider, "DECISION_PROVIDER", defaults.provider),
            device=setting(device, "LAYA_DEVICE", defaults.device),
            model_path=setting(model_path, "LAYA_MODEL_PATH", defaults.model_path),
            batch_size=int(setting(batch_size, "LAYA_BATCH_SIZE", defaults.batch_size)),
            candidate_count=int(setting(candidate_count, "LAYA_CANDIDATE_COUNT", defaults.candidate_count)),
        )


class DecisionProvider:
    def load(self):
        return self

    def judge(self, query, candidates, features, spec):
        raise NotImplementedError


class NoneDecisionProvider(DecisionProvider):
    def judge(self, query, candidates, features, spec):
        return []

    def health(self):
        return {"ready": True, "provider": "none", "model_load_count": 0, "local": True}


class LayaLocalDecisionProvider(DecisionProvider):
    def __init__(self, config=None, *, loader=None):
        self.config = config or DecisionConfig(provider="laya")
        self.loader = loader
        self.agent = None
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        self.load_count = self.inference_count = self.cache_hits = 0
        self.latencies = deque(maxlen=256)
        self.failure = None
        self.provenance = None

    def load(self):
        with self.lock:
            if self.agent is not None:
                return self
            started = time.perf_counter()
            try:
                device = (
                    "cuda"
                    if self.config.device == "auto" and torch.cuda.is_available()
                    else "cpu"
                    if self.config.device == "auto"
                    else self.config.device
                )
                if device == "cuda" and not torch.cuda.is_available():
                    raise DecisionError("LAYA_DEVICE_UNAVAILABLE")
                if self.loader is None:
                    path = Path(self.config.model_path).resolve()
                    # The official runtime falls back to a Hub tokenizer if its local
                    # tokenizer directory is absent. Refuse partial local bundles first.
                    if not path.is_dir() or any(not (path / name).is_file() for name in LOCAL_FILES):
                        raise DecisionError("LAYA_WEIGHTS_UNAVAILABLE")
                    import laya

                    loader = laya.load
                    target = str(path)
                else:
                    loader, target = self.loader, self.config.model_path
                agent = loader(target, device=device, backend="eager", revision=self.config.revision)
                if str(agent.device) != device:
                    raise DecisionError("LAYA_DEVICE_CHANGED")
                self.agent = agent
                self.load_count += 1
                self.provenance = {
                    "model": MODEL,
                    "revision": self.config.revision,
                    "weights_path": target,
                    "device": device,
                    "dtype": str(agent.dtype),
                    "library_versions": {
                        name: _package_version(name)
                        for name in ("laya", "torch", "transformers", "safetensors")
                    },
                    "prompt_version": PROMPT_VERSION,
                    "input_version": STATE_VERSION,
                    "prompt_sha256": hashlib.sha256(json.dumps(QUESTIONS, sort_keys=True).encode()).hexdigest(),
                    "load_seconds": time.perf_counter() - started,
                    "backend": "eager",
                    "local": True,
                }
                if self.loader is None:
                    from .qwen_store import digest

                    weights_digest = digest(path / "model.safetensors")
                    receipt = path / ".cache/huggingface/download/model.safetensors.metadata"
                    lines = receipt.read_text().splitlines() if receipt.exists() else []
                    self.provenance.update(
                        weights_sha256=weights_digest,
                        resolved_revision=lines[0] if len(lines) >= 2 and lines[1] == weights_digest else None,
                        revision_verified=len(lines) >= 2
                        and lines[0] == self.config.revision
                        and lines[1] == weights_digest,
                    )
                self.failure = None
                return self
            except Exception as exc:
                self.agent = None
                self.failure = getattr(exc, "code", "LAYA_LOAD_FAILED")
                LOGGER.exception(
                    "Local decision provider failed", extra={"decision_provider": "laya", "error_code": self.failure}
                )
                raise DecisionError(self.failure) from exc

    def bounded_state(self, query, candidate, features, spec):
        result = candidate["constraint_result"]
        actor = next((a for a in features["actors"] if a["id"] == result["actor_id"]), None)
        query_tokens = self.agent.tok.encode(query, add_special_tokens=False)
        token_ids = query_tokens[:64]
        relevant_events = [
            e
            for e in features["metadata"]["events"]
            if actor is None or e["subject"] == actor["id"] or e["object"] == actor["id"]
        ]
        state = {
            "query": self.agent.tok.decode(token_ids),
            "constraints": spec["constraints"][:8],
            "match": {k: result[k] for k in ("matched", "contradicted", "unknown", "missing")},
            "actor": {
                k: actor[k]
                for k in (
                    "id",
                    "type",
                    "subtype",
                    "appearance",
                    "position",
                    "motion",
                    "confidence",
                    "evidence_frames",
                    "associated_entity",
                )
            }
            if actor
            else None,
            "scene": features["scene"],
            "infrastructure": list(features["infrastructure"].values())[:3],
            "events": relevant_events[:2],
            "evidence_trimmed": len(query_tokens) > 64
            or len(spec["constraints"]) > 8
            or len(features["infrastructure"]) > 3
            or len(relevant_events) > 2,
        }

        # Valid JSON is reduced structurally, not sliced mid-field. Official budget is 512/192.
        def encoded():
            text = json.dumps(state, separators=(",", ":"), ensure_ascii=False)
            return text, len(self.agent.tok.encode(text, add_special_tokens=False))

        removals = [(state, k) for k in ("infrastructure", "scene", "events")]
        if actor:
            removals += [(state["actor"], "confidence")]
        for container, key in removals:
            if encoded()[1] <= 280:
                break
            container.pop(key, None)
            state["evidence_trimmed"] = True
        while encoded()[1] > 280 and len(state["constraints"]) > 1:
            state["constraints"].pop()
            state["evidence_trimmed"] = True
        if encoded()[1] > 280:
            state["actor"] = {"id": actor["id"], "type": actor["type"], "subtype": actor["subtype"]} if actor else None
            state["evidence_trimmed"] = True
        text, tokens = encoded()
        if tokens > 280:
            raise DecisionError("LAYA_INPUT_TOO_LARGE")
        return text, tokens, state["evidence_trimmed"]

    def cache_key(self, query, features, spec):
        identity = [
            hashlib.sha256(query.encode()).hexdigest(),
            hashlib.sha256(json.dumps(features["metadata"], sort_keys=True).encode()).hexdigest(),
            self.config.revision,
            PROMPT_VERSION,
            STATE_VERSION,
            self.config.model_path,
            self.config.device,
            self.config.batch_size,
            hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
        ]
        return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def judge(self, query, candidates, features, spec):
        with self.lock:
            self.load()
            if len(candidates) > self.config.candidate_count:
                raise ValueError("Laya candidate budget exceeded")
            outputs, misses = {}, []
            for i, c in enumerate(candidates):
                feature = features[c["row"]]
                key = self.cache_key(query, feature, spec)
                if key in self.cache:
                    self.cache_hits += 1
                    self.cache.move_to_end(key)
                    outputs[i] = {**self.cache[key], "cache_hit": True, "latency_seconds": 0}
                else:
                    state, tokens, trimmed = self.bounded_state(query, c, feature, spec)
                    misses.append((i, key, state, tokens, trimmed))
            started = time.perf_counter()
            try:
                for start in range(0, len(misses), self.config.batch_size):
                    batch = misses[start : start + self.config.batch_size]
                    t = time.perf_counter()
                    results = self.agent.predict_batch(
                        [m[2] for m in batch],
                        QUESTIONS,
                        batch_size=self.config.batch_size,
                        max_len=512,
                        head_max_len=192,
                    )
                    elapsed = time.perf_counter() - t
                    self.inference_count += 1
                    if len(results) != len(batch) or str(self.agent.device) != self.provenance["device"]:
                        raise DecisionError("LAYA_INFERENCE_FAILED")
                    for (i, key, _, tokens, trimmed), raw in zip(batch, results):
                        answer = raw["answers"]["match"]
                        scores = answer["probabilities"]
                        score = float(scores["A"])
                        if (
                            answer["choice"] not in ("A", "B")
                            or set(scores) != {"A", "B"}
                            or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in scores.values())
                            or abs(sum(scores.values()) - 1) > 0.01
                        ):
                            raise DecisionError("LAYA_INVALID_SCORE")
                        out = {
                            "decision": "MATCH" if answer["choice"] == "A" else "NOT_FOUND",
                            "score": score,
                            "raw_answer": answer,
                            "calibrated": False,
                            "latency_seconds": elapsed / len(batch),
                            "batch_seconds": elapsed,
                            "batch_size": len(batch),
                            "state_tokens": tokens,
                            "evidence_trimmed": trimmed,
                            "usage": raw.get("usage", {}),
                            "cache_hit": False,
                            "cache_key": key,
                        }
                        outputs[i] = out
                        if self.config.cache_entries:
                            self.cache[key] = out
                            while len(self.cache) > self.config.cache_entries:
                                self.cache.popitem(last=False)
                if misses:
                    self.latencies.append(time.perf_counter() - started)
                self.failure = None
                return [outputs[i] for i in range(len(candidates))]
            except Exception as exc:
                self.failure = getattr(exc, "code", "LAYA_INFERENCE_FAILED")
                LOGGER.exception("Local decision inference failed")
                raise DecisionError(self.failure) from exc

    def health(self):
        return {
            "ready": self.agent is not None and self.failure is None,
            "provider": "laya",
            "error_code": self.failure,
            "model_load_count": self.load_count,
            "inference_batches": self.inference_count,
            "cache_hits": self.cache_hits,
            "cache_size": len(self.cache),
            "provenance": self.provenance,
            "local": True,
        }


def create_decision_provider(config):
    return NoneDecisionProvider() if config.provider == "none" else LayaLocalDecisionProvider(config)
