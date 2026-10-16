"""Exactly two Qwen modes; bounded SDK transport, backpressure and failure evidence."""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

import cv2
import numpy as np
import openai

SERVER_URL = "http://210.245.53.88:5992/v1"
SERVER_MODEL = "qwen3-vl-4b"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class QwenConfig:
    mode: str = "local"
    base_url: str = "http://127.0.0.1:8080/v1"
    model: str | None = None
    api_key: str = field(default="sk-no-key-required", repr=False)
    connect_timeout_sec: float = 10
    read_timeout_sec: float = 180
    max_retries: int = 3
    concurrency: int = 1
    queue_size: int = 2
    breaker_threshold: int = 3
    breaker_cooldown_sec: float = 30

    def __post_init__(self):
        parsed = urlsplit(self.base_url)
        if self.mode not in ("local", "server"):
            raise ValueError("QWEN_MODE must be local or server")
        if (
            parsed.scheme not in ("http", "https")
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("Invalid Qwen base URL (credentials/query strings are forbidden)")
        if not parsed.path.rstrip("/").endswith("/v1"):
            raise ValueError("Qwen base URL must end in /v1")
        if any(
            not math.isfinite(v) or v <= 0
            for v in (self.connect_timeout_sec, self.read_timeout_sec, self.breaker_cooldown_sec)
        ):
            raise ValueError("Qwen timeouts must be finite and positive")
        if (
            not 0 <= self.max_retries <= 5
            or not 1 <= self.concurrency <= 2
            or not 0 <= self.queue_size <= 16
            or self.breaker_threshold < 1
        ):
            raise ValueError("Invalid bounded Qwen retry/concurrency configuration")
        if self.mode == "server" and not self.model:
            raise ValueError("Server mode requires an explicit model")

    def public(self):
        return {k: v for k, v in asdict(self).items() if k != "api_key"}

    @classmethod
    def resolve(cls, cli=None, config=None, environ=None):
        """CLI > env > config > defaults. Historical URL/model flags remain explicit overrides."""
        cli, config = cli or {}, config or {}
        env = os.environ if environ is None else environ
        mode = cli.get("qwen_mode") or env.get("QWEN_MODE") or config.get("qwen_mode") or "local"
        if mode not in ("local", "server"):
            raise ValueError("QWEN_MODE must be local or server")
        prefix = "QWEN_" + mode.upper() + "_"
        defaults = asdict(
            cls(
                mode=mode,
                base_url=SERVER_URL if mode == "server" else "http://127.0.0.1:8080/v1",
                model=SERVER_MODEL if mode == "server" else None,
            )
        )
        resolved = {"mode": mode}
        for name in defaults:
            if name == "mode":
                continue
            key = "qwen_" + mode + "_" + name
            legacy = {"base_url": "qwen_base_url", "model": "vlm_model"}.get(name)
            value = cli.get(key)
            if value is None and legacy:
                value = cli.get(legacy)
            if value is None:
                value = env.get(prefix + name.upper())
            if value is None:
                value = config.get(key)
            # A saved local legacy URL must never override the server default.
            if value is None and legacy and mode == "local":
                value = config.get(legacy)
            resolved[name] = defaults[name] if value is None else value
        for name in ("max_retries", "concurrency", "queue_size", "breaker_threshold"):
            resolved[name] = int(resolved[name])
        for name in ("connect_timeout_sec", "read_timeout_sec", "breaker_cooldown_sec"):
            resolved[name] = float(resolved[name])
        return cls(**resolved)


class QwenError(RuntimeError):
    def __init__(self, code, *, retryable=False, detail=None):
        self.code, self.retryable, self.detail = code, retryable, detail
        super().__init__(code)


def redact(value, secrets=()):
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if any(s in k.casefold() for s in ("api_key", "authorization")) else redact(v, secrets)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact(v, secrets) for v in value]
    if isinstance(value, str):
        for secret in secrets:
            if secret:
                value = value.replace(secret, "[REDACTED]")
        value = re.sub(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "[IMAGE REDACTED]", value)
        return re.sub(r"(?i)(bearer\s+|api[_-]?key[=: ]+)[^\s\"',}]+", r"\1[REDACTED]", value)
    return value


def jpeg_data_url(frame, quality=85):
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("JPEG quality must be an integer from 1 to 100")
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not frame.size
    ):
        raise ValueError("Expected a nonempty uint8 BGR image")
    ok, data = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")


# JSON grammar is probed only when schema response_format is explicitly unsupported.
JSON_GRAMMAR = r"""root ::= object
object ::= "{" ws (string ws ":" ws value (ws "," ws string ws ":" ws value)*)? ws "}"
array ::= "[" ws (value (ws "," ws value)*)? ws "]"
value ::= object | array | string | number | "true" | "false" | "null"
string ::= "\"" ([^"\\\x00-\x1F] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))* "\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [+-]? [0-9]+)?
ws ::= [ \t\n\r]*"""


class QwenVisionProvider:
    def __init__(
        self, config, *, client=None, evidence_dir=None, sleep=time.sleep, monotonic=time.monotonic, contract=None
    ):
        from . import qwen_metadata

        self.contract = contract or qwen_metadata
        self.config = config
        self.requested_model = config.model
        self.client = client or openai.OpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
            max_retries=0,
            timeout=openai.Timeout(
                config.read_timeout_sec,
                connect=config.connect_timeout_sec,
                read=config.read_timeout_sec,
                write=config.read_timeout_sec,
                pool=config.connect_timeout_sec,
            ),
        )
        self.evidence_dir = Path(evidence_dir) if evidence_dir else None
        self.sleep, self.clock = sleep, monotonic
        self.lock = threading.RLock()
        self.slots = threading.BoundedSemaphore(config.concurrency)
        self.admission = threading.BoundedSemaphore(config.concurrency + config.queue_size)
        self.events = deque(maxlen=10000)
        self.counts = dict(
            request_count=0,
            structured_request_count=0,
            success_count=0,
            retry_count=0,
            failure_count=0,
            json_valid_count=0,
            schema_valid_count=0,
            payload_bytes=0,
            response_bytes=0,
        )
        self.failures, self.open_until = 0, 0.0
        self.output_mode = "json_schema"
        self.capabilities = {}
        self.actual_models = set()

    @property
    def model(self):
        return self.config.model

    def _check_breaker(self):
        with self.lock:
            if self.clock() < self.open_until:
                raise QwenError("QWEN_UNAVAILABLE", retryable=True)

    def _failed(self):
        with self.lock:
            self.failures += 1
            if self.failures >= self.config.breaker_threshold:
                self.open_until = self.clock() + self.config.breaker_cooldown_sec

    def _evidence(self, event):
        event = redact(event, (self.config.api_key,))
        with self.lock:
            self.events.append(event)
            if self.evidence_dir:
                self.evidence_dir.mkdir(parents=True, exist_ok=True)
                with (self.evidence_dir / "requests.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        LOGGER.info(
            json.dumps(
                {
                    k: event.get(k)
                    for k in (
                        "request_id",
                        "correlation_id",
                        "index_id",
                        "operation",
                        "duration_ms",
                        "status",
                        "error_code",
                    )
                }
            )
        )

    def healthcheck(self):
        self._check_breaker()
        try:
            models = self.client.models.list()
            ids = [item.id for item in models.data]
            if not ids:
                raise QwenError("QWEN_UNAVAILABLE", retryable=True)
            if self.config.model is None:
                # Local historical configuration discovers its currently served model once.
                from dataclasses import replace

                self.config = replace(self.config, model=ids[0])
            return {
                "reachable": True,
                "requested_model": self.model,
                "advertised_models": ids,
                "alias_advertised": self.model in ids,
                "mode": self.config.mode,
                "base_url": self.config.base_url,
            }
        except openai.APIStatusError as exc:
            if exc.status_code in (404, 405, 501) and self.model:
                return {"reachable": True, "models_supported": False, "requested_model": self.model}
            raise QwenError("QWEN_UNAVAILABLE", retryable=exc.status_code in (429, 502, 503, 504)) from None
        except (openai.APIConnectionError, openai.APITimeoutError):
            self._failed()
            raise QwenError("QWEN_UNAVAILABLE", retryable=True) from None

    def _format(self, contract=None):
        contract = contract or self.contract
        if self.output_mode == "json_schema":
            return {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": "scene_metadata", "strict": True, "schema": contract.SCHEMA},
                }
            }
        if self.output_mode == "grammar":
            return {"extra_body": {"grammar": JSON_GRAMMAR}}
        if self.output_mode == "json_object":
            return {"response_format": {"type": "json_object"}}
        return {}

    def complete(
        self,
        prompt,
        image_urls=(),
        *,
        structured=False,
        window_id=None,
        index_id=None,
        correlation_id=None,
        max_tokens=1800,
        attempt=1,
        contract=None,
        pass_id=None,
    ):
        """One bounded wire request. Exceptions are classified, never raw SDK strings."""
        contract = contract or self.contract
        self._check_breaker()
        if self.model is None:
            self.healthcheck()
        if not self.admission.acquire(blocking=False):
            raise QwenError("QWEN_BUSY", retryable=True)
        acquired = False
        try:
            acquired = self.slots.acquire(timeout=self.config.connect_timeout_sec)
            if not acquired:
                raise QwenError("QWEN_BUSY", retryable=True)
            self._check_breaker()
            content = [{"type": "text", "text": prompt}]
            for url in image_urls:
                if not url.startswith("data:image/jpeg;base64,"):
                    raise ValueError("Qwen indexing requires JPEG data URLs")
                content.append({"type": "image_url", "image_url": {"url": url}})
            payload = dict(
                model=self.model, messages=[{"role": "user", "content": content}], max_tokens=max_tokens, temperature=0
            )
            if structured:
                payload.update(self._format(contract))
            if len(json.dumps(payload).encode()) > 8 * 1024 * 1024:
                raise ValueError("Qwen request payload exceeds 8 MiB")
            request_id = uuid4().hex
            event = dict(
                window_id=window_id,
                pass_id=pass_id,
                provider_mode=self.config.mode,
                request_id=request_id,
                correlation_id=correlation_id or request_id,
                index_id=index_id,
                operation="qwen_completion",
                attempt=attempt,
                http_status=None,
                status="FAILED",
                error_code=None,
                response_body="",
                json_valid=False,
                schema_valid=False,
                payload_bytes=len(json.dumps(payload).encode()),
                response_bytes=0,
            )
            started = self.clock()
            with self.lock:
                self.counts["request_count"] += 1
                self.counts["structured_request_count"] += structured
                self.counts["payload_bytes"] += event["payload_bytes"]
            try:
                result = self.client.chat.completions.create(**payload, extra_headers={"X-Request-ID": request_id})
                event.update(
                    http_status=200,
                    server_request_id=getattr(result, "_request_id", None),
                    actual_model=result.model,
                    finish_reason=result.choices[0].finish_reason,
                )
                usage = getattr(result, "usage", None)
                if usage is not None:
                    event["output_tokens"] = getattr(usage, "completion_tokens", None)
                    event["input_tokens"] = getattr(usage, "prompt_tokens", None)
                self.actual_models.add(result.model)
                text = result.choices[0].message.content
                event["response_bytes"] = len(text.encode()) if isinstance(text, str) else 0
                if not isinstance(text, str) or result.choices[0].finish_reason == "length":
                    raise QwenError("QWEN_SCHEMA_FAILURE", retryable=True)
                # Validation belongs to this same wire attempt for honest telemetry.
                if structured:
                    try:
                        json.loads(text)
                        event["json_valid"] = True
                        contract.parse_metadata(text)
                        event["schema_valid"] = True
                    except ValueError as exc:
                        event.update(response_body=text[:16000], parse_schema_error=str(exc))
                        raise QwenError("QWEN_SCHEMA_FAILURE", retryable=True, detail=str(exc)) from None
                event["status"] = "SUCCESS"
                with self.lock:
                    self.failures, self.open_until = 0, 0.0
                return text
            except openai.APIStatusError as exc:
                body = exc.response.text
                event.update(http_status=exc.status_code, response_body=body[:16000], response_bytes=len(body.encode()))
                if (
                    structured
                    and exc.status_code in (400, 422)
                    and any(s in body.lower() for s in ("response_format", "json_schema", "grammar", "unsupported"))
                ):
                    raise QwenError("QWEN_FORMAT_UNSUPPORTED", retryable=True) from None
                self._failed()
                raise QwenError("QWEN_UNAVAILABLE", retryable=exc.status_code in (429, 502, 503, 504)) from None
            except openai.APITimeoutError:
                self._failed()
                raise QwenError("QWEN_TIMEOUT", retryable=True) from None
            except openai.APIConnectionError:
                self._failed()
                raise QwenError("QWEN_UNAVAILABLE", retryable=True) from None
            except (KeyError, IndexError, AttributeError, TypeError):
                raise QwenError("QWEN_SCHEMA_FAILURE", retryable=True) from None
            finally:
                import sys

                error = sys.exc_info()[1]
                if error:
                    event["error_code"] = getattr(error, "code", "QWEN_SCHEMA_FAILURE")
                event["duration_ms"] = (self.clock() - started) * 1000
                with self.lock:
                    self.counts["success_count" if event["status"] == "SUCCESS" else "failure_count"] += 1
                    self.counts["json_valid_count"] += event["json_valid"]
                    self.counts["schema_valid_count"] += event["schema_valid"]
                    self.counts["response_bytes"] += event["response_bytes"]
                self._evidence(event)
        finally:
            if acquired:
                self.slots.release()
            self.admission.release()

    def generate_metadata(self, frames, request=None):
        request = request or {}
        urls = [f if isinstance(f, str) else jpeg_data_url(f, request.get("jpeg_quality", 85)) for f in frames]
        if not 1 <= len(urls) <= 12:
            raise ValueError("Supply 1 to 12 frames from one semantic window")
        correlation_id = request.get("correlation_id") or uuid4().hex
        last = None
        schema_failures = 0
        for attempt in range(self.config.max_retries + 1):
            prompt = self.contract.PROMPT
            if schema_failures:
                prompt += "\nCompact retry: use fewer entities, unknown for uncertainty, all required fields."
                if last and last.detail:
                    prompt += "\nCorrect this validation error from the previous attempt: " + last.detail
            if schema_failures >= 2:
                prompt += "\nRe-extract only visible critical fields; omit unsupported actors/events. Do not invent missing facts."
            try:
                result = self.complete(
                    prompt,
                    urls,
                    structured=True,
                    window_id=request.get("window_id"),
                    index_id=request.get("index_id"),
                    correlation_id=correlation_id,
                    attempt=attempt + 1,
                    max_tokens=getattr(self.contract, "MAX_TOKENS", 1800),
                )
                self.capabilities[self.output_mode] = "validated"
                return self.contract.parse_metadata(result)
            except QwenError as exc:
                last = exc
                if exc.code == "QWEN_FORMAT_UNSUPPORTED":
                    self.capabilities[self.output_mode] = "unsupported"
                    self.output_mode = {"json_schema": "grammar", "grammar": "prompt", "prompt": "prompt"}[
                        self.output_mode
                    ]
                elif exc.code == "QWEN_SCHEMA_FAILURE":
                    schema_failures += 1
                if not exc.retryable or schema_failures >= 3 or attempt >= self.config.max_retries:
                    break
                self._check_breaker()
                with self.lock:
                    self.counts["retry_count"] += 1
                self.sleep(min(4.0, 0.5 * 2**attempt))
        raise last or QwenError("QWEN_SCHEMA_FAILURE")

    def generate_structured(
        self,
        prompt,
        image_urls,
        contract,
        *,
        window_id=None,
        index_id=None,
        correlation_id=None,
        pass_id=None,
        max_tokens=None,
        max_retries=None,
    ):
        """Bounded schema-validated generation against an arbitrary pass contract.

        Returns the parsed value. Retries only on retryable, non-schema failures; a
        schema failure raises immediately so the multi-pass orchestrator can decide.
        """
        retries = self.config.max_retries if max_retries is None else max_retries
        limit = getattr(contract, "MAX_TOKENS", 1800) if max_tokens is None else max_tokens
        last = None
        for attempt in range(retries + 1):
            try:
                text = self.complete(
                    prompt,
                    image_urls,
                    structured=True,
                    window_id=window_id,
                    index_id=index_id,
                    correlation_id=correlation_id,
                    attempt=attempt + 1,
                    max_tokens=limit,
                    contract=contract,
                    pass_id=pass_id,
                )
                self.capabilities[self.output_mode] = "validated"
                return contract.parse_metadata(text)
            except QwenError as exc:
                last = exc
                if exc.code == "QWEN_FORMAT_UNSUPPORTED":
                    self.capabilities[self.output_mode] = "unsupported"
                    self.output_mode = {"json_schema": "grammar", "grammar": "prompt", "prompt": "prompt"}[
                        self.output_mode
                    ]
                if not exc.retryable or exc.code == "QWEN_SCHEMA_FAILURE" or attempt >= retries:
                    break
                self._check_breaker()
                with self.lock:
                    self.counts["retry_count"] += 1
                self.sleep(min(4.0, 0.5 * 2**attempt))
        raise last or QwenError("QWEN_SCHEMA_FAILURE")

    def metrics(self):
        with self.lock:
            counts = dict(self.counts)
            latencies = [e["duration_ms"] / 1000 for e in self.events]
        n = counts["structured_request_count"]
        return {
            **counts,
            "json_valid_rate": counts["json_valid_count"] / n if n else None,
            "schema_valid_rate": counts["schema_valid_count"] / n if n else None,
            "p50_latency_sec": float(np.percentile(latencies, 50)) if latencies else None,
            "p95_latency_sec": float(np.percentile(latencies, 95)) if latencies else None,
            "output_mode": self.output_mode,
            "actual_models": sorted(self.actual_models),
            "capabilities": self.capabilities,
        }

    def close(self):
        self.client.close()


class LocalQwenProvider(QwenVisionProvider):
    def __init__(self, config=None, **kwargs):
        config = config or QwenConfig.resolve(cli={"qwen_mode": "local"})
        if config.mode != "local":
            raise ValueError("LocalQwenProvider requires local mode")
        super().__init__(config, **kwargs)


class ServerQwenProvider(QwenVisionProvider):
    def __init__(self, config=None, **kwargs):
        config = config or QwenConfig.resolve(cli={"qwen_mode": "server"})
        if config.mode != "server":
            raise ValueError("ServerQwenProvider requires server mode")
        super().__init__(config, **kwargs)


def resolve_provider_config(config=None, *, cli=None, environ=None):
    config = {k: v for k, v in (config or {}).items() if v is not None}
    cli = {**config.get("qwen_cli", {}), **{k: v for k, v in (cli or {}).items() if v is not None}}
    config_path = (cli or {}).get("qwen_config") or config.get("qwen_config")
    if config_path:
        from src.indexing.config import load_structured

        config = {**load_structured(config_path), **config}
    return QwenConfig.resolve(cli=cli, config=config, environ=environ)


def create_provider(config=None, *, cli=None, **kwargs):
    resolved = resolve_provider_config(config, cli=cli)
    return {"local": LocalQwenProvider, "server": ServerQwenProvider}[resolved.mode](resolved, **kwargs)
