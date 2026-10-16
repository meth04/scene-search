"""V4.2 transport with an actual wall deadline, cancellable worker and evidence."""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

from . import metadata_v4_2 as contract
from .qwen_provider import QwenVisionProvider, QwenError


def classify(event):
    if event.get("finish_reason") == "length":
        return "MODEL_GENERATION"
    if event.get("error_code") == "QWEN_TRANSPORT":
        return "NETWORK_CLIENT_TRANSPORT"
    if event.get("error_code") in ("QWEN_SCHEMA_FAILURE", "QWEN_RESPONSE_BYTE_CAP"):
        return "SCHEMA_FAILURE"
    # No response/timing cannot distinguish a server queue from network silence.
    return "UNKNOWN"


class BoundedProvider(QwenVisionProvider):
    def complete(self, prompt, image_urls=(), *, structured=True, window_id=None, index_id=None,
                 correlation_id=None, max_tokens=1100, attempt=1, contract=None, pass_id=None):
        if self.config.concurrency != 1 or pass_id not in globals()["contract"].BUDGETS:
            raise ValueError("V4.2 requires serial per-pass requests")
        if not 1 <= max_tokens <= globals()["contract"].BUDGETS[pass_id]:
            raise ValueError("V4.2 output token cap")
        timeout = globals()["contract"].TIMEOUTS[pass_id]
        schema = globals()["contract"].wire_schema(contract.SCHEMA, pass_id)
        if len(json.dumps(schema).encode()) > globals()["contract"].MAX_SCHEMA_BYTES:
            raise ValueError("V4.2 schema byte cap")
        if any(not u.startswith("data:image/jpeg;base64,") for u in image_urls):
            raise ValueError("JPEG data URLs required")
        payload = {"model": self.model, "temperature": 0, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": [{"type": "text", "text": prompt},
                       *[{"type": "image_url", "image_url": {"url": u}} for u in image_urls]]}],
                   "response_format": {"type": "json_schema", "json_schema":
                       {"name": "metadata_" + pass_id, "strict": True, "schema": schema}}}
        request_id = uuid4().hex
        event = {"request_id": request_id, "correlation_id": correlation_id or request_id,
                 "window_id": window_id, "index_id": index_id, "pass_id": pass_id, "pass": pass_id,
                 "attempt": attempt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                 "frame_count": 8, "image_count": len(image_urls),
                 "payload_bytes": len(json.dumps(payload).encode()), "output_budget": max_tokens,
                 "wall_timeout_sec": timeout, "http_status": None, "response_bytes": 0,
                 "finish_reason": "unavailable", "prompt_tokens": "unavailable",
                 "completion_tokens": "unavailable", "prompt_eval_time": "unavailable",
                 "generation_eval_time": "unavailable", "tokens_per_sec": "unavailable",
                 "exception_class": None, "error_code": None, "status": "FAILED",
                 "json_valid": False, "schema_valid": False}
        if event["payload_bytes"] > 8 * 1024 * 1024:
            raise ValueError("Payload byte cap")
        data = {"payload": payload, "timeout": timeout, "byte_cap": globals()["contract"].MAX_RESPONSE_BYTES,
                "url": self.config.base_url.rstrip("/") + "/chat/completions",
                "headers": {"Authorization": "Bearer " + self.config.api_key, "X-Request-ID": request_id}}
        started = time.perf_counter()
        # Never submit a second request while a timed-out client worker is alive.
        with self.lock:
            proc = subprocess.Popen([sys.executable, "-B", "-m", "hybrid_v2.metadata_v4_2_transport"],
                                    cwd=Path(__file__).resolve().parents[1], stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                    creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0)
            try:
                try:
                    output, _ = proc.communicate(json.dumps(data), timeout=max(.01, timeout - (time.perf_counter() - started)))
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()
                    event["exception_class"] = "WallClockTimeout"
                    raise QwenError("QWEN_TIMEOUT", retryable=True) from None
                result = json.loads(output)
                event.update({k: result[k] for k in ("http_status", "response_bytes", "exception_class") if k in result})
                if result.get("error"):
                    raise QwenError(result["error"], retryable=True)
                if result["http_status"] != 200:
                    raise QwenError("QWEN_HTTP_FAILURE", retryable=result["http_status"] in (429, 500, 502, 503, 504))
                body = result["body"]
                choice = body["choices"][0]
                event["finish_reason"] = choice.get("finish_reason", "unavailable")
                usage, timing = body.get("usage") or {}, body.get("timings") or {}
                event.update(prompt_tokens=usage.get("prompt_tokens", "unavailable"),
                             completion_tokens=usage.get("completion_tokens", "unavailable"),
                             prompt_eval_time=timing.get("prompt_ms", "unavailable"),
                             generation_eval_time=timing.get("predicted_ms", "unavailable"),
                             tokens_per_sec=timing.get("predicted_per_second", "unavailable"),
                             actual_model=body.get("model", "unavailable"))
                if event["finish_reason"] == "length":
                    raise QwenError("QWEN_GENERATION_RUNAWAY", retryable=True)
                text = choice["message"]["content"]
                contract.parse_metadata(text)
                event.update(status="SUCCESS", json_valid=True, schema_valid=True)
                return text
            except globals()["contract"].ActorIDError as exc:
                event.update(error_code="V42_UNKNOWN_ACTOR_ID", exception_class=type(exc).__name__)
                raise QwenError("V42_UNKNOWN_ACTOR_ID", retryable=False) from None
            except QwenError as exc:
                event["error_code"] = exc.code
                raise
            except (ValueError, KeyError, TypeError, IndexError) as exc:
                event.update(error_code="QWEN_SCHEMA_FAILURE", exception_class=type(exc).__name__, validation_error=str(exc)[:500])
                raise QwenError("QWEN_SCHEMA_FAILURE", retryable=True) from None
            finally:
                event["duration_ms"] = (time.perf_counter() - started) * 1000
                event["slow_request_class"] = classify(event)
                self._evidence(event)
