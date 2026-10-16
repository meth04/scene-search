"""V4.1 multi-pass orchestration: per-pass atomicity, resume, circuit breaker,
Stage3A advisory grounding and a deterministic local merge.

One failed pass/window never aborts the batch: a failure receipt is written and
the caller continues. Completed passes are individually resumable and are never
destroyed by a later failure.
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from . import metadata_v4_1 as contract
from .metadata_v4 import VERSION as V4_VERSION
from .qwen_provider import QwenError, redact
from .qwen_store import atomic_json, digest, window_lock

PASS_STATES = {
    "PASS1_SUCCESS",
    "PASS2_SUCCESS",
    "PASS3_SUCCESS",
    "PASS3_SKIPPED",
    "FINAL_SUCCESS",
    "FAILED_RETRYABLE",
    "FAILED_PERMANENT",
}
# Which pass receipts gate reuse of a completed final record.
_REQUIRED = ("pass1", "pass2", "final")


class MultiPassStore:
    def __init__(self, root, *, contract_module=contract):
        self.root = Path(root)
        self.contract = contract_module

    def _paths(self, window_id, name):
        base = self.root / name
        return base / "rows" / (window_id + ".json"), base / "receipts" / (window_id + ".json")

    def _read(self, window_id, name):
        row, receipt = self._paths(window_id, name)
        if not receipt.exists():
            return None, None
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        if saved.get("state") not in PASS_STATES:
            raise ValueError("V4.1 receipt state invalid")
        if saved["state"].endswith("_SUCCESS") or saved["state"] == "PASS3_SKIPPED":
            if not row.exists() or digest(row) != saved.get("sha256"):
                raise ValueError("V4.1 successful pass missing/corrupt; refusing silent replacement")
            return json.loads(row.read_text(encoding="utf-8")), saved
        return None, saved

    def commit(self, window_id, name, value, *, fingerprint, latency_ms=None, correlation_id=None, state=None):
        row, receipt = self._paths(window_id, name)
        atomic_json(row, value)
        record = {
            "window_id": window_id,
            "pass": name,
            "fingerprint": fingerprint,
            "state": state or ("PASS1_SUCCESS" if name == "pass1" else "PASS2_SUCCESS" if name == "pass2" else "PASS3_SUCCESS" if name == "pass3" else "FINAL_SUCCESS"),
            "sha256": digest(row),
            "timestamp": datetime.now(UTC).isoformat(),
            "latency_ms": latency_ms,
            "correlation_id": correlation_id,
        }
        atomic_json(receipt, record)
        return value

    def commit_skip(self, window_id, name, *, fingerprint, reason):
        _, receipt = self._paths(window_id, name)
        atomic_json(
            receipt,
            {
                "window_id": window_id,
                "pass": name,
                "fingerprint": fingerprint,
                "state": "PASS3_SKIPPED",
                "reason": reason,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def commit_failure(self, window_id, name, *, fingerprint, error_code, retryable, latency_ms=None, correlation_id=None):
        _, receipt = self._paths(window_id, name)
        atomic_json(
            receipt,
            {
                "window_id": window_id,
                "pass": name,
                "fingerprint": fingerprint,
                "state": "FAILED_RETRYABLE" if retryable else "FAILED_PERMANENT",
                "error_code": error_code,
                "latency_ms": latency_ms,
                "correlation_id": correlation_id,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

    def resume(self, window_id, fingerprints):
        """Return reusable passes whose fingerprints still match; else None."""
        values = {}
        for name in ("pass1", "pass2", "pass3", "final"):
            value, saved = self._read(window_id, name)
            if value is None:
                if saved and saved["state"] in ("FAILED_PERMANENT",) and name == "final":
                    raise QwenError(saved["error_code"], retryable=False)
                continue
            if saved["fingerprint"] != fingerprints[name]:
                raise ValueError("V4.1 resume identity mismatch; choose a new output")
            values[name] = value
        return values

    def final_success(self, window_id):
        _, saved = self._read(window_id, "final")
        return bool(saved and saved["state"] == "FINAL_SUCCESS")

    def status(self):
        counts = dict.fromkeys(sorted(PASS_STATES), 0)
        for name in ("pass1", "pass2", "pass3", "final"):
            for path in (self.root / name / "receipts").glob("*.json"):
                saved = json.loads(path.read_text(encoding="utf-8"))
                counts[saved["state"]] += 1
        return counts

    def completed_rows(self):
        return sorted(p.stem for p in (self.root / "final" / "receipts").glob("*.json"))


def stage3a_advisory(manifest_row, tracks, masks):
    """Advisory motion/actor evidence only. Never fabricates actors."""
    row = manifest_row["row"]
    track_mask = masks[row]
    motion_frames = int(track_mask.sum())
    return {
        "motion_frames": motion_frames,
        "has_motion": motion_frames > 0,
        "suggests_actors": motion_frames >= 2,
        "advisory_only": True,
    }


def temporal_recommended(advisory, clip_category=None):
    """Pass 3 is conditional: run for known action-heavy clips or visible motion."""
    if clip_category == "action_temporal":
        return True, "action_heavy_clip"
    if advisory["has_motion"]:
        return True, "stage3a_motion_evidence"
    return False, "no_motion_evidence"


def generate_window(
    window_id,
    image_url,
    provider,
    store,
    *,
    index_id,
    source_identity,
    settings,
    advisory,
    clip_category=None,
    max_retries=3,
    empty_retry=True,
    sleep=time.sleep,
):
    """Run pass1→pass2→pass3 for one window; resume completed passes; never raise for
    a single window (the caller writes a failure receipt and continues)."""
    image_sha = hashlib.sha256(image_url.encode()).hexdigest()
    fingerprints = {
        name: hashlib.sha256(
            json.dumps(
                {
                    "pass": name,
                    "version": contract.VERSION,
                    "prompt_sha256": contract.PROMPT_SHA256.get(name, V4_VERSION),
                    "provider": {**provider.config.public(), "model": provider.requested_model},
                    "image_sha256": image_sha,
                    "source_identity": source_identity,
                    "settings": settings,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        for name in ("pass1", "pass2", "pass3", "final")
    }
    with window_lock(store.root / "locks" / (window_id + ".lock")):
        saved = store.resume(window_id, fingerprints)
        correlation_id = hashlib.sha256((window_id + image_sha).encode()).hexdigest()[:32]
        timings = {}

        # ---- Pass 1 ---------------------------------------------------------
        pass1 = saved.get("pass1")
        if pass1 is None:
            started = time.perf_counter()
            try:
                pass1 = _run_pass(
                    provider, contract.parse_pass1, contract.PASS1_PROMPT, image_url,
                    window_id, index_id, correlation_id, "pass1", contract.PASS1_MAX_TOKENS, max_retries, sleep,
                )
            except QwenError as exc:
                store.commit_failure(
                    window_id, "pass1", fingerprint=fingerprints["pass1"], error_code=exc.code,
                    retryable=exc.retryable, latency_ms=(time.perf_counter() - started) * 1000,
                    correlation_id=correlation_id,
                )
                return {"state": "FAILED_RETRYABLE" if exc.retryable else "FAILED_PERMANENT", "error_code": exc.code, "pass": "pass1"}
            except ValueError as exc:
                store.commit_failure(
                    window_id, "pass1", fingerprint=fingerprints["pass1"], error_code="PASS1_SCHEMA_INVALID",
                    retryable=False, latency_ms=(time.perf_counter() - started) * 1000, correlation_id=correlation_id,
                )
                return {"state": "FAILED_PERMANENT", "error_code": "PASS1_SCHEMA_INVALID", "pass": "pass1", "detail": str(exc)}
            timings["pass1"] = time.perf_counter() - started
            store.commit(window_id, "pass1", pass1, fingerprint=fingerprints["pass1"],
                         latency_ms=timings["pass1"] * 1000, correlation_id=correlation_id)
        actor_ids = [e["id"] for e in pass1["entities"]]

        # Stage3A advisory: one targeted recovery attempt for a suspiciously empty inventory.
        if empty_retry and advisory.get("suggests_actors") and not actor_ids:
            try:
                retry_prompt = contract.PASS1_PROMPT + (
                    "\nIMPORTANT: motion evidence indicates visible road users. Re-inspect the tiles and list "
                    "every clearly visible vehicle and person; an empty entities list is only correct if truly none."
                )
                pass1 = _run_pass(
                    provider, contract.parse_pass1, retry_prompt, image_url,
                    window_id, index_id, correlation_id, "pass1", contract.PASS1_MAX_TOKENS, 0, sleep,
                )
                actor_ids = [e["id"] for e in pass1["entities"]]
                store.commit(window_id, "pass1", pass1, fingerprint=fingerprints["pass1"],
                             latency_ms=0.0, correlation_id=correlation_id)
            except (QwenError, ValueError):
                pass  # keep the original (possibly empty) inventory

        # ---- Pass 2 ---------------------------------------------------------
        pass2 = saved.get("pass2")
        if pass2 is None:
            started = time.perf_counter()
            try:
                if actor_ids:
                    pass2 = _run_pass(
                        provider, lambda t: contract.parse_pass2(t, actor_ids),
                        contract.pass2_prompt(actor_ids), image_url,
                        window_id, index_id, correlation_id, "pass2", contract.PASS2_MAX_TOKENS, max_retries, sleep,
                    )
                else:
                    pass2 = {"schema_version": contract.VERSION, "entities": []}
                timings["pass2"] = time.perf_counter() - started
                store.commit(window_id, "pass2", pass2, fingerprint=fingerprints["pass2"],
                             latency_ms=timings["pass2"] * 1000, correlation_id=correlation_id)
            except QwenError as exc:
                store.commit_failure(
                    window_id, "pass2", fingerprint=fingerprints["pass2"], error_code=exc.code,
                    retryable=exc.retryable, latency_ms=(time.perf_counter() - started) * 1000,
                    correlation_id=correlation_id,
                )
                return {"state": "FAILED_RETRYABLE" if exc.retryable else "FAILED_PERMANENT", "error_code": exc.code, "pass": "pass2"}
            except (ValueError, contract.PassError) as exc:
                code = getattr(exc, "code", "PASS2_INVALID")
                store.commit_failure(
                    window_id, "pass2", fingerprint=fingerprints["pass2"], error_code=code,
                    retryable=False, latency_ms=(time.perf_counter() - started) * 1000, correlation_id=correlation_id,
                )
                return {"state": "FAILED_PERMANENT", "error_code": code, "pass": "pass2"}

        # ---- Pass 3 (conditional) ------------------------------------------
        pass3 = saved.get("pass3")
        run_p3, reason = temporal_recommended(advisory, clip_category)
        if pass3 is None:
            if not run_p3:
                store.commit_skip(window_id, "pass3", fingerprint=fingerprints["pass3"], reason=reason)
                pass3 = {"schema_version": contract.VERSION, "ego": None, "events": [], "scene_relations": []}
                timings["pass3_skipped"] = reason
            else:
                started = time.perf_counter()
                try:
                    pass3 = _run_pass(
                        provider, lambda t: contract.parse_pass3(t, actor_ids, [i["id"] for i in pass1["infrastructure"]]),
                        contract.pass3_prompt(actor_ids), image_url,
                        window_id, index_id, correlation_id, "pass3", contract.PASS3_MAX_TOKENS, max_retries, sleep,
                    )
                    timings["pass3"] = time.perf_counter() - started
                    store.commit(window_id, "pass3", pass3, fingerprint=fingerprints["pass3"],
                                 latency_ms=timings["pass3"] * 1000, correlation_id=correlation_id)
                except QwenError as exc:
                    store.commit_failure(
                        window_id, "pass3", fingerprint=fingerprints["pass3"], error_code=exc.code,
                        retryable=exc.retryable, latency_ms=(time.perf_counter() - started) * 1000,
                        correlation_id=correlation_id,
                    )
                    return {"state": "FAILED_RETRYABLE" if exc.retryable else "FAILED_PERMANENT", "error_code": exc.code, "pass": "pass3"}
                except (ValueError, contract.PassError) as exc:
                    code = getattr(exc, "code", "PASS3_INVALID")
                    store.commit_failure(
                        window_id, "pass3", fingerprint=fingerprints["pass3"], error_code=code,
                        retryable=False, latency_ms=(time.perf_counter() - started) * 1000, correlation_id=correlation_id,
                    )
                    return {"state": "FAILED_PERMANENT", "error_code": code, "pass": "pass3"}

        # ---- Deterministic merge + final commit -----------------------------
        final = contract.merge(pass1, pass2, pass3 if pass3.get("ego") else None)
        store.commit(window_id, "final", final, fingerprint=fingerprints["final"], latency_ms=0.0,
                     correlation_id=correlation_id)
        return {
            "state": "FINAL_SUCCESS",
            "value": final,
            "timings": timings,
            "pass3_ran": bool(run_p3),
            "pass3_reason": reason,
            "correlation_id": correlation_id,
        }


def _run_pass(provider, parse, prompt, image_url, window_id, index_id, correlation_id, pass_id, max_tokens, retries, sleep):
    """Bounded generation. Retries transient transport AND schema failures (a truncated
    JSON response is usually fixed on a second attempt); semantic validation (frozen ids)
    is applied by the caller's `parse` and is never retried."""
    last = None
    for attempt in range(retries + 1):
        try:
            text = provider.complete(
                prompt, [image_url], structured=True, window_id=window_id, index_id=index_id,
                correlation_id=correlation_id, attempt=attempt + 1, max_tokens=max_tokens,
                contract=_Contract(provider.contract, pass_id), pass_id=pass_id,
            )
            return parse(text)
        except QwenError as exc:
            last = exc
            retryable = exc.retryable or exc.code == "QWEN_SCHEMA_FAILURE"
            if not retryable or attempt >= retries:
                break
            sleep(min(4.0, 0.5 * 2**attempt))
    raise last or QwenError("QWEN_SCHEMA_FAILURE")


class _Contract:
    """Adapter exposing a per-pass SCHEMA to the provider's structured formatting."""

    def __init__(self, base, pass_id):
        self._base = base
        self.VERSION = contract.VERSION
        self.SCHEMA = getattr(contract, {"pass1": "PASS1_SCHEMA", "pass2": "PASS2_SCHEMA", "pass3": "PASS3_SCHEMA"}[pass_id])
        self.MAX_TOKENS = {"pass1": contract.PASS1_MAX_TOKENS, "pass2": contract.PASS2_MAX_TOKENS, "pass3": contract.PASS3_MAX_TOKENS}[pass_id]

    def parse_metadata(self, text):
        # Wire-level shape/schema check only; the orchestrator applies semantic
        # validation (frozen ids, references) after parsing.
        value = json.loads(text)
        contract.validate_shape(value, self.SCHEMA, "pass")
        return value
