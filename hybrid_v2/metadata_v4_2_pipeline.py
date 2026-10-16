"""Serial four-pass extraction; immutable freeze, bounded recovery, atomic resume."""
from __future__ import annotations
import hashlib
import json
import time
from datetime import UTC, datetime
from types import SimpleNamespace

import numpy as np

from . import metadata_v4_2 as contract
from .metadata_v4_1_pipeline import MultiPassStore as BaseStore, temporal_recommended
from .qwen_provider import QwenError
from .qwen_store import atomic_json, digest, window_lock

PASSES = ("pass1", "pass2a", "pass2b", "pass3", "final")


class MultiPassStore(BaseStore):
    def __init__(self, root):
        super().__init__(root, contract_module=contract)

    def commit(self, window_id, name, value, *, fingerprint, latency_ms=None, correlation_id=None, state=None):
        row, receipt = self._paths(window_id, name)
        # Receipt is the commit point. A crash after row write leaves no usable success.
        atomic_json(row, value)
        atomic_json(receipt, {"window_id": window_id, "pass": name, "fingerprint": fingerprint,
                             "state": state or name.upper() + "_SUCCESS", "sha256": digest(row),
                             "latency_ms": latency_ms, "correlation_id": correlation_id,
                             "timestamp": datetime.now(UTC).isoformat()})
        return value

    def _read(self, window_id, name):
        row, receipt = self._paths(window_id, name)
        if not receipt.exists():
            return None, None
        saved = json.loads(receipt.read_text(encoding="utf-8"))
        allowed = {p.upper() + "_SUCCESS" for p in PASSES} | {"PASS3_SKIPPED", "FAILED_RETRYABLE", "FAILED_PERMANENT"}
        if saved.get("state") not in allowed:
            raise ValueError("Invalid V4.2 receipt state")
        if saved["state"].endswith("_SUCCESS") or saved["state"] == "PASS3_SKIPPED":
            if not row.exists() or digest(row) != saved.get("sha256"):
                raise ValueError("Corrupt V4.2 committed pass; refusing replacement")
            return json.loads(row.read_text(encoding="utf-8")), saved
        return None, saved

    def resume(self, window_id, fingerprints):
        values = {}
        for name in PASSES:
            value, saved = self._read(window_id, name)
            if saved and saved["fingerprint"] != fingerprints[name]:
                raise ValueError("V4.2 resume identity mismatch; choose a new output")
            if value is not None or (saved and saved["state"] == "PASS3_SKIPPED"):
                values[name] = value
        return values

    def status(self):
        result = {}
        for name in PASSES:
            for path in (self.root / name / "receipts").glob("*.json"):
                state = json.loads(path.read_text())["state"]
                result[state] = result.get(state, 0) + 1
        return result


def stage3a_advisory(window, tracks, masks):
    """Read actual aggregate features; slot IDs are hints, never physical identity."""
    from stage3a.stage3a_motion import FEATURE_NAMES, MOTION_CLASSES
    features, mask = tracks[window["row"]], masks[window["row"]]
    if features.shape != (24, 29) or mask.shape != (24,):
        raise ValueError("Stage3A row/shape mismatch")
    hints = []
    for slot in np.flatnonzero(mask):
        f = features[slot]
        if not np.isfinite(f).all() or f[:6].sum() != 1:
            continue
        cls = MOTION_CLASSES[int(np.argmax(f[:6]))]
        values = dict(zip(FEATURE_NAMES, map(float, f)))
        hints.append({"aggregate_slot": f"slot_{slot:02d}", "coarse_category": "person" if cls == "person" else "vehicle",
                      "coarse_type": cls, "mean_center": [round(values["mean_cx"], 3), round(values["mean_cy"], 3)],
                      "start_center": [round(values["start_cx"], 3), round(values["start_cy"], 3)],
                      "end_center": [round(values["end_cx"], 3), round(values["end_cy"], 3)],
                      "visible_ratio": round(values["visible_ratio"], 3),
                      "displacement": round(float(np.hypot(values["dx"], values["dy"])), 3)})
    return {"has_motion": any(h["displacement"] > .01 for h in hints), "suggests_actors": bool(hints),
            "track_count": len(hints), "anchor_hints": hints, "advisory_only": True,
            "exact_mapping_safe": False, "original_track_ids_available": False}


def _run_pass(provider, name, prompt, image_url, inventory, window_id, index_id, correlation_id,
              retries=1, first_attempt=1, parse_override=None):
    parse = parse_override or (lambda t: contract.parse_pass(t, name, inventory))
    adapter = SimpleNamespace(VERSION=contract.VERSION, SCHEMA=contract.schema_for(name, inventory),
                              parse_metadata=parse)
    for attempt in range(first_attempt, first_attempt + min(retries, 1) + 1):
        try:
            # Shorter retry asks for compact facts, never changes the frozen identity set.
            retry = attempt > 1
            text = provider.complete(prompt + ("\nShort targeted retry: compact JSON; unknown for weak fields; no new IDs." if retry else ""),
                [image_url], structured=True, window_id=window_id, index_id=index_id,
                correlation_id=correlation_id, attempt=attempt, pass_id=name, contract=adapter,
                max_tokens=int(contract.BUDGETS[name] * (.85 if retry else 1)))
            return parse(text)
        except (QwenError, ValueError) as exc:
            # Semantic unknown-ID failures are hard failures, not silently repaired.
            if isinstance(exc, ValueError) or not exc.retryable or attempt == first_attempt + min(retries, 1):
                raise


def generate_window(window_id, image_url, provider, store, *, index_id, source_identity, settings,
                    advisory, clip_category=None):
    identity = {"version": contract.VERSION, "prompt": contract.COMBINED_PROMPT_SHA256,
                "image_sha256": hashlib.sha256(image_url.encode()).hexdigest(), "source": source_identity,
                "settings": settings, "provider": provider.config.public(), "advisory": advisory,
                "clip_category": clip_category}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    fingerprints = {p: fingerprint for p in PASSES}
    correlation_id = fingerprint[:32]
    with window_lock(store.root / "locks" / (window_id + ".lock")):
        saved = store.resume(window_id, fingerprints)
        if "final" in saved:
            return {"state": "FINAL_SUCCESS", "resumed": True, "value": saved["final"]}
        current = "pass1"
        started = time.perf_counter()
        timings = {}
        try:
            p1 = saved.get("pass1")
            if p1 is None:
                hints = json.dumps(advisory, separators=(",", ":"))
                prompt = contract.PROMPTS["pass1"] + "\nCoarse Stage3A hints: " + hints
                before = len(provider.events)
                p1 = _run_pass(provider, "pass1", prompt, image_url, (), window_id, index_id, correlation_id)
                used = len(provider.events) - before
                suspicion = contract.duplicate_suspicions(p1["entities"])
                diag = {"suspected_duplicates": suspicion, "reconciled_duplicates": [],
                        "unresolved_duplicate_warnings": suspicion, "empty_retry": False}
                # A total of two requests at most for Pass1, including transport/recovery.
                if used < 2 and advisory["suggests_actors"] and not p1["entities"]:
                    diag["empty_retry"] = True
                    p1 = _run_pass(provider, "pass1", prompt + "\nObject branch indicates visible actors. Re-examine frames; inventory only; do not fabricate.",
                                   image_url, (), window_id, index_id, correlation_id, retries=0, first_attempt=2)
                elif used < 2 and suspicion:
                    original = p1
                    try:
                        corrected = _run_pass(provider, "pass1", prompt + "\nReconcile ONLY visually certain duplicates in this inventory. Keep every distinct actor, no new or changed records. Return same scene/infrastructure and subset of original actors: " + json.dumps(original),
                            image_url, (), window_id, index_id, correlation_id, retries=0, first_attempt=2,
                            parse_override=lambda t: contract.validate_reconciliation(contract.parse_pass(t, "pass1"), original))
                        removed = set(a["id"] for a in original["entities"]) - set(a["id"] for a in corrected["entities"])
                        diag["reconciled_duplicates"] = sorted(removed)
                        p1 = corrected
                    except (QwenError, ValueError) as exc:
                        diag["reconciliation_failure"] = getattr(exc, "code", type(exc).__name__)
                diag["unresolved_duplicate_warnings"] = contract.duplicate_suspicions(p1["entities"])
                atomic_json(store.root / "diagnostics" / (window_id + ".json"), diag)
                timings["pass1"] = time.perf_counter() - started
                # Freeze happens once, after recovery/reconciliation, before any enrichment.
                store.commit(window_id, "pass1", p1, fingerprint=fingerprint, latency_ms=timings["pass1"] * 1000, correlation_id=correlation_id)
            inventory = p1["entities"]
            run3, reason = temporal_recommended(advisory, clip_category)
            values = {}
            for current in ("pass2a", "pass2b", "pass3"):
                started = time.perf_counter()
                if current in saved:
                    values[current] = saved[current]
                    continue
                if current == "pass3" and not run3:
                    values[current] = None
                    store.commit(window_id, current, None, fingerprint=fingerprint, latency_ms=0., state="PASS3_SKIPPED")
                    continue
                if current != "pass3" and not inventory:
                    value = {}
                else:
                    prompt = contract.PROMPTS[current] + "\nFrozen inventory: " + json.dumps(inventory, separators=(",", ":"))
                    value = _run_pass(provider, current, prompt, image_url, inventory, window_id, index_id, correlation_id)
                timings[current] = time.perf_counter() - started
                store.commit(window_id, current, value, fingerprint=fingerprint, latency_ms=timings[current] * 1000, correlation_id=correlation_id)
                values[current] = value
            current = "final"
            final = contract.merge(p1, values["pass2a"], values["pass2b"], values["pass3"])
            store.commit(window_id, "final", final, fingerprint=fingerprint, latency_ms=0., correlation_id=correlation_id)
            return {"state": "FINAL_SUCCESS", "timings": timings, "pass3_ran": run3, "pass3_reason": reason}
        except (QwenError, ValueError) as exc:
            code = getattr(exc, "code", "VALIDATION_FAILURE")
            retryable = exc.retryable if isinstance(exc, QwenError) else not isinstance(exc, contract.ActorIDError)
            store.commit_failure(window_id, current, fingerprint=fingerprint, error_code=code,
                                 retryable=retryable, latency_ms=(time.perf_counter() - started) * 1000, correlation_id=correlation_id)
            return {"state": "FAILED_RETRYABLE" if retryable else "FAILED_PERMANENT", "pass": current,
                    "error_code": code, "detail": str(exc), "timings": timings}
