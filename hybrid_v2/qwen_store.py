"""Crash-safe per-window state: metadata first, verified receipt last."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .qwen_provider import QwenError, jpeg_data_url, redact

STATES = {"PENDING", "RUNNING", "SUCCESS", "FAILED_RETRYABLE", "FAILED_PERMANENT"}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as stream:
            name = stream.name
            json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if name and os.path.exists(name):
            os.unlink(name)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def window_lock(path):
    """OS advisory lock is released on process death; no stale lock-file recovery guesses."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


class MetadataStore:
    def __init__(self, root, *, contract=None):
        from . import qwen_metadata

        self.contract = contract or qwen_metadata
        self.root = Path(root)

    def paths(self, window_id):
        key = str(window_id)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", key):
            raise ValueError("Invalid metadata window id")
        return self.root / "rows" / (key + ".json"), self.root / "receipts" / (key + ".json")

    def generate(self, window_id, frames, provider, request=None):
        request = {**(request or {}), "window_id": str(window_id)}
        urls = [f if isinstance(f, str) else jpeg_data_url(f, request.get("jpeg_quality", 85)) for f in frames]
        # Endpoint/model selection and frame identity bind SUCCESS reuse to the exact job.
        public_config = provider.config.public()
        public_config["model"] = provider.requested_model
        identity = dict(
            schema_version=self.contract.VERSION,
            prompt_sha256=self.contract.PROMPT_SHA256,
            provider=public_config,
            image_sha256=[hashlib.sha256(u.encode()).hexdigest() for u in urls],
            source_identity=request.get("source_identity"),
        )
        if request.get("generation_settings") is not None:
            identity["generation_settings"] = request["generation_settings"]
        fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        row, receipt = self.paths(window_id)
        with window_lock(self.root / "locks" / (str(window_id) + ".lock")):
            saved = json.loads(receipt.read_text(encoding="utf-8")) if receipt.exists() else None
            if saved:
                if saved.get("fingerprint") != fingerprint:
                    raise ValueError("Metadata resume identity mismatch; choose a new output")
                if saved.get("state") not in STATES:
                    raise ValueError("Metadata receipt state invalid")
                if saved["state"] == "SUCCESS":
                    if not row.exists() or digest(row) != saved["sha256"]:
                        raise ValueError("Successful metadata is missing/corrupt; refusing silent replacement")
                    return self.contract.validate_metadata(json.loads(row.read_text(encoding="utf-8")))
                if saved["state"] == "FAILED_PERMANENT":
                    raise QwenError(saved["error_code"], retryable=False)
            record = dict(
                window_id=str(window_id),
                fingerprint=fingerprint,
                identity=identity,
                correlation_id=uuid4().hex,
                state="PENDING",
                timestamp=datetime.now(UTC).isoformat(),
                runs=(saved or {}).get("runs", 0) + 1,
            )
            atomic_json(receipt, record)
            record["state"] = "RUNNING"
            atomic_json(receipt, record)
            try:
                result = provider.generate_metadata(urls, {**request, "correlation_id": record["correlation_id"]})
                self.contract.validate_metadata(result)
                atomic_json(row, result)
                # Validate what was actually written before the final SUCCESS receipt.
                self.contract.validate_metadata(json.loads(row.read_text(encoding="utf-8")))
                record.update(state="SUCCESS", sha256=digest(row))
                atomic_json(receipt, record)
                return result
            except QwenError as exc:
                record.update(state="FAILED_RETRYABLE" if exc.retryable else "FAILED_PERMANENT", error_code=exc.code)
                atomic_json(receipt, redact(record, (provider.config.api_key,)))
                raise

    def status(self):
        counts = dict.fromkeys(sorted(STATES), 0)
        for path in (self.root / "receipts").glob("*.json"):
            saved = json.loads(path.read_text(encoding="utf-8"))
            state = saved["state"]
            if state == "SUCCESS":
                row = self.root / "rows" / path.name
                if not row.exists() or digest(row) != saved["sha256"]:
                    raise ValueError("Invalid SUCCESS receipt")
                self.contract.validate_metadata(json.loads(row.read_text(encoding="utf-8")))
            counts[state] += 1
        return counts
