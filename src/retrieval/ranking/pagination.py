from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from src.retrieval.domain.errors import RetrievalError


class SignedCursorService:
    def __init__(self, secret: str, ttl_seconds: int = 3600, clock=None):
        if not secret:
            raise ValueError("cursor signing secret cannot be empty")
        self.secret = secret.encode()
        self.ttl_seconds = ttl_seconds
        self.clock = clock or (lambda: datetime.now(UTC))

    def sign(self, run_id: str, index_version: str, last_rank: int, result_hash: str) -> str:
        payload = {
            "run_id": run_id,
            "index_version": index_version,
            "last_rank": last_rank,
            "result_hash": result_hash,
            "expires_at": (self.clock() + timedelta(seconds=self.ttl_seconds)).isoformat(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        signature = hmac.new(self.secret, raw, hashlib.sha256).digest()
        return base64.urlsafe_b64encode(raw + signature).decode().rstrip("=")

    def verify(self, token: str) -> dict:
        try:
            raw_signed = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
            raw, supplied = raw_signed[:-32], raw_signed[-32:]
            expected = hmac.new(self.secret, raw, hashlib.sha256).digest()
            if not hmac.compare_digest(supplied, expected):
                raise ValueError
            payload = json.loads(raw)
            if datetime.fromisoformat(payload["expires_at"]) <= self.clock():
                raise RetrievalError("cursor expired", details={"error_code": "CURSOR_EXPIRED"})
            return payload
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError("invalid cursor", details={"error_code": "INVALID_CURSOR"}) from exc
