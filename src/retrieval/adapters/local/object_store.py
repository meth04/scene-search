from __future__ import annotations

import hashlib
import io
import os
import shutil
from pathlib import Path

from src.retrieval.domain.entities import ObjectInfo


class LocalRetrievalObjectStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / key).resolve()
        path.relative_to(self.root)
        return path

    def stat(self, key: str) -> ObjectInfo | None:
        path = self._path(key)
        if not path.is_file():
            return None
        content = path.read_bytes()
        return ObjectInfo(
            key=key,
            size_bytes=len(content),
            sha256=hashlib.sha256(content).hexdigest(),
        )

    def open_read(self, key: str):
        return self._path(key).open("rb")

    def put_staging(self, key: str, content, metadata: dict[str, str]) -> ObjectInfo:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        value = content.read()
        digest = hashlib.sha256(value).hexdigest()
        if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"immutable artifact collision: {key}")
        if not path.exists():
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(value)
            os.replace(temporary, path)
        return ObjectInfo(key=key, size_bytes=len(value), sha256=digest, metadata=metadata)

    def publish(self, staging_key: str, published_key: str) -> ObjectInfo:
        source, target = self._path(staging_key), self._path(published_key)
        if not source.is_file():
            raise FileNotFoundError(staging_key)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"immutable artifact collision: {published_key}")
        if not target.exists():
            temporary = target.with_suffix(target.suffix + ".tmp")
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
        return self.stat(published_key)  # type: ignore[return-value]

    def presign_get(self, key: str, ttl_seconds: int) -> str:
        if not self.exists(key):
            raise FileNotFoundError(key)
        # Local URLs intentionally mimic an expiring grant and never expose keys in DB.
        return f"{self._path(key).as_uri()}?ttl={ttl_seconds}"

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def put_bytes(self, key: str, value: bytes, metadata: dict[str, str] | None = None) -> ObjectInfo:
        return self.put_staging(key, io.BytesIO(value), metadata or {})
