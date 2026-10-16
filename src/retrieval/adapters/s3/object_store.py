import hashlib
import io
from datetime import timedelta

from src.retrieval.domain.entities import ObjectInfo


class S3RetrievalObjectStore:
    def __init__(self, endpoint, bucket, access_key, secret_key, secure=False):
        from minio import Minio

        self.client = Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)
        self.bucket = bucket
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def stat(self, key):
        try:
            value = self.client.stat_object(self.bucket, key)
        except Exception as exc:
            if getattr(exc, "code", "") in {"NoSuchKey", "NoSuchObject"}:
                return None
            raise
        return ObjectInfo(
            key=key,
            size_bytes=value.size,
            sha256=value.metadata.get("x-amz-meta-sha256", ""),
            metadata=value.metadata,
        )

    def open_read(self, key):
        response = self.client.get_object(self.bucket, key)
        try:
            return io.BytesIO(response.read())
        finally:
            response.close()
            response.release_conn()

    def put_staging(self, key, content, metadata):
        value = content.read()
        digest = hashlib.sha256(value).hexdigest()
        existing = self.stat(key)
        if existing and existing.sha256 != digest:
            raise ValueError(f"immutable artifact collision: {key}")
        if not existing:
            self.client.put_object(
                self.bucket,
                key,
                io.BytesIO(value),
                len(value),
                metadata={**metadata, "sha256": digest},
            )
        return ObjectInfo(key=key, size_bytes=len(value), sha256=digest, metadata=metadata)

    def publish(self, staging_key, published_key):
        from minio.commonconfig import CopySource

        source = self.stat(staging_key)
        if not source:
            raise FileNotFoundError(staging_key)
        existing = self.stat(published_key)
        if existing and existing.sha256 != source.sha256:
            raise ValueError(f"immutable artifact collision: {published_key}")
        if not existing:
            self.client.copy_object(self.bucket, published_key, CopySource(self.bucket, staging_key))
        return self.stat(published_key)

    def presign_get(self, key, ttl_seconds):
        return self.client.presigned_get_object(self.bucket, key, expires=timedelta(seconds=ttl_seconds))

    def exists(self, key):
        return self.stat(key) is not None
