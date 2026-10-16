from __future__ import annotations

import math

from src.indexing.models.driveretriever_adapter import checked_embedding


class WindowVectorStore:
    def __init__(self, repository, url=None, api_key=None):
        self.repository = repository
        self.client = None
        if url:
            from qdrant_client import QdrantClient
            self.client = QdrantClient(url=url, api_key=api_key, timeout=30)

    @staticmethod
    def collection_name(version):
        return f"scenes_driveretriever_{version}"

    def ensure(self, version: str, contract_hash: str, *, metadata_enabled: bool = False) -> str:
        """Create fixed vector geometry and optional nested keyword indexes.

        Args:
            version: Exact index version naming the collection.
            contract_hash: Pinned DriveRetriever embedding contract.
            metadata_enabled: Whether both scene label indexes are required.

        Returns:
            Collection name; local simulation does not claim server verification.

        Raises:
            ValueError: If vector geometry or required server payload indexes differ.
        """
        name = self.collection_name(version)
        existing = self.repository.get("collection", name)
        if existing and existing != {"dimension": 512, "distance": "Cosine", "contract_hash": contract_hash}:
            raise ValueError("EMBEDDING_CONTRACT_MISMATCH: existing collection belongs to another contract")
        if self.client:
            from qdrant_client import models
            if not self.client.collection_exists(name):
                self.client.create_collection(name, vectors_config=models.VectorParams(size=512, distance=models.Distance.COSINE))
            config = self.client.get_collection(name).config.params.vectors
            if not isinstance(config, models.VectorParams) or config.size != 512 or config.distance != models.Distance.COSINE:
                raise ValueError("EMBEDDING_CONTRACT_MISMATCH: collection geometry differs")
            for field in ("index_version", "tenant_id", "access_scope_id", "video_id", "camera", "dataset_version",
                          "media_policy_version", "privacy_status", "region", "embedding_contract_hash"):
                self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
            for field in ("scene_start_ms", "scene_end_ms"):
                self.client.create_payload_index(name, field, models.PayloadSchemaType.FLOAT, wait=True)
            self.client.create_payload_index(name, "captured_at", models.PayloadSchemaType.DATETIME, wait=True)
            if metadata_enabled:
                for field in ("metadata.weather.value", "metadata.lighting.value"):
                    self.client.create_payload_index(name, field, models.PayloadSchemaType.KEYWORD, wait=True)
                if not self.metadata_indexes_verified(name):
                    raise ValueError("METADATA_INDEX_MISSING: nested keyword indexes are required")
        self.repository.put("collection", name, {"dimension": 512, "distance": "Cosine", "contract_hash": contract_hash})
        return name

    def metadata_indexes_verified(self, collection: str) -> bool | None:
        """Inspect live Qdrant payload schema without pretending local is a server.

        Args:
            collection: Collection whose nested keyword indexes are inspected.

        Returns:
            None for the local store; otherwise whether both real indexes exist.
        """
        if self.client is None:
            return None
        from qdrant_client import models

        schema = self.client.get_collection(collection).payload_schema
        return all(key in schema and schema[key].data_type == models.PayloadSchemaType.KEYWORD
                   for key in ("metadata.weather.value", "metadata.lighting.value"))

    def upsert(self, collection, window, asset):
        catalog = self.repository.get("catalog", window.index_version)
        if not catalog or catalog["state"] != "staging":
            raise ValueError("INDEX_FROZEN")
        checked_embedding(window.embedding.vector)
        point = {"id": window.scene_id, "vector": window.embedding.vector, "payload": window.payload(asset)}
        if self.client:
            from qdrant_client.models import PointStruct
            self.client.upsert(collection, [PointStruct(**point)], wait=True)
        else:
            self.repository.put("vector:"+collection, window.scene_id, point)

    def points(self, collection):
        if not self.client:
            return self.repository.list("vector:"+collection)
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(collection, limit=256, offset=offset, with_payload=True, with_vectors=True)
            points.extend({"id": str(p.id), "vector": p.vector, "payload": p.payload} for p in batch)
            if offset is None:
                break
        return points

    def get(self, collection, scene_id):
        if not self.client:
            return self.repository.get("vector:"+collection, scene_id)
        points = self.client.retrieve(collection, [scene_id], with_vectors=True, with_payload=True)
        return {"id": str(points[0].id), "vector": points[0].vector, "payload": points[0].payload} if points else None

    def query(self, collection, vector, filters, limit):
        checked_embedding(vector)
        if self.client:
            points = self.client.query_points(collection, query=vector, query_filter=filters.qdrant(), limit=limit,
                                              with_payload=True, with_vectors=False).points
            return [{"id": str(p.id), "score": p.score, "payload": p.payload} for p in points]
        hits = []
        for point in self.points(collection):
            if filters.matches(point["payload"]):
                score = math.fsum(a*b for a, b in zip(vector, point["vector"], strict=True))
                hits.append({"id": point["id"], "score": score, "payload": point["payload"]})
        return sorted(hits, key=lambda p: (-p["score"], p["id"]))[:limit]
