"""Construct application services from validated settings without loading models on import."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.indexing.adapters.window_repository import WindowRepository
from src.indexing.adapters.window_vectors import WindowVectorStore
from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator
from src.retrieval.adapters.local.object_store import LocalRetrievalObjectStore
from src.retrieval.application.window_export_service import WindowExportService
from src.retrieval.application.window_result_set_service import WindowResultSetService
from src.retrieval.application.window_review_service import WindowReviewService
from src.retrieval.application.window_search_service import WindowSearchService


def build_indexing_storage(config, *, repository=None, vectors=None):
    """Resolve indexing backends, reusing explicitly supplied dependencies.

    Args:
        config: Pipeline settings with storage provider and local root.
        repository: Existing repository, or None to construct one.
        vectors: Existing vector store, or None to construct one.

    Returns:
        Shared repository and vector store; model weights are not loaded.

    Raises:
        KeyError: If production settings omit DATABASE_URL.
    """
    database = os.environ["DATABASE_URL"] if config.storage.provider == "production" else None
    repository = repository or WindowRepository(config.storage.root_dir, database)
    vectors = vectors or WindowVectorStore(
        repository, os.getenv("QDRANT_URL") if database else None, os.getenv("QDRANT_API_KEY")
    )
    return repository, vectors


def build_retrieval_storage(config, *, repository=None, vectors=None):
    """Resolve retrieval backends using its required production connection settings.

    Args:
        config: Retrieval settings with backend provider and local root.
        repository: Existing repository, or None to construct one.
        vectors: Existing vector store, or None to construct one.

    Returns:
        Shared repository and vector store; model weights are not loaded.

    Raises:
        KeyError: If DATABASE_URL is missing in production, or QDRANT_URL is
            missing when constructing production vectors.
    """
    database = os.environ["DATABASE_URL"] if config.provider == "production" else None
    repository = repository or WindowRepository(config.root_dir, database)
    vectors = vectors or WindowVectorStore(
        repository, os.environ["QDRANT_URL"] if database else None, os.getenv("QDRANT_API_KEY")
    )
    return repository, vectors


@dataclass(frozen=True)
class RetrievalServices:
    """Services sharing the same window repository and source authorization.

    Attributes:
        search: Query and session operations.
        review: Versioned review operations.
        result_sets: Confirmed-result collection operations.
        exports: Snapshot, export and download operations.
    """

    search: WindowSearchService
    review: WindowReviewService
    result_sets: WindowResultSetService
    exports: WindowExportService


def build_retrieval_services(config):
    """Build the supported retrieval stack, leaving encoder loading lazy.

    Args:
        config: Validated retrieval and backend settings.

    Returns:
        Services sharing one search service and export object store.

    Raises:
        ValueError: If an unvalidated configuration selects a removed model.
        KeyError: If required production connection settings are absent.
    """
    if config.model_family != "driveretriever":
        raise ValueError("only driveretriever is supported")
    search = WindowSearchService(config)
    if config.provider == "local":
        store = LocalRetrievalObjectStore(Path(config.root_dir) / "artifacts")
    else:
        from src.retrieval.adapters.s3.object_store import S3RetrievalObjectStore

        store = S3RetrievalObjectStore(
            os.environ["S3_ENDPOINT"],
            os.getenv("S3_BUCKET", "scene-previews"),
            os.environ["S3_ACCESS_KEY"],
            os.environ["S3_SECRET_KEY"],
        )
    return RetrievalServices(
        search, WindowReviewService(search), WindowResultSetService(search), WindowExportService(search, store)
    )


def build_indexing(config):
    """Build the window indexer without eagerly loading model weights.

    Args:
        config: Validated indexing and backend settings.

    Returns:
        Window orchestrator using the configured repository and vector store.

    Raises:
        ValueError: If an unvalidated configuration selects a removed model.
    """
    if config.model_family != "driveretriever":
        raise ValueError("only driveretriever is supported")
    return WindowIndexingOrchestrator(config)
