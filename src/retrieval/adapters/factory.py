"""Compatibility entry point for retrieval service construction."""

from src.bootstrap import RetrievalServices, build_retrieval_services

__all__ = ["RetrievalServices", "build_services"]


def build_services(config):
    """Build retrieval services using the shared application bootstrap.

    Args:
        config: Validated retrieval settings.

    Returns:
        Search, review, result-set and export services sharing their dependencies.
    """
    return build_retrieval_services(config)
