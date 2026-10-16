class IndexingError(RuntimeError):
    """Base error carrying a stable operator-facing code."""

    code = "indexing_error"


class RetryableIndexingError(IndexingError):
    code = "retryable"


class ItemIndexingError(IndexingError):
    code = "item_non_retryable"


class FatalIndexingError(IndexingError):
    code = "job_fatal"


class PrivacyGateError(ItemIndexingError):
    code = "privacy_gate_failed"


class ReconciliationError(RetryableIndexingError):
    code = "reconciliation_failed"
