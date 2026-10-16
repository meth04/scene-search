"""Shared record lookup and UTC timestamps for window services."""

from datetime import UTC, datetime

from src.retrieval.domain.errors import NotFoundError


def utc_now():
    """Read the current UTC time without arguments.

    Returns:
        Timezone-aware ISO timestamp string.
    """
    return datetime.now(UTC).isoformat()


def require(value, code="NOT_FOUND"):
    """Reject a missing persisted record while preserving the caller error code.

    Args:
        value: Record or request value to inspect.
        code: Error code used when the record is absent.

    Returns:
        The supplied value when it is not None.

    Raises:
        NotFoundError: If the requested record or source is unavailable.
    """
    if value is None:
        raise NotFoundError(code)
    return value
