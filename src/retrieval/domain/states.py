from enum import StrEnum


class ReviewStatus(StrEnum):
    UNREVIEWED = "unreviewed"
    RELEVANT = "relevant"
    NOT_RELEVANT = "not_relevant"


class ResultSetStatus(StrEnum):
    OPEN = "open"
    ARCHIVED = "archived"


class ExportStatus(StrEnum):
    QUEUED = "queued"
    VALIDATING = "validating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    REVOKED = "revoked"


EXPORT_TRANSITIONS = {
    ExportStatus.QUEUED: {ExportStatus.VALIDATING, ExportStatus.FAILED},
    ExportStatus.VALIDATING: {ExportStatus.RUNNING, ExportStatus.FAILED},
    ExportStatus.RUNNING: {ExportStatus.COMPLETED, ExportStatus.FAILED},
    ExportStatus.COMPLETED: {ExportStatus.EXPIRED, ExportStatus.REVOKED},
    ExportStatus.FAILED: set(),
    ExportStatus.EXPIRED: set(),
    ExportStatus.REVOKED: set(),
}


def assert_export_transition(current: ExportStatus, target: ExportStatus) -> None:
    if target not in EXPORT_TRANSITIONS[current]:
        raise ValueError(f"invalid export transition: {current} -> {target}")
