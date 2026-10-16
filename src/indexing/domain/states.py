from enum import StrEnum


class JobStatus(StrEnum):
    QUEUED = "queued"
    PAUSED = "paused"
    CANCELLED = "cancelled"
    RETIRED = "retired"
    PENDING = "pending"
    RUNNING = "running"
    VALIDATING = "validating"
    READY = "ready"
    PUBLISHED = "published"
    FAILED = "failed"


class SceneStatus(StrEnum):
    PROCESSING = "processing"
    QUARANTINED = "quarantined"
    FAILED = "failed"
    RETRY_PENDING = "retry_pending"
    PREVIEW_WRITTEN = "preview_written"
    DB_STAGED = "db_staged"
    VECTOR_STAGED = "vector_staged"
    RECONCILED = "reconciled"
    CHECKPOINTED = "checkpointed"


SCENE_TRANSITIONS: dict[SceneStatus, set[SceneStatus]] = {
    SceneStatus.PROCESSING: {
        SceneStatus.QUARANTINED,
        SceneStatus.FAILED,
        SceneStatus.RETRY_PENDING,
        SceneStatus.PREVIEW_WRITTEN,
    },
    SceneStatus.RETRY_PENDING: {SceneStatus.PROCESSING},
    SceneStatus.PREVIEW_WRITTEN: {SceneStatus.DB_STAGED},
    SceneStatus.DB_STAGED: {SceneStatus.VECTOR_STAGED},
    SceneStatus.VECTOR_STAGED: {SceneStatus.RECONCILED},
    SceneStatus.RECONCILED: {SceneStatus.CHECKPOINTED},
}


def assert_scene_transition(current: SceneStatus, target: SceneStatus) -> None:
    if target not in SCENE_TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid scene transition: {current} -> {target}")
