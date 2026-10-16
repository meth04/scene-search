import contextvars
import threading

from temporalio import activity
from temporalio.exceptions import ApplicationError


class WindowActivities:
    def __init__(self, indexing, exports):
        self.indexing, self.exports = indexing, exports

    @activity.defn(name="index_window_group")
    def index_group(self, job_id: str, start: int) -> dict:
        stop = threading.Event()
        context = contextvars.copy_context()
        def keep_alive():
            while not stop.wait(10):
                context.run(activity.heartbeat, {"job_id": job_id, "start": start})
        thread = threading.Thread(target=keep_alive, daemon=True)
        thread.start()
        try:
            activity.heartbeat({"job_id": job_id, "start": start})
            return self.indexing.process_group(job_id, start, activity.heartbeat)
        except (ValueError, FileNotFoundError, PermissionError) as exc:
            self.indexing._job_status(job_id, "failed")
            raise ApplicationError(str(exc), type="ContractError", non_retryable=True) from exc
        except Exception:
            self.indexing._job_status(job_id, "failed")
            raise
        finally:
            stop.set()
            thread.join(timeout=2)

    @activity.defn(name="validate_window_index")
    def validate(self, job_id: str) -> str:
        job = self.indexing.jobs.get_job(job_id)
        report = self.indexing.validate(job.index_version)
        if not report.passed:
            raise ApplicationError("index validation failed", type="ContractError", non_retryable=True)
        return "ready"

    def _export_phase(self, job_id, phase):
        from src.retrieval.domain.errors import ForbiddenError, NotFoundError
        try:
            return self.exports.run_export(job_id, phase)["status"]
        except (ForbiddenError, NotFoundError) as exc:
            raise ApplicationError(str(exc), type="SourceError", non_retryable=True) from exc

    @activity.defn(name="validate_window_export")
    def export_validate(self, job_id: str) -> str:
        return self._export_phase(job_id, "validate")

    @activity.defn(name="build_window_manifest")
    def export_build(self, job_id: str) -> str:
        return self._export_phase(job_id, "build")

    @activity.defn(name="publish_window_manifest")
    def export_publish(self, job_id: str) -> str:
        return self._export_phase(job_id, "publish")
