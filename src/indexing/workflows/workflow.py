from datetime import timedelta

from temporalio import workflow
from temporalio.common import RetryPolicy


@workflow.defn
class WindowIndexWorkflow:
    @workflow.run
    async def run(self, job_id: str, start: int = 0) -> str:
        for _ in range(100):
            result = await workflow.execute_activity(
                "index_window_group", args=[job_id, start],
                start_to_close_timeout=timedelta(minutes=30), heartbeat_timeout=timedelta(minutes=2),
                retry_policy=RetryPolicy(maximum_attempts=3, initial_interval=timedelta(seconds=2),
                                         non_retryable_error_types=["ContractError", "SourceError"]),
            )
            if result["status"] == "paused":
                await workflow.sleep(timedelta(seconds=5))
                continue
            if result["done"]:
                if result["status"] in {"cancelled", "ready", "published"}:
                    return result["status"]
                return await workflow.execute_activity("validate_window_index", job_id,
                                                       start_to_close_timeout=timedelta(minutes=30),
                                                       retry_policy=RetryPolicy(maximum_attempts=3))
            start = result["next"]
        workflow.continue_as_new(args=[job_id, start])


@workflow.defn
class ManifestExportWorkflow:
    @workflow.run
    async def run(self, job_id: str) -> str:
        for name in ("validate_window_export", "build_window_manifest", "publish_window_manifest"):
            status = await workflow.execute_activity(name, job_id,
                                                     start_to_close_timeout=timedelta(minutes=10),
                                                     retry_policy=RetryPolicy(maximum_attempts=3,
                                                                               non_retryable_error_types=["SourceError"]))
            if status in {"revoked", "expired"}:
                return status
        return status
