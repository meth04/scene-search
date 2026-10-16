"""Run with python -m src.indexing.workflows.worker --indexing-config ... --retrieval-config ..."""

from __future__ import annotations

import argparse
import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

from src.bootstrap import build_indexing
from src.indexing.config import load_pipeline_config
from src.retrieval.adapters.factory import build_services
from src.retrieval.config import load_retrieval_config


async def dispatch(indexing, exports, client=None):
    jobs = await asyncio.to_thread(indexing.jobs.list, "job")
    for job in jobs:
        if job["status"] != "queued":
            continue
        if client:
            from temporalio.common import WorkflowIDReusePolicy
            from temporalio.exceptions import WorkflowAlreadyStartedError

            from src.indexing.workflows.workflow import WindowIndexWorkflow

            try:
                await client.start_workflow(
                    WindowIndexWorkflow.run,
                    job["job_id"],
                    id="index/" + job["job_id"],
                    task_queue=indexing.config.workflow.task_queue,
                    id_reuse_policy=WorkflowIDReusePolicy.ALLOW_DUPLICATE_FAILED_ONLY,
                )
            except WorkflowAlreadyStartedError:
                pass
        else:
            try:
                await asyncio.to_thread(indexing.run_job, job["job_id"])
            except Exception:
                # Failure is durable in the job record; other jobs still progress.
                import logging

                logging.getLogger("indexing").exception("index job failed", extra={"job_id": job["job_id"]})
    await exports.dispatch_outbox(client)


async def main_async(args):
    """Start metrics and run the configured local or Temporal worker loop.

    Args:
        args: Parsed worker command-line options.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    from prometheus_client import start_http_server

    from src.indexing.telemetry import configure_tracing

    configure_tracing()
    start_http_server(int(os.getenv("WORKER_METRICS_PORT", "9464")))
    config = load_pipeline_config(args.indexing_config)
    retrieval = load_retrieval_config(args.retrieval_config)
    if config.storage.root_dir != retrieval.root_dir or config.workflow != retrieval.workflow:
        raise ValueError("indexing/retrieval worker storage and workflow configurations must match")
    indexing = build_indexing(config)
    from src.indexing.models.driveretriever_adapter import build_driveretriever

    indexing.encoder = build_driveretriever(config.driveretriever)
    exports = build_services(retrieval).exports
    if config.workflow.provider == "local":
        while True:
            await dispatch(indexing, exports)
            if args.once:
                return
            await asyncio.sleep(2)
    else:
        from temporalio.client import Client
        from temporalio.worker import Worker

        from src.indexing.workflows.activities import WindowActivities
        from src.indexing.workflows.workflow import ManifestExportWorkflow, WindowIndexWorkflow

        client = await Client.connect(config.workflow.address, namespace=config.workflow.namespace)
        activities = WindowActivities(indexing, exports)
        with ThreadPoolExecutor(max_workers=2) as executor:
            async with Worker(
                client,
                task_queue=config.workflow.task_queue,
                workflows=[WindowIndexWorkflow, ManifestExportWorkflow],
                activities=[
                    activities.index_group,
                    activities.validate,
                    activities.export_validate,
                    activities.export_build,
                    activities.export_publish,
                ],
                activity_executor=executor,
                max_concurrent_activities=1,
            ):
                while True:
                    await dispatch(indexing, exports, client)
                    await asyncio.sleep(2)


def main():
    parser = argparse.ArgumentParser(description="DriveRetriever index/export worker")
    parser.add_argument("--indexing-config", required=True)
    parser.add_argument("--retrieval-config", required=True)
    parser.add_argument("--once", action="store_true", help="drain local queue once")
    asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    main()
