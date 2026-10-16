"""Start a local retrieval test using the existing indexing configuration."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from src.indexing.config import load_pipeline_config
from src.retrieval.config import RetrievalConfig, SearchConfig


def retrieval_config(indexing):
    if (
        indexing.model_family != "driveretriever"
        or indexing.storage.provider != "local"
        or indexing.workflow.provider != "local"
    ):
        raise ValueError("Local retrieval test requires local DriveRetriever indexing")
    return RetrievalConfig(
        model_family="driveretriever",
        provider="local",
        root_dir=indexing.storage.root_dir,
        driveretriever=indexing.driveretriever.model_copy(deep=True),
        workflow=indexing.workflow.model_copy(deep=True),
        search=SearchConfig(production_alias=indexing.storage.production_alias, ranking_config_version="window-nms-v1"),
    )


def prepare_index(orchestrator, config, index_version=None):
    if index_version is None:
        jobs = [job for job in orchestrator.jobs.list("job") if job["status"] == "ready"]
        if not jobs:
            raise ValueError("No ready index. Run indexing first.")
        index_version = max(jobs, key=lambda job: (job["created_at"], job["job_id"]))["index_version"]
    report = orchestrator.validate(index_version, config)
    if not report.passed:
        raise ValueError("Index validation failed; inspect index validate output")
    alias = config.storage.production_alias
    active = orchestrator.jobs.active(alias)
    if not active or active["index_version"] != index_version:
        orchestrator.publish(index_version, alias)
    return index_version


def serve(args):
    from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator
    from src.retrieval.application.window_search_service import WindowSearchService

    config = load_pipeline_config(args.indexing_config)
    retrieval = retrieval_config(config)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    # Fail before changing the active index if another server owns this port.
    import socket

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", args.port))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    orchestrator = WindowIndexingOrchestrator(config)
    version = prepare_index(orchestrator, config, args.index_version)
    search = WindowSearchService(retrieval)
    search._encoder(search._catalog(version))
    output = Path("data/reports/retrieval-test")
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "retrieval-config.json"
    config_path.write_text(retrieval.model_dump_json(indent=2), encoding="utf-8")
    (output / "startup.json").write_text(
        json.dumps(
            {
                "index_version": version,
                "indexing_config": str(Path(args.indexing_config).resolve()),
                "retrieval_config": str(config_path.resolve()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    env = dict(
        os.environ,
        INDEXING_CONFIG=str(Path(args.indexing_config).resolve()),
        RETRIEVAL_CONFIG=str(config_path.resolve()),
    )
    children = []
    try:
        children.append(
            subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "src.main:app", "--host", "127.0.0.1", "--port", str(args.port)],
                env=env,
            )
        )
        children.append(
            subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "src.indexing.workflows.worker",
                    "--indexing-config",
                    env["INDEXING_CONFIG"],
                    "--retrieval-config",
                    env["RETRIEVAL_CONFIG"],
                ],
                env=env,
            )
        )
        print(f"Index: {version}\nUI: http://127.0.0.1:{args.port}/ui/driveretriever.html\nCtrl+C to stop.", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        raise RuntimeError("API or worker stopped; see output above")
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
