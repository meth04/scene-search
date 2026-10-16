from __future__ import annotations

import argparse
import json
import time

from src.indexing.application.commands import RunIndexCommand
from src.indexing.application.window_orchestrator import WindowIndexingOrchestrator
from src.indexing.config import PipelineConfig, load_pipeline_config
from src.indexing.validation.reports import report_json, save_acceptance_report


def _orchestrator(config: PipelineConfig) -> WindowIndexingOrchestrator:
    """Construct the window indexer through the shared bootstrap.

    Args:
        config: Validated configuration; saved snapshots retain their original serialized fields.

    Returns:
        Configured window orchestrator.
    """
    from src.bootstrap import build_indexing

    return build_indexing(config)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scenesearch-index", description="Offline scene indexing")
    groups = parser.add_subparsers(dest="group", required=True)
    models = groups.add_parser("models")
    model_commands = models.add_subparsers(dest="command", required=True)
    for name in ("prefetch", "validate"):
        command = model_commands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "validate":
            command.add_argument("--output", default="data/reports/model-acceptance.json")
    index = groups.add_parser("index")
    index_commands = index.add_subparsers(dest="command", required=True)
    for name in ("run", "resume"):
        command = index_commands.add_parser(name)
        inputs = command.add_mutually_exclusive_group(required=name == "run")
        inputs.add_argument("--manifest")
        inputs.add_argument("--folder", help="Automatically index videos in this folder")
        if name == "resume":
            command.add_argument("--job-id")
        command.add_argument("--config", required=True)
    watch = index_commands.add_parser("watch")
    watch.add_argument("--folder", default="data/videos")
    watch.add_argument("--config", required=True)
    watch.add_argument("--interval", type=float, default=5)
    validate = index_commands.add_parser("validate")
    validate.add_argument("--index-version", required=True)
    validate.add_argument("--config", required=True)
    for name in ("publish", "rollback"):
        command = index_commands.add_parser(name)
        command.add_argument("--index-version", required=True)
        command.add_argument("--config", required=True)
        command.add_argument("--alias", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and execute the requested supported operation.

    Args:
        argv: Command-line arguments, or None to read process arguments.

    Returns:
        Process exit code for CLI commands.

    Raises:
        ValueError: If persisted inputs, configuration or operation invariants are invalid.
    """
    args = build_parser().parse_args(argv)
    config = load_pipeline_config(args.config)
    if args.group == "models":
        from src.indexing.models.driveretriever_adapter import build_driveretriever

        if args.command == "prefetch":
            from huggingface_hub import snapshot_download

            path = snapshot_download(
                repo_id="google/siglip2-base-patch16-224",
                revision=config.driveretriever.revision,
                cache_dir="data/model-cache/huggingface",
            )
            print(json.dumps({"backbone_snapshot": str(path)}))
            return 0
        import numpy as np

        encoder = build_driveretriever(config.driveretriever)
        text = encoder.encode_text(["a vehicle on a road"])[0]
        video = encoder.encode_video_batch([[np.zeros((224, 224, 3), dtype=np.uint8)] * 12])[0]
        report = {
            "passed": text.dimension == video.dimension == 512,
            "is_fake": encoder.contract.is_fake,
            "embedding_contract": encoder.contract.model_dump(mode="json"),
            "contract_hash": encoder.contract.hash,
            "quality_status": "not_evaluated",
        }
        save_acceptance_report(report, args.output)
        print(json.dumps(report, indent=2))
        return 0 if report["passed"] else 1
    orchestrator = _orchestrator(config)
    if args.command == "watch":
        from src.indexing.application.folder_import import folder_command, scan_videos

        if args.interval <= 0:
            raise ValueError("interval must be greater than zero")
        previous = None
        completed = None
        try:
            while True:
                try:
                    paths = scan_videos(args.folder, config)
                    snapshot = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
                    if snapshot and snapshot == previous and snapshot != completed:
                        command = folder_command(args.folder, config)
                        job = orchestrator.run(command)
                        print(job.model_dump_json(), flush=True)
                        # Leave failures available for an explicit resume; retry after input changes.
                        completed = snapshot
                    previous = snapshot
                except Exception as exc:
                    print(json.dumps({"error": str(exc)}), flush=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    if args.command in {"run", "resume"}:
        if args.command == "resume" and args.job_id:
            existing = orchestrator.jobs.get_job(args.job_id)
            if not existing:
                raise ValueError("JOB_NOT_FOUND")
            if existing.status in {"failed", "paused"}:
                orchestrator.control(args.job_id, "retry" if existing.status == "failed" else "resume")
            job = orchestrator.run_job(args.job_id)
            print(job.model_dump_json(indent=2))
            return 0 if job.status != "failed" else 1
        if args.folder:
            from src.indexing.application.folder_import import folder_command

            command = folder_command(args.folder, config)
            if command is None:
                print(json.dumps({"status": "empty", "folder": args.folder}))
                return 0
        elif args.manifest:
            command = RunIndexCommand.from_files(args.manifest, args.config)
        else:
            raise ValueError("resume requires --job-id or --manifest")
        job = orchestrator.resume(command) if args.command == "resume" else orchestrator.run(command)
        print(job.model_dump_json(indent=2))
        return 0 if job.status != "failed" else 1
    if args.command == "validate":
        report = orchestrator.validate(args.index_version, config)
        print(report_json(report))
        return 0 if report.passed else 1
    alias = args.alias or config.storage.production_alias
    previous = (
        orchestrator.publish(args.index_version, alias)
        if args.command == "publish"
        else orchestrator.rollback(args.index_version, alias)
    )
    print(json.dumps({"active": args.index_version, "previous_collection": previous}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
