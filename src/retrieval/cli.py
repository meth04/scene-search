from __future__ import annotations

import argparse


def main(argv=None) -> int:
    """Parse command-line arguments and execute the requested supported operation.

    Args:
        argv: Command-line arguments, or None to read process arguments.

    Returns:
        Process exit code for CLI commands.
    """
    parser = argparse.ArgumentParser(prog="retrieval")
    sub = parser.add_subparsers(dest="group", required=True)
    test = sub.add_parser("test")
    test.add_argument("--indexing-config", default="configs/indexing/local.yaml")
    test.add_argument("--index-version", help="Use this index instead of the newest ready job")
    test.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if args.group == "test":
        from src.retrieval.local_test import serve

        return serve(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
