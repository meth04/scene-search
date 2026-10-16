"""Launch Scene Search and its user video library in the existing engine environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("EMBEDDINGGEMMA2_PYTHON", str(ROOT / ".venv_embeddinggemma2/Scripts/python.exe"))
    import uvicorn

    from demo.backend import create_app

    uvicorn.run(create_app(), host="127.0.0.1", port=args.port, access_log=False)


if __name__ == "__main__":
    main()
