"""Bounded offline geometry/privacy phase, isolated from the resident EG2 model."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from demo.product_privacy import prepare, validate
from hybrid_v2.embeddinggemma2 import ROOT, digest
from src.privacy.tracks import write_json


def main(plan_path):
    import socket

    def blocked(*args, **kwargs):
        raise RuntimeError("PRODUCT_OFFLINE_NETWORK_PROHIBITED")

    socket.socket.connect = blocked
    socket.create_connection = blocked
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    sys.path.insert(0, str(ROOT / "scripts"))
    from demo.product_geometry import detector

    for item in plan["rows"]:
        folder = Path(item["cache"])
        receipt_path = folder / "offline_receipt.json"
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            assert receipt["identity"] == item["identity"]
            assert all(digest(folder / n) == h for n, h in receipt["hashes"].items())
            validate(item["raw"], folder / "privacy", item["duration"])
            print(json.dumps({"row": item["row"], "stage": "offline_reused"}), flush=True)
            continue
        except (OSError, ValueError, KeyError, AssertionError):
            pass
        # Invalid cache generations are quarantined, never silently reused.
        if folder.exists():
            import shutil

            shutil.rmtree(folder)
        folder.mkdir(parents=True)
        prepare(item["raw"], folder / "privacy", item["duration"])
        detector([{"source_path": item["inference"], "source_sha256": digest(item["inference"])}], folder / "geometry")
        write_json(
            receipt_path,
            {
                "identity": item["identity"],
                "hashes": {
                    str(p.relative_to(folder)): digest(p)
                    for p in folder.rglob("*")
                    if p.is_file() and p != receipt_path
                },
            },
        )
        print(json.dumps({"row": item["row"], "stage": "offline_complete"}), flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
