"""Native two-video acceptance using synthetic media and the installed models."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import cv2
    import numpy as np
    from fastapi.testclient import TestClient

    from demo.backend import create_app

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/install_acceptance")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    os.environ["SCENESEARCH_DEVICE"] = args.device
    destination = args.output.resolve()
    if not destination.is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Acceptance artifacts must be under this installation's outputs")
    destination.mkdir(parents=True, exist_ok=True)
    os.environ["SCENESEARCH_PRODUCT_ROOT"] = str(destination / "runtime")
    videos = []
    for number in range(2):
        path = destination / f"synthetic_{number}.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 96))
        assert writer.isOpened()
        for frame in range(30):
            image = np.full((96, 160, 3), 30 + number * 60, np.uint8)
            cv2.rectangle(image, (frame * 3 % 120, 40), (frame * 3 % 120 + 30, 65), (20, 140, 220), -1)
            writer.write(image)
        writer.release()
        videos.append(path)
    started = time.perf_counter()
    app = create_app(library_root=destination / "library")
    with TestClient(app) as client:
        deadline = time.monotonic() + 300
        while client.get("/ready").status_code != 200:
            if time.monotonic() >= deadline:
                raise RuntimeError("Native encoder did not become ready")
            time.sleep(0.5)
        response = client.post(
            "/api/v1/library/uploads",
            json={
                "folder_name": "Synthetic installation acceptance",
                "files": [{"relative_path": p.name, "size_bytes": p.stat().st_size} for p in videos],
            },
        )
        assert response.status_code == 201, response.text
        record = response.json()
        upload = record["upload_id"]
        base = "/api/v1/library/uploads/" + upload
        for path, file in zip(videos, record["files"]):
            assert client.put(base + "/files/" + file["file_id"], content=path.read_bytes()).status_code == 200
        assert client.post(base + "/complete").status_code == 200
        assert client.post(base + "/start").status_code == 202
        status = client.get(base).json()
        assert status["status"] == "ready", status
        root = "/api/v1/library/" + upload
        clips = client.get(root + "/clips").json()["clips"]
        assert len(clips) == 2 and len({r["source_id"] for r in clips}) == 2
        assert all(r["end_sec"] <= 3.01 and r["privacy"] == "privacy-v2.1" for r in clips)
        orders = {}
        for mode in ("eg2_only", "shadow", "active"):
            response = client.post(root + "/search", json={"query": "colored moving shapes", "mode": mode})
            assert response.status_code == 200, response.text
            orders[mode] = [r["row"] for r in response.json()["results"]]
        assert orders["eg2_only"] == orders["shadow"]
        grant = client.post(root + "/previews/0/url").json()
        assert client.get(grant["url"]).status_code == 200
        assert client.get(grant["poster_url"]).status_code == 200
        assert client.post(root + "/export", json={"rows": [0, 1], "format": "csv"}).status_code == 200
        state = app.state.library.worker.request("state")
        assert not state["training"] and not state["any_parameter_requires_grad"] and not state["any_gradient_tensor"]
        published = app.state.library.uploads.path(upload, "_product/published")
        (published / "frame_vectors.npy").unlink()
        response = client.post(root + "/search", json={"query": "colored moving shapes", "mode": "active"})
        assert response.status_code == 200 and response.json()["decision"]["fallback"]
        assert [r["row"] for r in response.json()["results"]] == orders["eg2_only"]
        assert client.delete(base).status_code == 200
        assert client.get(grant["url"]).status_code == 404
    receipt = {
        "passed": True,
        "hardware": state["GPU"],
        "dtype": state["dtype"],
        "revision": state["revision"],
        "sources": 2,
        "clips": 2,
        "native_models": True,
        "training": False,
        "private_media_and_poster": True,
        "exact_eg2_fallback": True,
        "delete_revocation": True,
        "seconds": time.perf_counter() - started,
        "media": "synthetic generated shapes; no datasets or reserves",
    }
    (destination / "acceptance.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
