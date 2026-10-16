"""Canonical product contracts with real media I/O and isolated model fixtures."""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from demo.backend import create_app
from demo.product_index import read
from tests.test_demo.test_library import client as client
from tests.test_demo.test_library import library as library
from tests.test_demo.test_library import ready, uploaded
from tests.test_demo.test_library import video as video


def test_worker_close_waits_for_owned_interpreter_child(tmp_path):
    import subprocess
    import sys

    import numpy as np
    import psutil

    from demo.product_worker import ProductWorker
    from hybrid_v2.embeddinggemma2_index import FROZEN_CONFIG

    frames = tmp_path / "frames.npy"
    np.save(frames, np.zeros((8, 2, 2, 3), dtype=np.uint8))
    child_code = (
        "import numpy as np,time; a=np.load("
        + repr(str(frames))
        + ",mmap_mode='r'); print('ready',flush=True); time.sleep(120)"
    )
    parent_code = (
        "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c',"
        + repr(child_code)
        + "]); print(p.pid,flush=True); time.sleep(120)"
    )
    worker = ProductWorker(json.loads(FROZEN_CONFIG.read_text(encoding="utf-8"))["config"])
    worker.process = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    )
    child = psutil.Process(int(worker.process.stdout.readline()))
    try:
        assert worker.process.stdout.readline().strip() == "ready"
        worker.close()
        assert not child.is_running()
        frames.unlink()  # Windows must release the mmap before this returns.
    finally:
        worker.close()
        if child.is_running():
            child.kill()


@pytest.mark.parametrize("asset", ["receipt.json", "tracks/privacy_meta.json", "safe.mp4", "poster.jpg"])
def test_privacy_corruption_revokes_existing_media_and_search(client, library, video, asset):
    uid, _ = ready(client, video)
    root = f"/api/v1/library/{uid}"
    grant = client.post(root + "/previews/0/url").json()
    library.uploads.path(uid, "_product/published/media/0/" + asset).unlink()
    assert client.get(grant["url"]).status_code == 409
    assert client.get(grant["poster_url"]).status_code == 409
    assert client.post(root + "/search", json={"query": "cars"}).status_code == 409


def test_missing_local_evidence_degrades_to_exact_eg2_and_reports_readiness(client, library, video):
    uid, _ = ready(client, video)
    root = f"/api/v1/library/{uid}"
    baseline = client.post(root + "/search", json={"query": "car"}).json()
    library.uploads.path(uid, "_product/published/frame_vectors.npy").unlink()
    for mode in ("shadow", "active"):
        response = client.post(root + "/search", json={"query": "car", "mode": mode})
        assert response.status_code == 200
        payload = response.json()
        assert payload["decision"]["fallback"] is True
        assert payload["decision"]["user_ranking"] == "EG2"
        assert [r["row"] for r in payload["results"]] == [r["row"] for r in baseline["results"]]
    state = client.get(root + "/readiness").json()
    assert state["service_state"] == "DEGRADED_EG2_ONLY" and state["privacy_ready"] is True


@pytest.mark.parametrize("initial,after", [("auto", "cpu"), ("cpu", "auto")])
def test_device_change_preserves_private_collection_access(client, library, video, monkeypatch, initial, after):
    from demo import product_privacy

    monkeypatch.setenv("SCENESEARCH_DEVICE", initial)
    scan = product_privacy.scan_video_privacy
    devices = []

    def observed_scan(raw, folder, cfg):
        devices.append(cfg.plate_device)
        assert cfg.fp16 is False
        return scan(raw, folder, cfg)

    monkeypatch.setattr(product_privacy, "scan_video_privacy", observed_scan)
    uid, _ = ready(client, video)
    assert devices and all(d == ("cpu" if initial == "cpu" else None) for d in devices)
    root = f"/api/v1/library/{uid}"
    grant = client.post(root + "/previews/0/url").json()
    monkeypatch.setenv("SCENESEARCH_DEVICE", after)
    assert client.get(grant["url"]).status_code == 200
    assert client.get(grant["poster_url"]).status_code == 200
    assert client.post(root + "/search", json={"query": "car"}).status_code == 200


def test_atomic_publication_failure_is_private_and_retry_reuses_verified_vectors(client, library, video, monkeypatch):
    import demo.product_index as module

    uid, base = uploaded(client, video)
    rename = module.os.rename

    def interrupted(source, target):
        if Path(target).name == "published":
            raise OSError("simulated crash before atomic rename")
        return rename(source, target)

    monkeypatch.setattr(module.os, "rename", interrupted)
    client.post(base + "/start")
    assert client.get(base).json()["status"] == "failed"
    assert client.get(f"/api/v1/library/{uid}/clips").status_code == 409
    assert not library.uploads.path(uid, "_product/published").exists()
    receipt = library.uploads.path(uid, "_product/cache/000000/embedding.json")
    before = receipt.read_bytes(), receipt.stat().st_mtime_ns
    monkeypatch.setattr(module.os, "rename", rename)
    assert client.post(base + "/retry").status_code == 202
    assert client.get(base).json()["status"] == "ready"
    assert (receipt.read_bytes(), receipt.stat().st_mtime_ns) == before


def test_ready_publication_is_immutable_and_unique_identity(client, library, video):
    uid, base = ready(client, video)
    metadata = library.uploads.path(uid, "_product/published/index_meta.json")
    before = metadata.read_bytes(), metadata.stat().st_mtime_ns
    client.post(base + "/start")
    client.post(base + "/retry")
    assert (metadata.read_bytes(), metadata.stat().st_mtime_ns) == before
    meta = read(metadata)
    assert meta["publication_identity"] != meta["build_identity"]


def test_filters_apply_to_corpus_before_v6_and_exports_retain_scores(client, library, video):
    uid, _ = ready(client, video)
    root = f"/api/v1/library/{uid}"
    request = {"query": "car", "mode": "active", "filters": {"start_sec": 5, "end_sec": 8}}
    payload = client.post(root + "/search", json=request).json()
    assert [r["row"] for r in payload["results"]] == [1]
    assert payload["retrieval"]["filtered_corpus_size"] == 1
    assert (
        client.post(root + "/search", json={"query": "car", "filters": {"source_ids": ["absent"]}}).json()["results"]
        == []
    )
    export = client.post(root + "/export", json={"rows": [1], "search": request}).json()["clips"][0]
    assert export["rank"] == 1 and export["mode"] == "active" and export["global_score"] == 0
    assert client.post(root + "/export", json={"rows": [0], "search": request}).status_code == 422


def test_literal_query_is_excluded_from_telemetry(client, library, video):
    uid, _ = ready(client, video)
    query = "private literal query never in telemetry"
    client.post(f"/api/v1/library/{uid}/search", json={"query": query, "mode": "shadow"})
    telemetry = (library.root / "telemetry/search.jsonl").read_text()
    assert query not in telemetry
    event = json.loads(telemetry.splitlines()[-1])
    assert len(event["query_hash"]) == 64 and event["deployment_mode"] == "shadow"


@pytest.mark.parametrize("headers", [{"Origin": "https://remote.example"}, {"Host": "remote.example"}])
def test_local_boundary_rejects_remote_origins_and_hosts(tmp_path, headers):
    with TestClient(create_app(library_root=tmp_path, warmup=False)) as client:
        assert client.get("/health", headers=headers).status_code == 403


def test_invalid_mode_filter_and_optional_metadata_are_rejected_before_inference(client):
    root = "/api/v1/library/unknown/search"
    for extra in ({"mode": "promoted"}, {"filters": {"color": "red"}}, {"filters": {"start_sec": 5, "end_sec": 1}}):
        assert client.post(root, json={"query": "cars", **extra}).status_code == 422


def test_insufficient_working_cache_storage_fails_before_any_encoder(client, library, video, monkeypatch):
    from types import SimpleNamespace

    import demo.product_index as module

    monkeypatch.setattr(module.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    uid, base = uploaded(client, video)
    client.post(base + "/start")
    record = client.get(base).json()
    assert record["status"] == "failed" and record["error"] == "INSUFFICIENT_PRODUCT_STORAGE"
    assert library.worker is None
    assert client.get(f"/api/v1/library/{uid}/clips").status_code == 409
