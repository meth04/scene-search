"""Library lifecycle and access checks; inference is isolated in these unit tests."""

import json
import threading
from types import SimpleNamespace

import av
import numpy as np
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from demo.backend import DemoSearchRequest
from demo.library import VideoLibrary, library_router
from hybrid_v2.embeddinggemma2 import decode_frames, digest
from src.privacy.render import ffmpeg_binary


class FixtureEncoder:
    def __init__(self, config, **kwargs):
        self.dimension = config["dimension"]
        self.count = 0

    def load(self):
        return self

    def encode_video(self, path, config):
        # Keep the real model's input checks even when inference is replaced.
        decode_frames(path, config["frame_count"])
        vector = np.zeros(self.dimension, np.float32)
        vector[self.count % 2] = 1
        self.count += 1
        return vector, {}

    def encode_text(self, query):
        vector = np.zeros(self.dimension, np.float32)
        vector[0] = 1
        return vector, {}

    def encode_evidence(self, path, indices, **kwargs):
        vector = np.zeros(self.dimension, np.float32)
        vector[0] = 1
        return vector, {}

    def close(self):
        pass


@pytest.fixture
def library(tmp_path, monkeypatch):
    def authorize(actor, **kwargs):
        if actor.tenant_id != "default" or "default" not in actor.access_scope_ids or "Engineer" not in actor.roles:
            raise HTTPException(403, "SOURCE_FORBIDDEN")

    service = SimpleNamespace(
        compute=threading.Lock(),
        lock=threading.RLock(),
        authorize=authorize,
        engines=SimpleNamespace(clear=lambda: None),
    )
    value = VideoLibrary(service, tmp_path / "library", worker_factory=FixtureEncoder)
    # Reconciliation: product indexing now requires V6 local evidence and
    # complete privacy-v2.1 receipts. Unit fixtures replace detector inference,
    # while retaining real decode, safe rendering, hashes and publication.
    import sys
    from dataclasses import asdict
    from pathlib import Path

    from demo import product_privacy
    from hybrid_v2.embeddinggemma2 import ROOT
    from src.privacy.tracks import probe, save_tracks, write_json

    sys.path.insert(0, str(ROOT / "scripts"))
    from build_driveevidence_v3 import decode

    def fixture_scan(raw, folder, cfg):
        info = probe(raw)
        return save_tracks(
            folder,
            {"video": info, "covered_frames": info["frame_count"], "complete": True, "config": asdict(cfg)},
            [],
            [],
        )

    monkeypatch.setattr(product_privacy, "scan_video_privacy", fixture_scan)

    def offline(plan_path):
        for item in json.loads(plan_path.read_text())["rows"]:
            folder = Path(item["cache"])
            folder.mkdir(parents=True, exist_ok=True)
            product_privacy.prepare(item["raw"], folder / "privacy", item["duration"])
            geometry = folder / "geometry/000000"
            decode({"source_path": item["inference"], "source_sha256": digest(item["inference"])}, geometry)
            write_json(geometry / "objects.json", {"objects": []})
            # Stable identity: fixtures produce equivalent empty tracks but new
            # cache filenames; key the receipt by the exact decoded input.
            write_json(
                folder / "offline_receipt.json",
                {"identity": item["identity"], "decoded_sha256": digest(geometry / "frames.npy")},
            )

    value.offline = offline
    monkeypatch.setattr(VideoLibrary, "offline", lambda self, path: offline(path))
    monkeypatch.setattr(
        VideoLibrary,
        "asset_identity",
        lambda self: json.loads((ROOT / "configs/product/assets.json").read_text())["assets"],
    )
    yield value
    value.close()


@pytest.fixture
def client(library):
    app = FastAPI()
    app.include_router(library_router(library, DemoSearchRequest), prefix="/api/v1")
    with TestClient(app) as client:
        yield client


def make_video(tmp_path, frames=80, rate=10):
    path = tmp_path / "drive.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width, stream.height, stream.pix_fmt = 64, 48, "yuv420p"
        for i in range(frames):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 3) % 256, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path.read_bytes()


@pytest.fixture
def video(tmp_path):
    return make_video(tmp_path)


def uploaded(client, video):
    body = {"folder_name": "Camera hành trình", "files": [{"relative_path": "drive.mp4", "size_bytes": len(video)}]}
    response = client.post("/api/v1/library/uploads", json=body, headers={"Idempotency-Key": "video-1"})
    assert response.status_code == 201
    record = response.json()
    base = f"/api/v1/library/uploads/{record['upload_id']}"
    assert client.put(base + f"/files/{record['files'][0]['file_id']}", content=video).status_code == 200
    response = client.post(base + "/complete")
    assert response.status_code == 200
    assert response.json()["status"] == "uploaded"
    return record["upload_id"], base


def ready(client, video):
    if not ffmpeg_binary():
        pytest.skip("FFmpeg is required for actual clip conversion")
    uid, base = uploaded(client, video)
    assert client.post(base + "/start").status_code == 202
    record = client.get(base).json()
    assert record["status"] == "ready", record
    assert record["clips"] == 2
    return uid, base


def test_receive_is_separate_from_processing_and_survives_restart(client, library, video):
    uid, base = uploaded(client, video)
    assert library.worker is None
    reopened = VideoLibrary(library.service, library.root, worker_factory=FixtureEncoder)
    assert (
        reopened.catalog(
            SimpleNamespace(
                actor_id="local-user", tenant_id="default", roles=["Engineer"], access_scope_ids=["default"]
            )
        )[0]["status"]
        == "uploaded"
    )
    assert client.get(base).json()["upload_id"] == uid


def test_full_lifecycle_trims_last_clip_and_revokes_deleted_video(client, video):
    uid, base = ready(client, video)
    prefix = f"/api/v1/library/{uid}"
    clips = client.get(prefix + "/clips").json()["clips"]
    assert [(r["start_sec"], r["end_sec"]) for r in clips] == [(0.0, 5.0), (5.0, 8.0)]
    hits = client.post(prefix + "/search", json={"query": "car", "top_k": 5}).json()["results"]
    assert [hit["row"] for hit in hits] == [0, 1]
    assert [hit["score"] for hit in hits] == [1.0, 0.0]
    grant = client.post(prefix + "/previews/1/url").json()
    assert grant["privacy"] == "privacy-v2.1" and grant["start_sec"] == 0 and grant["end_sec"] == 3
    response = client.get(grant["url"], headers={"Range": "bytes=0-100"})
    assert response.status_code == 206 and len(response.content) == 101
    assert client.get(grant["poster_url"]).status_code == 200
    assert client.get(grant["url"] + "tampered").status_code == 403
    assert client.get(grant["url"].replace("/1/media", "/0/media")).status_code == 403
    assert client.delete(base).status_code == 200
    assert client.get(grant["url"]).status_code == 404
    assert client.get("/api/v1/library").json()["uploads"] == []


@pytest.mark.parametrize("frames,rate", [(1, 10), (7, 10), (507, 10), (2, 1), (5, 1)])
def test_short_inputs_and_subsecond_tail_keep_original_playback(client, library, tmp_path, frames, rate):
    if not ffmpeg_binary():
        pytest.skip("FFmpeg is required for actual clip conversion")
    video = make_video(tmp_path, frames=frames, rate=rate)
    uid, base = uploaded(client, video)
    source = library.uploads.path(uid, "drive.mp4")
    source_hash = digest(source)
    assert client.post(base + "/start").status_code == 202
    record = client.get(base).json()
    assert record["status"] == "ready", record
    total = (frames + 5 * rate - 1) // (5 * rate)
    assert record["completed_scenes"] == record["total_scenes"] == total
    prefix = f"/api/v1/library/{uid}"
    clips = client.get(prefix + "/clips").json()["clips"]
    assert len(clips) == total
    assert clips[-1]["end_sec"] == pytest.approx(frames / rate)
    grant = client.post(prefix + f"/previews/{total - 1}/url").json()
    tail_duration = frames / rate - 5 * (total - 1)
    assert grant["end_sec"] == pytest.approx(tail_duration)
    with av.open(str(library.uploads.path(uid, f"_product/published/media/{total - 1}/safe.mp4"))) as container:
        assert container.duration / av.time_base == pytest.approx(tail_duration, abs=0.05)
    assert digest(source) == source_hash
    assert not list(library.uploads.path(uid, "_product").glob(".*.embedding.mp4"))
    assert client.post(prefix + "/search", json={"query": "car"}).status_code == 200


@pytest.mark.parametrize("changed", [None, "embedding", "configuration"])
def test_retry_after_restart_reuses_only_verified_completed_scenes(client, library, video, changed):
    if not ffmpeg_binary():
        pytest.skip("FFmpeg is required for actual clip conversion")
    calls = []

    class InterruptedEncoder(FixtureEncoder):
        def encode_video(self, path, config):
            calls.append(path.name)
            if len(calls) == 2:
                raise RuntimeError("PROCESSING_INTERRUPTED")
            return super().encode_video(path, config)

    library.worker_factory = InterruptedEncoder
    uid, base = uploaded(client, video)
    assert client.post(base + "/start").status_code == 202
    failed = client.get(base).json()
    assert failed["status"] == "failed" and failed["completed_scenes"] == 1
    assert failed["total_scenes"] == 2 and failed["progress"] == pytest.approx(0.65)
    folder = library.uploads.path(uid, "_product/cache/000000")
    checkpoint_time = (folder / "embedding.json").stat().st_mtime_ns
    if changed == "embedding":
        (folder / "global.npy").write_bytes(b"changed")
    elif changed == "configuration":
        checkpoint = json.loads((folder / "embedding.json").read_text(encoding="utf-8"))
        checkpoint["identity"] = "changed"
        (folder / "embedding.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    reopened = VideoLibrary(library.service, library.root, worker_factory=InterruptedEncoder)
    app = FastAPI()
    app.include_router(library_router(reopened, DemoSearchRequest), prefix="/api/v1")
    try:
        with TestClient(app) as next_client:
            assert next_client.post(base + "/start").status_code == 202
            record = next_client.get(base).json()
            assert record["status"] == "ready", record
            assert record["completed_scenes"] == 2
            assert next_client.post(f"/api/v1/library/{uid}/search", json={"query": "car"}).status_code == 200
        assert len(calls) == (3 if changed is None else 4)
        if changed is None:
            assert (folder / "embedding.json").stat().st_mtime_ns == checkpoint_time
        assert not list(folder.glob(".*.embedding.mp4"))
    finally:
        reopened.close()


@pytest.mark.parametrize(
    "headers",
    [{"X-Actor-Id": "other"}, {"X-Tenant-Id": "other"}, {"X-Access-Scopes": "other"}, {"X-Roles": "Reviewer"}],
)
def test_other_identity_cannot_access_or_delete_upload(client, video, headers):
    uid, base = uploaded(client, video)
    assert client.get(base, headers=headers).status_code == 403
    assert client.delete(base, headers=headers).status_code == 403
    assert client.post(base + "/start", headers=headers).status_code == 403
    assert client.get(base).status_code == 200


@pytest.mark.parametrize("path", ["../escape.mp4", "a/../escape.mp4", "_scenes/0.mp4", "video.txt", "C:/video.mp4"])
def test_invalid_names_never_create_upload(client, path):
    response = client.post(
        "/api/v1/library/uploads", json={"folder_name": "Videos", "files": [{"relative_path": path, "size_bytes": 10}]}
    )
    assert response.status_code == 422
    assert client.get("/api/v1/library").json()["uploads"] == []


def test_corrupt_video_does_not_complete_or_start(client):
    record = client.post(
        "/api/v1/library/uploads",
        json={"folder_name": "Videos", "files": [{"relative_path": "bad.mp4", "size_bytes": 4}]},
    ).json()
    base = f"/api/v1/library/uploads/{record['upload_id']}"
    assert client.put(base + f"/files/{record['files'][0]['file_id']}", content=b"oops").status_code == 422
    assert client.post(base + "/complete").status_code == 409
    assert client.post(base + "/start").status_code == 409


def test_compute_busy_does_not_start_or_lose_uploaded_files(client, library, video):
    _, base = uploaded(client, video)
    library.service.compute.acquire()
    try:
        assert client.post(base + "/start").status_code == 429
        assert client.get(base).json()["status"] == "uploaded"
        with pytest.raises(HTTPException) as error:
            library.prepare_builtin()
        assert error.value.status_code == 429
    finally:
        library.service.compute.release()


@pytest.mark.parametrize("asset", ["source", "manifest", "matrix", "preview"])
def test_changed_library_artifacts_are_blocked(client, library, video, asset):
    uid, base = ready(client, video)
    filename = {
        "source": "drive.mp4",
        "manifest": "_product/published/manifest.json",
        "matrix": "_product/published/global_vectors.npy",
        "preview": "_product/published/media/0/safe.mp4",
    }[asset]
    library.uploads.path(uid, filename).write_bytes(b"changed")
    endpoint = f"/api/v1/library/{uid}"
    if asset == "preview":
        response = client.post(endpoint + "/previews/0/url")
    else:
        response = client.post(endpoint + "/search", json={"query": "car"})
    assert response.status_code == 409


def test_interrupted_processing_becomes_retryable(library, client, video):
    uid, base = uploaded(client, video)
    library.update(uid, status="indexing")
    reopened = VideoLibrary(library.service, library.root, worker_factory=FixtureEncoder)
    record = reopened.uploads.repository.get("demo_upload", uid)
    assert record["status"] == "failed" and record["error"] == "PROCESSING_INTERRUPTED"
    assert client.get(base).json()["files"][0]["status"] == "received"


@pytest.mark.parametrize("query", ["", "   "])
def test_empty_search_is_rejected_without_inference(client, query):
    response = client.post("/api/v1/library/unknown/search", json={"query": query})
    assert response.status_code == 422
