import hashlib
import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from demo.backend import DemoService, create_app


def write_index(path, *, split="train", enabled=False, corrupt=False):
    manifest = json.dumps({"clips": [{"row": 0, "split": split, "start_sec": 0, "end_sec": 5}]}).encode()
    (path / "manifest.json").write_bytes(manifest)
    (path / "index_meta.json").write_text(
        json.dumps(
            {
                "format": "driveevidence-v6",
                "num_clips": 1,
                "policy": {"enabled": enabled, "state": "EG2_ONLY"},
                "hashes": {"manifest.json": "bad" if corrupt else hashlib.sha256(manifest).hexdigest()},
            }
        )
    )


@pytest.mark.parametrize("options", [{"split": "final"}, {"split": "shadow"}, {"enabled": True}, {"corrupt": True}])
def test_guard_rejects_heldout_or_changed_policy_before_model_loading(tmp_path, options):
    write_index(tmp_path, **options)
    service = object.__new__(DemoService)
    with pytest.raises(HTTPException) as exc:
        service.train_snapshot(tmp_path)
    assert exc.value.status_code == 409


def test_workspace_exposes_library_but_keeps_corpus_and_engine_protected(tmp_path):
    app = create_app(library_root=tmp_path / "library", warmup=False)
    paths = set(app.openapi()["paths"])
    assert not any("jobs" in path or path.endswith("/clip") for path in paths)
    assert "/api/v1/hybrid/indexes/{index_id}/media" not in paths
    assert "/api/v1/library/uploads" in paths
    assert "/api/v1/library/uploads/{upload_id}/start" in paths


def test_canonical_product_does_not_require_research_corpus_or_expose_raw_media(tmp_path):
    # Reviewed migration: arbitrary private uploads are canonical. Scientific
    # TRAIN loaders retain their separate heldout guard above.
    with TestClient(create_app(library_root=tmp_path / "library", warmup=False)) as client:
        assert client.get("/api/v1/hybrid/indexes").json() == {"indexes": []}
        assert client.get("/health").json()["ready"] is False
        assert client.get("/ready").status_code == 503
        assert client.get("/api/v1/library").json() == {"uploads": []}
        assert client.get("/api/v1/hybrid/indexes/corpus/media").status_code == 404
