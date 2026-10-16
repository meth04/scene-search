"""Installation integrity, resumable downloads and portable execution contracts."""

import hashlib
import io
import zipfile
from types import SimpleNamespace

import pytest

from demo.product_assets import runtime_model_assets
from demo.product_runtime import execution_device
from scripts.install_product_assets import destination, download, import_bundle


@pytest.mark.parametrize("path", ["../outside", "/absolute", "C:/Windows/file", "folder\\file"])
def test_download_destination_cannot_escape_install(tmp_path, path):
    with pytest.raises(ValueError):
        destination(tmp_path, path)


def test_resume_does_not_append_when_server_ignores_range(tmp_path):
    payload = b"pinned public weight bytes"
    target = tmp_path / "model"
    target.with_name("model.download").write_bytes(payload[:8])
    record = {
        "path": "model",
        "url": "https://models.invalid/file",
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }

    def opener(request, timeout):
        assert request.headers["Range"] == "bytes=8-"
        response = io.BytesIO(payload)
        response.status, response.headers = 200, {}
        return response

    download(record, target, opener=opener)
    assert target.read_bytes() == payload


def test_wrong_download_hash_never_replaces_existing_model(tmp_path):
    target = tmp_path / "model"
    target.write_bytes(b"original")
    record = {"path": "model", "url": "https://models.invalid/file", "bytes": 4, "sha256": "0" * 64}

    def opener(request, timeout):
        response = io.BytesIO(b"oops")
        response.status, response.headers = 200, {}
        return response

    with pytest.raises(ValueError):
        download(record, target, repair=True, opener=opener)
    assert target.read_bytes() == b"original"


def test_bundle_only_extracts_declared_model_members(tmp_path):
    bundle = tmp_path / "models.zip"
    payload = b"valid weights"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("data/model", payload)
        archive.writestr("../../escape", b"forbidden")
    record = {"path": "data/model", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
    import_bundle(bundle, [record], tmp_path / "install")
    assert (tmp_path / "install/data/model").read_bytes() == payload
    assert not (tmp_path / "escape").exists()


def test_unused_historical_head_is_optional_but_runtime_hashes_remain_exact():
    manifest = {"assets": {"encoder": "required"}, "provenance_assets": {"unused_head": "historical"}}
    assert runtime_model_assets({"encoder": "required", "unused_head": "historical"}, manifest) == {
        "encoder": "required"
    }
    for stored in (
        {},
        {"encoder": "changed"},
        {"encoder": "required", "unused_head": "changed"},
        {"encoder": "required", "unknown": "extra"},
    ):
        with pytest.raises(ValueError):
            runtime_model_assets(stored, manifest)


@pytest.mark.parametrize(
    "requested,available,expected",
    [("auto", False, "cpu"), ("auto", True, "cuda"), ("cpu", True, "cpu"), ("cuda", True, "cuda")],
)
def test_device_selection_does_not_require_a_named_gpu(monkeypatch, requested, available, expected):
    monkeypatch.setenv("SCENESEARCH_DEVICE", requested)
    assert execution_device(SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: available))) == expected
