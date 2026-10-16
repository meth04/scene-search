import pytest

from demo.client import DemoClient, DemoError, backend_url


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com",
        "http://127.0.0.1:8090/raw",
        "http://user@localhost:8090",
        "http://localhost:8090?token=secret",
    ],
)
def test_external_or_ambiguous_backend_url_is_rejected(monkeypatch, url):
    monkeypatch.setenv("SCENESEARCH_DEMO_API", url)
    with pytest.raises(DemoError):
        backend_url()


@pytest.mark.parametrize(
    "media",
    [
        "http://127.0.0.1:8090/api/v1/hybrid/indexes/demo/media?token=x",
        "http://other:8090/api/v1/hybrid/indexes/demo/previews/2/media?token=x",
        "http://127.0.0.1:8090/api/v1/hybrid/indexes/demo/previews/3/media?token=x",
    ],
)
def test_preview_cannot_use_raw_media_or_wrong_source(monkeypatch, media):
    client = DemoClient("http://127.0.0.1:8090")
    monkeypatch.setattr(
        client,
        "request",
        lambda *a, **kw: {
            "url": media,
            "poster_url": "http://127.0.0.1:8090/api/v1/hybrid/indexes/demo/previews/2/poster?token=x",
        },
    )
    with pytest.raises(DemoError):
        client.preview("demo", 2)
