"""Loopback-only HTTP client. Video URLs must come from the authorized grant API."""

from __future__ import annotations

import os
from urllib.parse import quote, urlparse

import httpx


class DemoError(Exception):
    pass


ERRORS = {
    403: "Bạn không có quyền truy cập kho video này.",
    404: "Không tìm thấy dữ liệu hoặc bản xem trước.",
    409: "Dữ liệu chưa sẵn sàng hoặc đang được xử lý. Vui lòng làm mới trạng thái.",
    413: "Video vượt quá dung lượng cho phép.",
    422: "Dữ liệu không hợp lệ. Kiểm tra định dạng video hoặc nội dung tìm kiếm.",
    429: "Hệ thống đang xử lý video hoặc tìm kiếm. Vui lòng thử lại sau.",
    503: "Dịch vụ xử lý chưa sẵn sàng. Vui lòng thử lại sau.",
}

DETAIL_ERRORS = {
    "INVALID_VIDEO": "Không đọc được video. Hãy chọn một tệp video hợp lệ.",
    "UNSUPPORTED_VIDEO_FORMAT": "Định dạng video chưa được hỗ trợ.",
    "DUPLICATE_RELATIVE_PATH": "Các video có tên trùng nhau. Đổi tên tệp trước khi tải lên.",
    "UPLOAD_INCOMPLETE": "Chưa tải xong tất cả video. Hãy thử tải lên lại.",
    "DATASET_BUSY": "Bộ video đang được xử lý. Bạn có thể xóa sau khi xử lý xong.",
    "VIDEO_ENCODER_UNAVAILABLE": "Dịch vụ chuyển đổi video chưa sẵn sàng.",
    "UPLOAD_TOO_LARGE": "Video vượt quá dung lượng cho phép.",
}


def backend_url(value=None):
    value = (value or os.getenv("SCENESEARCH_DEMO_API", "http://127.0.0.1:8090")).rstrip("/")
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or parsed.path
        or parsed.username
        or parsed.query
        or parsed.fragment
    ):
        raise DemoError("Địa chỉ dịch vụ phải là HTTP local (127.0.0.1 hoặc localhost).")
    return value


class DemoClient:
    def __init__(self, base=None):
        self.base = backend_url(base)

    def request(self, method, path, *, body=None, content=None, headers=None, timeout=30):
        try:
            response = httpx.request(
                method, self.base + path, json=body, content=content, headers=headers, timeout=timeout, trust_env=False
            )
            if response.status_code >= 400:
                detail = response.json().get("detail")
                code = detail.get("code") if isinstance(detail, dict) else detail if isinstance(detail, str) else None
                raise DemoError(
                    DETAIL_ERRORS.get(code, ERRORS.get(response.status_code, "Có lỗi xảy ra. Vui lòng thử lại."))
                )
            return response.json()
        except httpx.TimeoutException:
            raise DemoError(
                "Hệ thống chưa phản hồi. Lần khởi động đầu có thể mất vài phút; hãy chờ rồi thử lại."
            ) from None
        except (httpx.HTTPError, ValueError):
            raise DemoError("Không kết nối được dịch vụ Scene Search. Vui lòng thử kết nối lại.") from None

    def search(self, index_id, query, top_k, *, uploaded=False, mode="eg2_only", filters=None):
        return self.request(
            "POST",
            f"/api/v1/library/{quote(index_id, safe='')}/search"
            if uploaded
            else f"/api/v1/hybrid/indexes/{quote(index_id, safe='')}/search",
            body={"query": query, "top_k": top_k, "mode": mode, "filters": filters or {}},
            timeout=600,
        )

    def preview(self, index_id, row, *, uploaded=False):
        root = (
            f"/api/v1/library/{quote(index_id, safe='')}"
            if uploaded
            else f"/api/v1/hybrid/indexes/{quote(index_id, safe='')}"
        )
        prefix = f"{root}/previews/{int(row)}"
        grant = self.request("POST", prefix + "/url")
        if grant.get("privacy") not in ("privacy-v2.1", "scene-search-privacy-v2.1"):
            raise DemoError("Bản xem trước riêng tư chưa sẵn sàng. Hãy xử lý video rồi thử lại.")
        origin = urlparse(self.base)
        for field, asset in (("url", "media"), ("poster_url", "poster")):
            parsed = urlparse(grant.get(field, ""))
            if (
                (parsed.scheme, parsed.netloc) != (origin.scheme, origin.netloc)
                or parsed.path != prefix + "/" + asset
                or not parsed.query.startswith("token=")
                or parsed.fragment
            ):
                raise DemoError("Không xác thực được đường dẫn video. Vui lòng tải lại.")
        return grant

    def upload(self, files, name, key, progress=None):
        record = self.request(
            "POST",
            "/api/v1/library/uploads",
            body={"folder_name": name, "files": [{"relative_path": f.name, "size_bytes": f.size} for f in files]},
            headers={"Idempotency-Key": key},
        )
        if record["status"] == "receiving":
            for i, (file, target) in enumerate(zip(files, record["files"], strict=True)):
                if target["status"] != "received":
                    file.seek(0)
                    self.request(
                        "PUT",
                        f"/api/v1/library/uploads/{record['upload_id']}/files/{target['file_id']}",
                        content=file,
                        headers={"Content-Type": "application/octet-stream"},
                        timeout=600,
                    )
                if progress:
                    progress((i + 1) / len(files))
            record = self.request("POST", f"/api/v1/library/uploads/{record['upload_id']}/complete", timeout=120)
        return record

    def video_bytes(self, index_id, row, *, uploaded=False):
        grant = self.preview(index_id, row, uploaded=uploaded)
        try:
            response = httpx.get(grant["url"], timeout=120, trust_env=False)
            response.raise_for_status()
            return response.content
        except httpx.HTTPError:
            raise DemoError("Không tải được đoạn video. Vui lòng thử lại.") from None
