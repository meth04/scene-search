"""Scene Search: search and explore a personal video library over HTTP."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import time
from html import escape
from pathlib import Path, PurePosixPath
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from demo.client import DemoClient, DemoError, backend_url

st.set_page_config(page_title="Scene Search | Thư viện video thông minh", page_icon="🔎", layout="wide")
st.markdown(
    """
<style>
.stApp { background: #f5f7fb; color: #172b46; }
.block-container { max-width: 1380px; padding-top: 2.6rem; padding-bottom: 3rem; }
h1, h2, h3 { color: #142b45; letter-spacing: -.035em; }
h1 { font-size: 2.5rem !important; font-weight: 800 !important; }
h3 { font-size: 1.15rem !important; }
[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e3e9f1; }
[data-testid="stMetric"] { background: #fff; border: 1px solid #e3e9f1; border-radius: 14px; padding: 16px 20px; }
[data-testid="stMetricLabel"] { color: #6b7c91; }
[data-testid="stMetricValue"] { font-size: 1.8rem; color: #172b46; }
[data-testid="stVerticalBlockBorderWrapper"] > div { border-color: #e3e9f1 !important; border-radius: 14px; }
[data-testid="stForm"] { background: #fff; border: 1px solid #e3e9f1; border-radius: 14px; padding: 20px; }
.brand { display: flex; align-items: center; gap: 10px; font-size: 1.2rem; font-weight: 800; color: #142b45; }
.brand-icon { background: #0c827e; color: white; padding: 8px 12px; border-radius: 12px; }
.eyebrow { color: #087e7a; font-size: .72rem; font-weight: 700; letter-spacing: .16em; margin-bottom: .35rem; }
.status-pill { display: inline-block; background: #e8f6f0; color: #177456; border: 1px solid #c9e9da;
  border-radius: 20px; padding: 5px 11px; font-size: .73rem; font-weight: 600; }
.hero { background: linear-gradient(115deg, #132b46, #194c59); border-radius: 18px; padding: 24px 28px;
  color: #fff; margin: 12px 0 24px; position: relative; overflow: hidden; }
.hero::after { content: ''; position: absolute; right: -50px; top: -95px; width: 270px; height: 270px;
  border: 38px solid #ffffff08; border-radius: 50%; pointer-events: none; }
.hero strong { display: block; font-size: 1.15rem; margin-bottom: 7px; }
.hero p { color: #c8dbe2; font-size: .86rem; margin: 0; max-width: 730px; }
.empty { border: 1px dashed #cad5e3; border-radius: 16px; padding: 38px 24px; text-align: center;
  background: #ffffff80; margin-top: 18px; color: #65778d; }
.empty strong { display: block; font-size: 1.05rem; color: #203b55; margin-bottom: 8px; }
.scene-title { font-size: .91rem; font-weight: 650; color: #213a55; overflow-wrap: anywhere; }
.scene-time { font-size: .77rem; color: #728399; margin-top: 7px; }
.score { display: inline-block; font-size: .74rem; font-weight: 700; color: #087d78; background: #eaf6f4;
  border-radius: 7px; padding: 4px 9px; }
button { border-radius: 9px !important; }
footer { visibility: hidden; }
@media(max-width: 640px) {
 .block-container { padding: 2rem 1rem; } h1 { font-size: 2rem !important; }
 .hero { padding: 20px; } [data-testid="stMetric"] { padding: 12px; }
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=5, show_spinner=False)
def workspace(base):
    client = DemoClient(base)
    health = client.request("GET", "/health")
    catalog = client.request("GET", "/api/v1/hybrid/indexes")["indexes"]
    uploads = client.request("GET", "/api/v1/library")["uploads"]
    config = client.request("GET", "/api/v1/library/config")
    indexes = [{**item, "label": "Kho giao thông", "uploaded": False} for item in catalog]
    indexes += [
        {"index_id": item["upload_id"], "label": item["folder_name"], "uploaded": True, "clips": item["clips"]}
        for item in uploads
        if item["status"] == "ready"
    ]
    return health, indexes, uploads, config


def timestamp(seconds):
    minutes, seconds = divmod(max(0, float(seconds)), 60)
    return f"{int(minutes):02d}:{seconds:04.1f}"


def clip_name(hit):
    return PurePosixPath(
        (hit.get("metadata") or {}).get("filename", hit.get("filename", hit.get("clip_id", "Video")))
    ).name


def source_bounds(hit):
    metadata = hit.get("metadata") or hit
    return metadata.get("source_start_sec", hit["start_sec"]), metadata.get("source_end_sec", hit["end_sec"])


def interval(hit):
    start, end = source_bounds(hit)
    return f"{timestamp(start)} – {timestamp(end)}"


def navigate(page):
    st.session_state.page = page


def select_collection(index_id):
    st.session_state.update(collection=index_id, payload=None, selected_row=None, page="Tìm kiếm")


def empty(title, description):
    st.markdown(
        f'<div class="empty"><strong>{escape(title)}</strong>{escape(description)}</div>', unsafe_allow_html=True
    )


def save_result(hit, collection):
    key = f"{collection['index_id']}:{hit['row']}"
    if key in st.session_state.saved:
        del st.session_state.saved[key]
    else:
        st.session_state.saved[key] = {
            "hit": hit,
            "collection": collection,
            "query": st.session_state.get("searched_query", ""),
        }


def csv_results(hits):
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "rank",
            "clip_id",
            "source_id",
            "filename",
            "start_sec",
            "end_sec",
            "global_score",
            "final_score",
            "mode",
            "privacy",
        ]
    )
    for rank, hit in enumerate(hits, 1):
        filename = clip_name(hit)
        if filename.lstrip().startswith(("=", "+", "-", "@")):
            filename = "'" + filename
        start, end = source_bounds(hit)
        writer.writerow(
            [
                hit.get("rank", rank),
                hit["clip_id"],
                (hit.get("metadata") or {}).get("source_id", ""),
                filename,
                start,
                end,
                hit["score"],
                hit.get("final_score", hit["score"]),
                hit.get("mode", "eg2_only"),
                "privacy-v2.1",
            ]
        )
    return "\ufeff" + output.getvalue()


def result_card(hit, rank, collection):
    with st.container(border=True):
        text_col, score_col = st.columns([4, 1])
        text_col.markdown(
            f'<div class="scene-title">{rank:02d} &nbsp; {escape(clip_name(hit))}</div>'
            f'<div class="scene-time">{interval(hit)}</div>',
            unsafe_allow_html=True,
        )
        score_col.markdown(f'<span class="score">{hit["score"]:.3f}</span>', unsafe_allow_html=True)
        view_col, save_col = st.columns([2, 1])
        view_col.button(
            "Xem đoạn video",
            key=f"view_{hit['row']}",
            width="stretch",
            on_click=lambda row=hit["row"]: st.session_state.update(selected_row=row),
        )
        is_saved = f"{collection['index_id']}:{hit['row']}" in st.session_state.saved
        save_col.button(
            "Đã lưu" if is_saved else "Lưu",
            key=f"save_{hit['row']}",
            width="stretch",
            icon=":material/bookmark:" if is_saved else ":material/bookmark_border:",
            on_click=save_result,
            args=(hit, collection),
        )


def player(hit, collection, *, key="result"):
    index_id, row = collection["index_id"], hit["row"]
    st.markdown(f"### {escape(clip_name(hit))}")
    st.caption(f"Vị trí trong video · {interval(hit)}")
    try:
        grant = client.preview(index_id, row, uploaded=collection["uploaded"])
        # Each preview is already trimmed. Streamlit truncates end_time to whole
        # seconds, which rejects a subsecond clip with start=0 and end=0.
        st.video(grant["url"], start_time=float(grant["start_sec"]))
        st.caption("Bản xem trước đã che mặt và biển số bằng privacy-v2.1. Hãy kiểm tra trước khi chia sẻ.")
        st.caption(f"Đoạn phát · {timestamp(grant['start_sec'])} – {timestamp(grant['end_sec'])}")
        refresh_col, download_col = st.columns(2)
        if refresh_col.button("Tải lại video", key=f"refresh_{key}", width="stretch", icon=":material/refresh:"):
            st.rerun()
        download_key = f"{index_id}:{row}"
        if download_col.button("Tải đoạn video", key=f"prepare_{key}", width="stretch", icon=":material/download:"):
            with st.spinner("Đang chuẩn bị tệp…"):
                st.session_state.download = {
                    "key": download_key,
                    "data": client.video_bytes(index_id, row, uploaded=collection["uploaded"]),
                }
        download = st.session_state.get("download", {})
        if download.get("key") == download_key:
            st.download_button(
                "Lưu tệp MP4",
                download["data"],
                file_name=f"{PurePosixPath(clip_name(hit)).stem}_{row}.mp4",
                mime="video/mp4",
                key=f"download_{key}",
                width="stretch",
                on_click="ignore",
            )
    except DemoError as exc:
        st.warning(str(exc))
    with st.expander("Thông tin đoạn video"):
        start, end = source_bounds(hit)
        metadata = {
            "Video": clip_name(hit),
            "Bắt đầu": timestamp(start),
            "Kết thúc": timestamp(end),
            "Bộ video": collection["label"],
        }
        if "score" in hit:
            metadata["Điểm tương đồng"] = f"{hit['score']:.4f}"
        st.dataframe([{"Thông tin": k, "Giá trị": v} for k, v in metadata.items()], hide_index=True, width="stretch")
        st.caption("Điểm tương đồng dùng để xếp hạng kết quả, không phải xác suất chính xác.")
        evidence = (hit.get("evidence") or {}).get("matches", [])
        if evidence:
            st.markdown("**Dấu hiệu liên quan đến truy vấn**")
            st.dataframe(
                [
                    {
                        "Nhóm": m.get("family"),
                        "Đối tượng": m.get("detector_class", m.get("factor_type")),
                        "Bắt đầu (s)": m.get("start"),
                        "Kết thúc (s)": m.get("end"),
                    }
                    for m in evidence
                ],
                hide_index=True,
                width="stretch",
            )


def use_example(query):
    st.session_state.query = query


def restore_search(entry):
    st.session_state.update(
        query=entry["query"],
        payload=entry["payload"],
        searched_query=entry["query"],
        elapsed=entry["elapsed"],
        selected_row=None,
    )


def search_view(collection):
    st.markdown(
        '<div class="hero"><strong>Tìm đúng khoảnh khắc. Bằng một câu mô tả.</strong>'
        "<p>Mô tả phương tiện, hành động hoặc bối cảnh. Scene Search tìm những đoạn video liên quan "
        "và đưa bạn đến đúng thời điểm.</p></div>",
        unsafe_allow_html=True,
    )
    filter_key = collection["index_id"] if collection else "empty"
    filter_time = st.checkbox("Lọc khoảng thời gian trong video nguồn", key=f"time_filter_{filter_key}")
    with st.form("search_form"):
        query_col, count_col, button_col = st.columns([6, 1.5, 1.5], vertical_alignment="bottom")
        query = query_col.text_input(
            "Mô tả cảnh muốn tìm", key="query", max_chars=1000, placeholder="Ví dụ: white car overtaking on the right"
        )
        top_k = count_col.selectbox("Số kết quả", [1, 3, 5, 10, 20], index=2)
        mode_label = st.selectbox(
            "Chế độ tìm kiếm",
            ["EG2 — mặc định", "V6 shadow — thử nghiệm", "V6 active — thử nghiệm"],
            key=f"mode_{filter_key}",
        )
        mode = {"EG2 — mặc định": "eg2_only", "V6 shadow — thử nghiệm": "shadow", "V6 active — thử nghiệm": "active"}[
            mode_label
        ]
        clip_rows = collection_clips(base, collection["index_id"]) if collection else []
        source_names = {r["source_id"]: r["filename"] for r in clip_rows}
        selected_sources = st.multiselect(
            "Lọc video nguồn",
            list(source_names),
            format_func=lambda x: source_names[x],
            help="Để trống để tìm trong tất cả video.",
        )
        left_time, right_time = st.columns(2)
        start_filter = left_time.number_input(
            "Từ giây", min_value=0.0, value=0.0, disabled=not filter_time, key=f"start_{filter_key}"
        )
        end_filter = right_time.number_input(
            "Đến giây",
            min_value=0.1,
            value=max(0.1, max([r["end_sec"] for r in clip_rows], default=5.0)),
            disabled=not filter_time,
            key=f"end_{filter_key}",
        )
        filters = {"source_ids": selected_sources}
        if filter_time:
            filters.update(start_sec=start_filter, end_sec=end_filter)
        if mode != "eg2_only":
            st.caption("V6 là thử nghiệm. Kết quả khoa học đang HOLD; EG2 vẫn là mặc định.")

        submit = button_col.form_submit_button(
            "Tìm kiếm", type="primary", width="stretch", icon=":material/search:", disabled=collection is None
        )
    st.caption("Mẹo: mô tả ngắn, rõ hành động và bối cảnh. Bạn có thể tìm bằng tiếng Việt hoặc tiếng Anh.")
    examples = [
        ("Xe vượt bên phải", "white car overtaking on the right"),
        ("Người đi bộ qua đường", "a pedestrian crossing the road"),
        ("Cao tốc ban đêm", "night highway"),
    ]
    for col, (label, query_example) in zip(st.columns(3), examples, strict=True):
        col.button(label, on_click=use_example, args=(query_example,), width="stretch")
    if not collection:
        empty("Thư viện chưa có video để tìm kiếm", "Tải video lên và xử lý để bắt đầu.")
        st.button("Tải video đầu tiên", type="primary", on_click=navigate, args=("Tải video lên",))
        return
    if submit:
        if filter_time and end_filter <= start_filter:
            st.warning("Thời điểm kết thúc phải lớn hơn thời điểm bắt đầu.")
        elif not query.strip():
            st.warning("Vui lòng nhập mô tả cảnh trước khi tìm kiếm.")
        else:
            st.session_state.update(payload=None, selected_row=None)
            try:
                started = time.perf_counter()
                with st.spinner("Đang tìm các đoạn video liên quan… Lần khởi động đầu có thể mất vài phút."):
                    payload = client.search(
                        collection["index_id"],
                        query.strip(),
                        top_k,
                        uploaded=collection["uploaded"],
                        mode=mode,
                        filters=filters,
                    )
                elapsed = time.perf_counter() - started
                st.session_state.update(payload=payload, searched_query=query.strip(), elapsed=elapsed)
                history = st.session_state.history
                history.insert(
                    0,
                    {
                        "index_id": collection["index_id"],
                        "query": query.strip(),
                        "payload": payload,
                        "elapsed": elapsed,
                    },
                )
                del history[8:]
            except DemoError as exc:
                st.error(str(exc))
    payload = st.session_state.get("payload")
    if not payload:
        empty("Bạn đang tìm khoảnh khắc nào?", "Nhập mô tả hoặc chọn một gợi ý ở trên để bắt đầu.")
    else:
        hits = payload.get("results", [])
        if payload.get("decision", {}).get("fallback"):
            st.warning("Bằng chứng V6 không sẵn sàng. Đang trả thứ hạng EG2 an toàn.")
        elif payload.get("retrieval", {}).get("deployment_mode") == "shadow":
            st.caption("Shadow giữ thứ hạng EG2. Đề xuất V6 được ghi nhận để so sánh.")
            st.json(
                {"EG2 Top-5": payload["retrieval"].get("eg2_top5"), "V6 Top-5": payload["retrieval"].get("v6_top5")},
                expanded=False,
            )
        st.divider()
        heading, export = st.columns([4, 1])
        heading.subheader(f"{len(hits)} đoạn video liên quan")
        heading.caption(f"{st.session_state['searched_query']} · {st.session_state['elapsed']:.2f} giây")
        if hits:
            export.download_button(
                "Xuất kết quả",
                csv_results(hits),
                file_name="scenesearch_results.csv",
                mime="text/csv",
                icon=":material/download:",
                on_click="ignore",
                width="stretch",
            )
            visible = hits
            if not visible:
                st.info("Không có kết quả thuộc bối cảnh này. Chọn Tất cả để xem các đoạn đã tìm thấy.")
            else:
                options = [h["row"] for h in visible]
                if st.session_state.get("selected_row") not in options:
                    st.session_state.selected_row = options[0]
                list_col, preview_col = st.columns([1, 1.3], gap="large")
                with list_col:
                    for rank, hit in enumerate(visible, 1):
                        result_card(hit, rank, collection)
                with preview_col, st.container(border=True):
                    player(next(h for h in visible if h["row"] == st.session_state.selected_row), collection)
        else:
            empty("Chưa có kết quả để hiển thị", "Thử mô tả khác hoặc tìm trong một bộ video khác.")
    history = [h for h in st.session_state.history if h["index_id"] == collection["index_id"]]
    if history:
        with st.expander("Tìm kiếm gần đây"):
            for i, entry in enumerate(history):
                st.button(entry["query"], key=f"history_{i}", on_click=restore_search, args=(entry,), width="stretch")


STATUS_LABELS = {
    "receiving": "Đang tải lên",
    "uploaded": "Chờ xử lý",
    "indexing": "Đang xử lý",
    "ready": "Sẵn sàng",
    "failed": "Cần thử lại",
}
PROCESSING_ERRORS = {
    "INSUFFICIENT_PRODUCT_STORAGE": "Không đủ dung lượng để xử lý video. Hãy giải phóng dung lượng rồi thử lại.",
    "PROCESSING_INTERRUPTED": "Quá trình xử lý bị gián đoạn. Bạn có thể thử lại.",
    "VIDEO_TOO_LONG": "Video dài hơn 30 phút. Hãy cắt ngắn video trước khi tải lên.",
    "INVALID_VIDEO": "Không đọc được video. Hãy tải lại một tệp hợp lệ.",
    "EG2_WINDOW_DURATION_MISMATCH": "Không chuẩn bị được một đoạn video để xử lý. Vui lòng thử lại.",
    "EG2_INSUFFICIENT_FRAMES": "Một đoạn video có quá ít khung hình để xử lý. Vui lòng thử lại.",
    "SOURCE_CHECKSUM_MISMATCH": "Video lưu trữ đã bị thay đổi. Vui lòng tải video lên lại.",
}


@st.fragment(run_every="3s")
def processing_status(upload_id):
    try:
        record = client.request("GET", f"/api/v1/library/uploads/{upload_id}")
        st.markdown(f"**{escape(record['folder_name'])}** · {STATUS_LABELS.get(record['status'], record['status'])}")
        if record["status"] == "indexing":
            stages = {
                "prepare": "Chuẩn bị video",
                "privacy_and_geometry": "Che mặt, biển số và phân tích chuyển động",
                "embedding": "Tạo vector tìm kiếm",
                "publish": "Kiểm tra và công bố thư viện",
            }
            st.caption(stages.get(record.get("stage"), "Đang xử lý video"))
            st.progress(
                max(0.0, min(1.0, float(record.get("progress", 0)))),
                text=f"Đã xử lý {record.get('completed_scenes', 0)} / {record.get('total_scenes', '…')} đoạn video",
            )
            st.caption("Bạn có thể chuyển sang thư viện để theo dõi. Video sẽ sẵn sàng tìm kiếm khi xử lý xong.")
        elif record["status"] == "ready":
            st.success(f"Đã xử lý xong · {record['clips']} đoạn video sẵn sàng tìm kiếm.")
            workspace.clear()
            if st.button("Tìm trong bộ video này", type="primary", key=f"open_ready_{upload_id}"):
                st.session_state.update(pending_collection=upload_id, pending_page="Tìm kiếm")
                st.rerun()
        elif record["status"] == "failed":
            st.error(PROCESSING_ERRORS.get(record.get("error"), "Xử lý chưa hoàn tất. Vui lòng thử lại."))
            if record.get("total_scenes"):
                st.caption(
                    f"Đã hoàn tất {record.get('completed_scenes', 0)} / {record['total_scenes']} đoạn. "
                    "Những đoạn đã lưu tiến độ sẽ được tiếp tục khi bạn thử lại."
                )
        if record["status"] in {"uploaded", "failed"}:
            if st.button("Thử xử lý lại", key=f"retry_{upload_id}"):
                client.request("POST", f"/api/v1/library/uploads/{upload_id}/start")
                st.rerun(scope="fragment")
    except DemoError as exc:
        st.error(str(exc))


def upload_view():
    st.write("Thêm video vào thư viện, xử lý và tìm kiếm nội dung bằng một câu mô tả.")
    st.markdown(
        '<div class="hero"><strong>01 &nbsp; Tải lên &nbsp; → &nbsp; 02 &nbsp; Xử lý &nbsp; → &nbsp; '
        "03 &nbsp; Tìm kiếm</strong><p>Video được chia thành các đoạn 5 giây. Kết quả giữ lại mốc thời gian "
        "trong video gốc để bạn dễ kiểm tra và trích xuất.</p></div>",
        unsafe_allow_html=True,
    )
    if not upload_config["enabled"]:
        st.warning("Dịch vụ chuyển đổi video chưa sẵn sàng. Vui lòng thử lại sau.")
        return
    with st.container(border=True):
        name = st.text_input("Tên bộ video", placeholder="Ví dụ: Camera hành trình · Tháng 10", max_chars=255)
        files = st.file_uploader(
            "Chọn video",
            type=[e.lstrip(".") for e in upload_config["video_extensions"]],
            accept_multiple_files=True,
            max_upload_size=upload_config["max_file_bytes"] // 1024**2,
            key=f"upload_files_{st.session_state.get('upload_generation', 0)}",
        )
        st.caption(
            f"Tối đa {upload_config['max_files']} video mỗi lần · "
            f"{upload_config['max_file_bytes'] // 1024**2} MB/video · Thời lượng tối đa 30 phút/video."
        )
        if files:
            st.dataframe(
                [{"Video": f.name, "Dung lượng": f"{f.size / 1024**2:.1f} MB"} for f in files],
                hide_index=True,
                width="stretch",
            )
            st.caption("Bản xem trước chỉ xuất hiện sau khi hoàn tất bảo vệ riêng tư.")
        if st.button(
            "Tải lên và xử lý", type="primary", icon=":material/cloud_upload:", disabled=not files, width="stretch"
        ):
            if len(files) > upload_config["max_files"]:
                st.error(f"Vui lòng chọn tối đa {upload_config['max_files']} video mỗi lần.")
            elif not name.strip():
                st.warning("Vui lòng đặt tên cho bộ video trước khi tải lên.")
            elif len({f.name.casefold() for f in files}) != len(files):
                st.error("Các video có tên trùng nhau. Đổi tên tệp trước khi tải lên.")
            else:
                fingerprint = hashlib.sha256(
                    json.dumps([name.strip(), [(f.name, f.size, f.file_id) for f in files]]).encode()
                ).hexdigest()
                if st.session_state.get("upload_fingerprint") != fingerprint:
                    st.session_state.update(upload_fingerprint=fingerprint, upload_key=uuid4().hex)
                progress = st.progress(0, text="Đang tải video lên…")
                try:
                    record = client.upload(
                        files,
                        name.strip(),
                        st.session_state.upload_key,
                        lambda value: progress.progress(value, text="Đang tải video lên…"),
                    )
                    st.session_state.active_upload = record["upload_id"]
                    client.request("POST", f"/api/v1/library/uploads/{record['upload_id']}/start")
                    workspace.clear()
                    st.session_state.upload_generation = st.session_state.get("upload_generation", 0) + 1
                    st.rerun()
                except DemoError as exc:
                    st.error(str(exc))
                finally:
                    progress.empty()
    if st.session_state.get("active_upload"):
        with st.container(border=True):
            processing_status(st.session_state.active_upload)


def library_view(collection):
    toolbar, action = st.columns([4, 1])
    toolbar.write("Quản lý các bộ video và xem những đoạn đã sẵn sàng tìm kiếm.")
    action.button(
        "Thêm video", type="primary", on_click=navigate, args=("Tải video lên",), width="stretch", icon=":material/add:"
    )
    for record in uploads:
        with st.container(border=True):
            title, open_col, delete_col = st.columns([4, 1.4, 1])
            title.markdown(f"**{escape(record['folder_name'])}**")
            title.caption(f"{len(record['files'])} video · {STATUS_LABELS.get(record['status'], record['status'])}")
            if record["status"] == "ready":
                open_col.button(
                    "Tìm kiếm",
                    key=f"open_{record['upload_id']}",
                    width="stretch",
                    on_click=select_collection,
                    args=(record["upload_id"],),
                )
            elif record["status"] in {"uploaded", "failed"}:
                if open_col.button("Xử lý", key=f"process_{record['upload_id']}", width="stretch"):
                    try:
                        client.request("POST", f"/api/v1/library/uploads/{record['upload_id']}/start")
                        workspace.clear()
                        st.rerun()
                    except DemoError as exc:
                        st.error(str(exc))
            delete_col.button(
                "Xóa",
                key=f"delete_{record['upload_id']}",
                width="stretch",
                disabled=record["status"] == "indexing",
                on_click=lambda uid=record["upload_id"]: st.session_state.update(confirm_delete=uid),
            )
            if st.session_state.get("confirm_delete") == record["upload_id"]:
                st.warning("Xóa bộ video sẽ xóa các tệp đã tải lên và các đoạn video đã xử lý.")
                yes, no = st.columns(2)
                if yes.button("Xác nhận xóa", key=f"confirm_{record['upload_id']}", type="primary"):
                    try:
                        uid = record["upload_id"]
                        client.request("DELETE", f"/api/v1/library/uploads/{uid}")
                        st.session_state.saved = {
                            k: v for k, v in st.session_state.saved.items() if v["collection"]["index_id"] != uid
                        }
                        if st.session_state.get("active_upload") == uid:
                            st.session_state.pop("active_upload", None)
                        st.session_state.pop("confirm_delete", None)
                        st.session_state.pop("download", None)
                        st.session_state.update(payload=None, selected_row=None, library_clip=None)
                        st.session_state.history = [h for h in st.session_state.history if h["index_id"] != uid]
                        workspace.clear()
                        st.rerun()
                    except DemoError as exc:
                        st.error(str(exc))
                no.button(
                    "Giữ lại",
                    key=f"cancel_{record['upload_id']}",
                    on_click=lambda: st.session_state.pop("confirm_delete", None),
                )
            if record["status"] == "indexing":
                processing_status(record["upload_id"])
    if not collection:
        empty("Bắt đầu xây dựng thư viện của bạn", "Thêm video để tìm kiếm và xem lại những khoảnh khắc cần thiết.")
        return
    st.subheader(f"Các đoạn video · {collection['label']}")
    try:
        path = (
            f"/api/v1/library/{collection['index_id']}/clips"
            if collection["uploaded"]
            else f"/api/v1/library/indexes/{collection['index_id']}/clips"
        )
        clips = client.request("GET", path)["clips"]
        keyword = st.text_input("Tìm video theo tên", placeholder="Nhập tên video…")
        clips = [clip for clip in clips if keyword.casefold() in clip_name(clip).casefold()]
        if not clips:
            st.info("Không có video khớp với tên đã nhập.")
            return
        page = st.selectbox(
            "Trang", range(1, max(1, (len(clips) + 8) // 9) + 1), key=f"library_page_{collection['index_id']}"
        )
        columns = st.columns(3)
        for i, clip in enumerate(clips[(page - 1) * 9 : page * 9]):
            with columns[i % 3], st.container(border=True):
                try:
                    grant = client.preview(collection["index_id"], clip["row"], uploaded=collection["uploaded"])
                    st.image(grant["poster_url"], width="stretch")
                except DemoError:
                    st.caption("Ảnh xem trước chưa sẵn sàng")
                st.markdown(f"**{escape(clip_name(clip))}**")
                st.caption(interval(clip))
                st.button(
                    "Mở video",
                    key=f"library_clip_{clip['row']}",
                    width="stretch",
                    on_click=lambda hit=clip: st.session_state.update(library_clip=hit),
                )
        selected = st.session_state.get("library_clip")
        if selected and any(clip["row"] == selected["row"] for clip in clips):
            with st.container(border=True):
                player(selected, collection, key="library")
    except DemoError as exc:
        st.warning(str(exc))


def saved_view():
    st.write("Những đoạn video bạn đã đánh dấu để xem lại.")
    st.caption("Danh sách đã lưu và lịch sử tìm kiếm được giữ trong phiên làm việc hiện tại.")
    saved = st.session_state.saved
    if not saved:
        empty("Chưa có đoạn video đã lưu", "Bấm Lưu trên một kết quả tìm kiếm để thêm vào đây.")
        st.button("Bắt đầu tìm kiếm", on_click=navigate, args=("Tìm kiếm",), type="primary")
        return
    st.download_button(
        "Xuất danh sách đã lưu",
        json.dumps(list(saved.values()), ensure_ascii=False, indent=2),
        file_name="scenesearch_saved.json",
        mime="application/json",
        on_click="ignore",
    )
    selected = st.selectbox(
        "Chọn đoạn đã lưu",
        list(saved),
        format_func=lambda k: f"{clip_name(saved[k]['hit'])} · {interval(saved[k]['hit'])}",
    )
    entry = saved[selected]
    with st.container(border=True):
        player(entry["hit"], entry["collection"], key="saved")
        st.button("Bỏ lưu đoạn video", on_click=save_result, args=(entry["hit"], entry["collection"]))


@st.cache_data(ttl=5, show_spinner=False)
def collection_clips(base, uid):
    return DemoClient(base).request("GET", f"/api/v1/library/{uid}/clips")["clips"]


def system_view():
    st.write("Xử lý và tìm kiếm chạy trên máy này. EG2 là mặc định; V6 đang thử nghiệm.")
    capabilities = client.request("GET", "/api/v1/capabilities")
    state = client.request("GET", "/health")
    st.success("Mô hình đã khởi động và sẵn sàng.") if state.get("ready") else st.info(
        "Mô hình đang khởi động hoặc GPU đang xử lý video."
    )
    st.dataframe(
        [
            {"Mục": "Bảo vệ bản xem trước", "Trạng thái": capabilities["privacy"]},
            {"Mục": "Mặc định", "Trạng thái": capabilities["default_mode"]},
            {"Mục": "Đánh giá V6", "Trạng thái": capabilities["scientific_status"]},
            {"Mục": "Mục tiêu Precision / Recall / nDCG", "Trạng thái": capabilities["precision_recall_ndcg"]},
            {"Mục": "Dịch vụ ngoài", "Trạng thái": "Tắt"},
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption("Ứng dụng cho một người dùng tin cậy trên localhost; tiêu đề định danh không phải hệ thống đăng nhập.")
    if uploads:
        st.dataframe(
            [
                {
                    "Bộ video": r["folder_name"],
                    "Trạng thái": STATUS_LABELS.get(r["status"], r["status"]),
                    "Riêng tư": r.get("privacy_state", "Chưa sẵn sàng"),
                    "V6": r.get("v6_state", "Chưa sẵn sàng"),
                }
                for r in uploads
            ],
            hide_index=True,
            width="stretch",
        )
    if st.button("Kiểm tra lại trạng thái"):
        workspace.clear()
        st.rerun()


for key, default in {"saved": {}, "history": [], "page": "Tìm kiếm", "query": ""}.items():
    if key not in st.session_state:
        st.session_state[key] = default
try:
    base = backend_url()
    client = DemoClient(base)
    health, indexes, uploads, upload_config = workspace(base)
    valid_ids = {item["index_id"] for item in indexes}
    st.session_state.saved = {
        k: v for k, v in st.session_state.saved.items() if v["collection"]["index_id"] in valid_ids
    }
    st.session_state.history = [h for h in st.session_state.history if h["index_id"] in valid_ids]
    if st.session_state.get("download", {}).get("key", "").split(":")[0] not in valid_ids:
        st.session_state.pop("download", None)
except DemoError as exc:
    st.title("Scene Search")
    st.error(str(exc))
    st.caption("Thư viện sẽ hiển thị khi kết nối được khôi phục.")
    if st.button("Kết nối lại", type="primary"):
        workspace.clear()
        st.rerun()
    st.stop()

if "pending_collection" in st.session_state:
    st.session_state.collection = st.session_state.pop("pending_collection")
    st.session_state.page = st.session_state.pop("pending_page")

with st.sidebar:
    st.markdown('<div class="brand"><span class="brand-icon">S</span> Scene Search</div>', unsafe_allow_html=True)
    st.caption("Thư viện video thông minh")
    st.divider()
    st.radio(
        "Không gian làm việc",
        ["Tìm kiếm", "Thư viện video", "Tải video lên", "Đã lưu", "Trạng thái hệ thống"],
        key="page",
        label_visibility="collapsed",
    )
    st.divider()
    if indexes:
        choices = [item["index_id"] for item in indexes]
        if st.session_state.get("collection") not in choices:
            st.session_state.collection = choices[0]
        index_id = st.selectbox(
            "Bộ video đang mở",
            choices,
            key="collection",
            format_func=lambda uid: next(item["label"] for item in indexes if item["index_id"] == uid),
        )
        collection = next(item for item in indexes if item["index_id"] == index_id)
        st.caption(f"{collection['clips']} đoạn video · Sẵn sàng tìm kiếm")
    else:
        collection, index_id = None, None
        st.caption("Chưa có bộ video sẵn sàng")
    st.caption("● Sẵn sàng tìm kiếm" if health.get("ready") else "● Đang khởi động hoặc xử lý")
    if st.button("Làm mới thư viện", width="stretch", icon=":material/refresh:"):
        workspace.clear()
        st.rerun()
    with st.expander("Hướng dẫn sử dụng"):
        st.write(
            "1. Chọn một bộ video hoặc tải video mới lên.\n\n2. Mô tả cảnh cần tìm, rồi bấm Tìm kiếm.\n\n"
            "3. Mở đoạn liên quan, lưu lại hoặc tải tệp MP4."
        )
    st.caption("Không gian làm việc cá nhân")

if st.session_state.get("active_collection") != index_id:
    st.session_state.update(active_collection=index_id, payload=None, selected_row=None, library_clip=None)
page = st.session_state.page
st.markdown('<div class="eyebrow">VIDEO INTELLIGENCE WORKSPACE</div>', unsafe_allow_html=True)
st.title(
    {
        "Tìm kiếm": "Tìm kiếm khoảnh khắc",
        "Thư viện video": "Thư viện video",
        "Tải video lên": "Thêm video mới",
        "Đã lưu": "Khoảnh khắc đã lưu",
        "Trạng thái hệ thống": "Trạng thái hệ thống",
    }[page]
)
if page == "Tìm kiếm":
    st.write("Tra cứu nội dung video bằng ngôn ngữ tự nhiên.")
    clips_col, collections_col, saved_col = st.columns(3)
    clips_col.metric("Đoạn video", sum(item["clips"] for item in indexes))
    collections_col.metric("Bộ video", len(indexes))
    saved_col.metric("Đã lưu", len(st.session_state.saved))
    search_view(collection)
elif page == "Thư viện video":
    library_view(collection)
elif page == "Tải video lên":
    upload_view()
elif page == "Đã lưu":
    saved_view()
else:
    system_view()
