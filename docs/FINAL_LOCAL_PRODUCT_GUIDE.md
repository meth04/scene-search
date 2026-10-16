# Scene Search — final local product guide

The canonical application is Streamlit (`demo/streamlit_app.py`) with the local
FastAPI service (`demo/backend.py`). Start both with `run_demo.ps1`.
The HTML presentation and shared service infrastructure are separate from this
product path. Unused standalone research and training drivers have been removed;
compatibility modules still required by runtime or tests remain.

## Quick start on Windows (CPU or NVIDIA)

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
powershell -NoProfile -File run_demo.ps1
```

Open **http://127.0.0.1:8501**. API: **http://127.0.0.1:8090**.
Keep the launcher terminal open; Ctrl+C stops only the services it started.
The first start verifies local asset hashes, starts one frozen EG2 worker,
warms text inference and verifies existing published libraries before readiness.
Daily launch does not install packages, download models, or run evaluations.

```powershell
# Non-destructive checks; starts no services and processes no video:
powershell -NoProfile -File run_demo.ps1 -Verify
# Alternative free ports:
powershell -NoProfile -File run_demo.ps1 -Port 8502 -ApiPort 8091
```

The launcher resolves repository paths relative to its own location. From another
directory, invoke its absolute path. Do not start a second UI on an occupied port.
An existing compatible local API can be reused; another service on the API port
is rejected.

## Initial installation and local assets

[INSTALL.md](../INSTALL.md) is the canonical fresh-machine installation guide.
It covers the public source/model releases, Python 3.13, CPU/CUDA selection,
resuming downloads and repair. Three isolated environments remain `.venv`,
`.venv_embeddinggemma2` and `.venv_demo`.

The installer downloads only explicit public URLs and verifies every declared
model's size and SHA-256 using `configs/product/downloads.json`. It also installs
managed FFmpeg/ffprobe under `data/bin/ffmpeg/`. A model bundle is available for
transferring the same pinned model bytes between machines.

The model, query prefix, FP32, normalization, sampling and selected V6 scorer
remain unchanged. Product-owned execution adapters support CPU and other NVIDIA
GPUs. The original GTX1650 execution path has native bit-parity checks. This does
not claim bit-identical floating-point output across different hardware.

The historical Stage3A learned checkpoint is provenance-only and is not loaded
by product geometry. New installations do not need it, optimizer states or
research datasets. Old publications may retain its exact historical metadata
hash; all actual runtime model hashes remain mandatory.

See [third-party notices](../THIRD_PARTY_NOTICES.md) for model/software licenses.
Daily launch never downloads models, installs packages or runs research.

## Using the application

1. Open **Tải video lên**, name the collection, select one or more MP4 files,
   then choose **Tải lên và xử lý**. Receiving bytes and completing indexing are
   separate durable states. Limits: 20 files, 200 MB per file, 30 minutes per file.
2. Follow preparation, privacy/geometry, embedding and publication progress.
   Closing the browser does not stop the backend job. A backend interruption
   becomes retryable after restart. Choose **Thử lại** or **Xử lý** on a failed
   upload; verified phase receipts are reused.
3. Once ready, open the collection and search with a natural language description.
   Choose source videos and an optional time range before searching. Time filters
   select clips overlapping that interval; playback/export retains each full
   actual clip's source bounds.
4. Open results, save selections for this browser session, export CSV/JSON,
   or download the private MP4. Library cards provide private posters.
5. Delete a collection in the library. Search, grants, cached indexes and session
   references are revoked. Deletion briefly pauses GPU use to release the frozen
   worker's Windows memory map and warm the encoder again.

Saved results and search history are session features, not a persistent account.
No raw upload preview is displayed. Processing must complete privacy publication
before previews, posters, playback or downloads become available.

## Retrieval modes and scientific status

**EG2 is the production default.** V6 is fully integrated but experimental.

| Mode | Returned order | Purpose |
|---|---|---|
| `eg2_only` | EG2 cosine ranking | Default local product |
| `shadow` | EG2 ranking | Compute/display V6 comparison without changing order |
| `active` | Selected bounded V6 proposal | Explicit experimental choice |

The encoder and scoring policy are frozen: EG2-B, five endpoint frames,
768-dimensional normalized FP32 vectors, fixed query prefix, audio disabled;
eight FRAME units, three two-second WINDOW units, up to three four-frame TUBEs;
selected V6 alpha 0.25, TopK 5 and pairwise protection. Source/time filtering
happens before EG2 Top-5 selection and V6 scoring. No detector, tracker, visual
encoder, planner, Qwen or external inference is called during search.

V6 can only promote a candidate from the original Top-5; the candidate set and
tail remain unchanged. Invalid local evidence, runtime or telemetry failure
returns the exact EG2 order with an explicit fallback. Invalid GLOBAL vectors,
source mapping, privacy or frozen identities block access.

Similarity is a ranking score, not calibrated confidence. Empty filtered results
mean no clips matched the filter; the application does not infer semantic absence.
Vehicle colors, physical entity identities and day/night are not certified product
metadata filters. These controls are not exposed without corresponding metadata.

Scientific status is **HOLD**. The existing predeclared gate lacks a qualified
freeze and all three safe outer folds. FINAL_V6 is NOT_RUN; SHADOW is untouched.
The supported modes and research decisions are summarized in
[product status](PRODUCT_STATUS.md). Precision/recall/nDCG ≥0.70 is
**NOT_ESTABLISHED**, and is not an application claim.

## Index and privacy contracts

Product data lives under `outputs/product/library`. Each upload has a UUID-owned
directory with original files, `_product/cache` phase receipts, and an immutable
`_product/published` index. The product schema is
`driveevidence-v6-product-v1`; it does not relabel arbitrary user videos as TRAIN
or relax the separate research loader.

Clips use actual source-relative bounds: `[0,5)`, `[5,10)`, and the final real tail.
Low-frame-rate or short clips receive an inference-only five-second padded input.
Playback is not padded; evidence from held final frames does not establish new
motion. Public results carry unique clip IDs, source IDs, original filenames and
actual source bounds. Their row resolves its own source, never a representative
first video or a concatenated dataset offset.

Offline privacy uses the existing v2.1 scanner/tracker and strong pixelation,
mandatory global face/plate passes, ROI refinement and small-object rescue.
Product privacy detection uses FP32 on CPU/CUDA to avoid the previously observed plate
pre-NMS FP16 failure. Retrieval inference remains the frozen FP32 EG2 path.
Published video is muted; audio is not analyzed or exported. Posters come from
private video. A complete track cache, its receipt and every media artifact are
hash-bound before publication. No raw fallback exists.

Human privacy certification is unavailable. The automated contract and a limited
agent visual inspection pass do not certify detector recall; small/distant targets
can be missed and signs can be falsely masked. Inspect private output before
sharing. The deterministic development review packet is under
`outputs/finalization/privacy_review`.

Resume identities bind source bytes/bounds, frozen model/config/scorer identities,
decoded frames, specifications and privacy receipts. Completed vectors are reused
only after their hashes, dimensions, finiteness and unit norms pass. A new
publication is validated in a private staging directory and renamed atomically
on the same volume. Invalid generations are quarantined, never edited in place or
served. The retry endpoint can repair an invalid publication from valid owned
working caches. Research indexes and historical reports remain read-only.

Decoded working caches are much larger than the semantic index and are retained
for reliable resume. Working-cache size depends on video duration and resolution. Delete a collection
to remove its working data.
Indexing estimates the remaining decoded-cache requirement before encoding and
rejects insufficient free space with a retryable storage error. The stated upload
limits do not imply that every such collection fits on the current disk.

## API and diagnostics

`GET /health` is liveness; `GET /ready` is warmed runtime readiness. Both are local.
`GET /api/v1/library/{upload_id}/readiness` checks that collection's GLOBAL,
source, schema, policy, representation and privacy, and reports V6 degradation.
`GET /api/v1/capabilities` documents supported modes, filters and claims.
API documentation: `http://127.0.0.1:8090/docs`.

The upload API supports create, streamed bytes, complete, start, status, retry and
delete. Create is idempotent with `Idempotency-Key`. Source data is never served
directly. Preview grants are short-lived and bind collection/row/actor; media and
poster routes recheck publication integrity on every request. Range requests use
the validated private file. Download uses the same private bytes.

`POST /api/v1/library/{id}/search` accepts `query`, `top_k`, `mode` and `filters`
(`source_ids`, `start_sec`, `end_sec`). Unavailable metadata fields are rejected.
`POST /api/v1/library/{id}/export` accepts selected row IDs and JSON/CSV format.
Provide a `search` object to include actual returned rank, global/final scores and
mode; exporting plain library selections leaves scores empty. CSV strings are
escaped against spreadsheet formulas. The UI CSV includes search scores and mode.

The backend admits at most four search requests to the shared GPU slot and returns
429 when busy beyond its bounded wait. Indexing uses that same slot. Readiness and
status remain responsive during offline processing. No model unload/reload occurs
per query; literal query caches are separated by mode and frozen identities.

Logs are under `outputs/product/logs`. Product search telemetry under
`outputs/product/library/telemetry/search.jsonl` contains request IDs, hashed
queries, mode/index/policy identities, decisions, fallback and timings. Access logs
are disabled; literal queries are not written to telemetry. Queries remain visible
to the user in their own response and browser session history.

This is one trusted local user's application. Loopback binding, host/origin checks,
restricted CORS and signed media grants are enforced. Convenience identity headers
are not authentication for a hostile multi-user server. Do not expose these ports
through a proxy or bind them to a LAN address.

## Troubleshooting

| Symptom | Action |
|---|---|
| Doctor reports missing/changed asset | Restore the exact pinned asset; do not replace hashes or weights |
| Missing environment/dependency | Run setup with Python 3.13; the daily launcher does not install dependencies |
| Port occupied | Stop the owned launcher or choose free `-Port`/`-ApiPort` values |
| Startup warming | Wait for readiness; inspect `outputs/product/logs/worker.stderr.log` on failure |
| Busy / 429 | Wait for indexing to finish or retry the bounded query queue |
| Interrupted processing | Restart the application, then retry the upload; verified receipts are reused |
| Privacy/source/GLOBAL integrity failure | Access is blocked; retry the owned upload or upload the source again |
| V6 local evidence failure | EG2 remains available with a degradation notice; retry repairs the owned index |
| Expired preview | Use **Tải lại video** to request a fresh grant |
| Deleted item | Its old grants and search return 404; select a remaining ready collection |

## Reproducible checks

```powershell
powershell -NoProfile -File run_demo.ps1 -Verify
powershell -NoProfile -File scripts/test_local_product.ps1
.venv\Scripts\python.exe -B scripts/verify_product_install.py --device cpu
```

The optional native smoke uses two generated videos, actual installed models,
private media, search/shadow/active, exact fallback and delete/revoke checks.
Stop the running application before the smoke so only its owned encoder is
resident. Artifacts go under `outputs/install_acceptance`; no dataset or
scientific reserve is read. This is product acceptance, not retrieval-quality
qualification.

Current supported behavior is documented in [product status](PRODUCT_STATUS.md).
