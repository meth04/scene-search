# Product status

The supported local application is `run_demo.ps1`: Streamlit, local FastAPI,
frozen EmbeddingGemma2 retrieval and privacy-v2.1 face/plate redaction.
EG2 is the default. DriveEvidence V6 remains an explicit experimental
shadow/active option with exact EG2 fallback.

V6.1 and V6.2 were rejected and are not installed as product modes. The V6.2
BDD100K study stopped at calibration: no tested threshold met the predeclared
harmful-risk and minimum-intervention requirements. Qualification and external
transfer quality were not run. FINAL_V6 and SHADOW were not accessed. This is
not a scientific final qualification or a general retrieval-accuracy claim.

Daily operation requires no benchmark, training, remote server or API key.
Model/config/scorer identities remain pinned by `configs/product/assets.json`.
Privacy previews and exports use validated redacted media; there is no raw
media fallback. Inspect redacted output before sharing it.

One-off experiment drivers, unused training stages, generated evaluation
reports and compiler caches have been removed from the source tree. Historical
tracked implementations remain recoverable through Git. Shared runtime code
and regression tests are retained, including older modules that the frozen
product imports. The HTML presentation and shared service infrastructure remain
separate from the canonical local launcher.

Use [the product guide](FINAL_LOCAL_PRODUCT_GUIDE.md) for installation,
operation, privacy contracts, diagnostics and acceptance commands.

The public Windows installer supports CPU and NVIDIA execution, always using the
same FP32 model/representation/scorer contracts. It downloads checksum-pinned
public runtime assets and managed FFmpeg. The unused historical Stage3A learned
checkpoint is provenance-only and is not required on a new machine.
See [INSTALL.md](../INSTALL.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).
