"""Runtime assets versus historical checkpoints never used by this product."""

from __future__ import annotations


def runtime_model_assets(stored, manifest):
    required = manifest["assets"]
    historical = manifest.get("provenance_assets", {})
    # Explicit compatibility for old publications. Only the pinned, unused
    # provenance checkpoint may be omitted from a new installation.
    if stored != required and stored != {**required, **historical}:
        raise ValueError("PRODUCT_MODEL_IDENTITIES_INVALID")
    return required
