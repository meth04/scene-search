"""Build a public source snapshot/model bundle from explicit allowlists only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "INSTALL.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "ARCHITECTURE.md",
    "SETUP_ENGINE.md",
    "run_demo.ps1",
    "install.ps1",
    "ruff.toml",
}
SCRIPT_FILES = {
    "run_streamlit_demo_backend.py",
    "setup_local_product.ps1",
    "test_local_product.ps1",
    "install_product_assets.py",
    "package_product_release.py",
    "verify_product_install.py",
    "build_driveevidence_v3.py",
    "build_driveevidence_v6.py",
    "prepare_driveevidence_v3.py",
    "prepare_driveevidence_v6.py",
    "prepare_driveevidence_v2.py",
    "prepare_driveevidence.py",
    "freeze_driveevidence_v3_dev.py",
    "README.md",
}


def checksum(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def source_files():
    selected = {ROOT / name for name in ROOT_FILES if (ROOT / name).is_file()}
    for folder in ("demo", "hybrid_v2", "src"):
        selected.update((ROOT / folder).rglob("*.py"))
    selected.update(ROOT / "scripts" / name for name in SCRIPT_FILES if (ROOT / "scripts" / name).is_file())
    for folder in ("configs/product", "configs/models", "configs/indexing", "configs/retrieval", "configs/metadata"):
        selected.update(p for p in (ROOT / folder).glob("*") if p.suffix in {".json", ".yaml", ".lock"})
    selected.add(ROOT / "configs/hybrid/driveevidence_v6.yaml")
    for folder in ("docs/licenses", "tests/test_demo"):
        selected.update(p for p in (ROOT / folder).rglob("*") if p.is_file() and "__pycache__" not in p.parts)
    for name in (
        "stage2a/__init__.py",
        "stage2a/stage2a_common.py",
        "stage3a/__init__.py",
        "stage3a/stage3a_infer.py",
        "stage3a/stage3a_model.py",
        "stage3a/stage3a_motion.py",
        "docs/FINAL_LOCAL_PRODUCT_GUIDE.md",
        "docs/PRODUCT_STATUS.md",
        "tests/__init__.py",
        "tests/test_installation.py",
    ):
        if (ROOT / name).is_file():
            selected.add(ROOT / name)
    for path in selected:
        if not path.resolve().is_relative_to(ROOT) or path.name in {"remote.txt", ".env"}:
            raise ValueError("Invalid release source")
        if path.suffix in {".py", ".json", ".yaml", ".ps1", ".md", ".lock"}:
            value = path.read_text(encoding="utf-8-sig")
            if re.search(
                r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----|gh[pousr]_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{32,}|AKIA[A-Z0-9]{16}",
                value,
            ):
                raise ValueError("Potential credential in release source: " + path.relative_to(ROOT).as_posix())
    return sorted(selected)


def export(destination):
    if not destination.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Release staging must be under outputs")
    records = {}
    for path in source_files():
        relative = path.relative_to(ROOT)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        records[relative.as_posix()] = {"bytes": target.stat().st_size, "sha256": checksum(target)}
    (destination / "SOURCE_MANIFEST.json").write_text(json.dumps({"schema": 1, "files": records}, indent=2) + "\n")
    return records


def models(target):
    manifest = json.loads((ROOT / "configs/product/downloads.json").read_text())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".building")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as archive:
        for record in manifest["models"]:
            path = ROOT / record["path"]
            if not path.resolve().is_relative_to((ROOT / "data/model-cache").resolve()):
                raise ValueError("Only declared model files may enter the model bundle")
            if path.stat().st_size != record["bytes"] or checksum(path) != record["sha256"]:
                raise ValueError("Model changed before publication: " + record["path"])
            archive.write(path, record["path"])
        archive.writestr("ASSET_MANIFEST.json", json.dumps(manifest, indent=2))
        for path in [ROOT / "THIRD_PARTY_NOTICES.md", *(ROOT / "docs/licenses").glob("*")]:
            if path.is_file():
                archive.write(path, path.relative_to(ROOT).as_posix())
    temporary.replace(target)
    result = {"bytes": target.stat().st_size, "sha256": checksum(target)}
    if result["bytes"] >= 2 * 1024**3:
        raise ValueError("Model bundle exceeds GitHub's per-asset limit")
    print(json.dumps({"model_bundle": str(target), **result}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "outputs/product_release/public_source")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--models", type=Path)
    args = parser.parse_args()
    if args.models:
        models(args.models)
    records = export(args.destination)
    if args.archive:
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.archive, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in [*records, "SOURCE_MANIFEST.json"]:
                archive.write(args.destination / name, "scene-search/" + name)
    print(f"Public source allowlist: {len(records)} files; no credentials, history, datasets, outputs or model caches.")


if __name__ == "__main__":
    main()
