"""Download only declared public runtime assets, with bounded size and SHA-256."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def checksum(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            result.update(block)
    return result.hexdigest()


def destination(root, relative):
    name = PurePosixPath(relative)
    if name.is_absolute() or ".." in name.parts or "\\" in relative or ":" in relative:
        raise ValueError("Invalid asset destination")
    target = root.joinpath(*name.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("Asset destination escapes installation")
    return target


def valid(path, record):
    return path.is_file() and path.stat().st_size == record["bytes"] and checksum(path) == record["sha256"]


def download(record, target, *, repair=False, offline=False, opener=urllib.request.urlopen):
    if valid(target, record):
        return
    if target.exists() and not repair:
        raise ValueError(f"Changed existing asset: {record['path']}; use -Repair to replace it")
    if offline:
        raise ValueError(f"Missing/changed offline asset: {record['path']}")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".download")
    for attempt in range(3):
        start = partial.stat().st_size if partial.exists() else 0
        if start > record["bytes"]:
            partial.unlink()
            start = 0
        if start == record["bytes"]:
            if valid(partial, record):
                os.replace(partial, target)
                return
            partial.unlink()
            start = 0
        request = urllib.request.Request(record["url"], headers={"User-Agent": "SceneSearch-Installer/1.0"})
        if start:
            request.add_header("Range", f"bytes={start}-")
        try:
            with opener(request, timeout=90) as response:
                if start and response.status == 206:
                    wanted = f"bytes {start}-{record['bytes'] - 1}/{record['bytes']}"
                    if response.headers.get("Content-Range") != wanted:
                        raise ValueError("Invalid download range")
                elif response.status == 200:
                    start = 0  # Server ignored Range; restart rather than append.
                else:
                    raise ValueError("Unexpected download status")
                written, last_progress = start, time.monotonic()
                with partial.open("ab" if start else "wb") as stream:
                    while block := response.read(1024**2):
                        written += len(block)
                        if written > record["bytes"]:
                            raise ValueError("Asset exceeds declared size")
                        stream.write(block)
                        if time.monotonic() - last_progress > 15:
                            print(
                                f"Downloading {Path(record['path']).name}: {100 * written / record['bytes']:.0f}%",
                                flush=True,
                            )
                            last_progress = time.monotonic()
            if not valid(partial, record):
                partial.unlink(missing_ok=True)
                raise ValueError("Downloaded asset size/SHA-256 mismatch")
            os.replace(partial, target)
            print(f"Verified {record['path']}", flush=True)
            return
        except (OSError, urllib.error.URLError) as exc:
            if attempt == 2:
                # Never print redirect URLs, access tokens or auth headers.
                raise RuntimeError(f"Download failed for {record['path']} ({type(exc).__name__})") from None
            time.sleep(1 + attempt)


def import_bundle(bundle, records, root, *, repair=False):
    with zipfile.ZipFile(bundle) as archive:
        for record in records:
            target = destination(root, record["path"])
            if valid(target, record):
                continue
            if target.exists() and not repair:
                raise ValueError(f"Changed existing asset: {record['path']}")
            entry = archive.getinfo(record["path"])
            if entry.file_size != record["bytes"] or entry.is_dir():
                raise ValueError("Bundle size mismatch")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".download")
            with archive.open(entry) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024**2)
            if not valid(temporary, record):
                temporary.unlink(missing_ok=True)
                raise ValueError("Bundle SHA-256 mismatch")
            os.replace(temporary, target)


def install_tools(root, record, *, repair=False, offline=False):
    receipt_path = destination(root, "data/bin/ffmpeg/receipt.json")
    try:
        receipt = json.loads(receipt_path.read_text())
        expected_paths = {"data/bin/ffmpeg/bin/ffmpeg.exe", "data/bin/ffmpeg/bin/ffprobe.exe"}
        if (
            set(receipt["files"]) == expected_paths
            and receipt["archive_sha256"] == record["sha256"]
            and all(valid(destination(root, name), value) for name, value in receipt["files"].items())
        ):
            return
    except (OSError, ValueError, KeyError):
        pass
    archive_path = destination(root, record["path"])
    download(record, archive_path, repair=repair, offline=offline)
    outputs = {}
    with zipfile.ZipFile(archive_path) as archive:
        for binary in ("ffmpeg.exe", "ffprobe.exe"):
            candidates = [name for name in archive.namelist() if name.endswith("/bin/" + binary)]
            if len(candidates) != 1:
                raise ValueError("FFmpeg archive layout changed")
            entry = archive.getinfo(candidates[0])
            if not 0 < entry.file_size < 300 * 1024**2:
                raise ValueError("FFmpeg binary size invalid")
            relative = "data/bin/ffmpeg/bin/" + binary
            target = destination(root, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".install")
            with archive.open(entry) as source, temporary.open("wb") as stream:
                shutil.copyfileobj(source, stream, length=1024**2)
            outputs[relative] = {"bytes": temporary.stat().st_size, "sha256": checksum(temporary)}
            if target.exists() and not repair:
                same = valid(target, outputs[relative])
                temporary.unlink()
                if not same:
                    raise ValueError("Existing managed FFmpeg changed; use -Repair")
                continue  # Resume an interrupted extraction without replacing valid bytes.
            os.replace(temporary, target)
    temporary_receipt = receipt_path.with_suffix(".install")
    temporary_receipt.write_text(json.dumps({"archive_sha256": record["sha256"], "files": outputs}, indent=2))
    os.replace(temporary_receipt, receipt_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--repair", action="store_true")
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((ROOT / "configs/product/downloads.json").read_text(encoding="utf-8"))
    for record in [*manifest["models"], manifest["ffmpeg_windows"]]:
        if not record["url"].startswith("https://") or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]):
            raise ValueError("Invalid pinned download record")
        if not 0 < record["bytes"] < 2 * 1024**3:
            raise ValueError("Invalid pinned asset size")
    if args.bundle:
        import_bundle(args.bundle, manifest["models"], root, repair=args.repair)
    for record in manifest["models"]:
        download(record, destination(root, record["path"]), repair=args.repair, offline=args.offline)
    install_tools(root, manifest["ffmpeg_windows"], repair=args.repair, offline=args.offline)
    print("Runtime models and FFmpeg verified. No datasets, indexes or training checkpoints required.", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, KeyError) as error:
        print(f"Asset setup failed: {error}", flush=True)
        raise SystemExit(1) from None
