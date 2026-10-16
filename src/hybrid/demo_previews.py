"""Authorize and validate per-clip derivatives without loading retrieval models."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from pathlib import Path

from fastapi import HTTPException

from src.hybrid.service import ROOT
from src.retrieval.domain.entities import ActorContext

PREVIEWS = ROOT / 'outputs/driveevidence_v6/ui_demo/previews'
SECRET = secrets.token_bytes(32)
TTL = 300


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def context(service, index_id, actor):
    index = service.resolve_index(index_id, actor)
    if read(index / 'index_meta.json').get('format') != 'driveevidence-v6':
        raise HTTPException(404, 'PREVIEW_UNAVAILABLE')
    rows = read(index / 'manifest.json')['clips']
    manifest_hash = digest(index / 'manifest.json')
    catalog = read(PREVIEWS / 'catalog.json')
    if catalog['manifest_sha256'] != manifest_hash or catalog['index_id'] != index_id:
        raise ValueError('Preview catalog belongs to another index')
    return rows, manifest_hash, catalog


def resolve(service, index_id, row_id, actor, asset='preview.mp4', *, snapshot=None):
    rows, manifest_hash, catalog = snapshot or context(service, index_id, actor)
    row = next((r for r in rows if r['row'] == row_id), None)
    if row is None or row['split'] != 'train':
        raise HTTPException(404, 'PREVIEW_UNAVAILABLE')
    source = service.source(row['source_path'])
    receipt = read(PREVIEWS / str(row_id) / 'receipt.json')
    if (receipt != catalog['entries'][str(row_id)] or receipt['row'] != row_id
        or receipt['clip_id'] != row['clip_id'] or receipt['manifest_sha256'] != manifest_hash
        or receipt['source_sha256'] != row['source_sha256'] or digest(source) != row['source_sha256']
        or receipt['source_start_sec'] != row['source_start_sec'] or receipt['source_end_sec'] != row['source_end_sec']
        or not receipt['privacy_scan_complete'] or receipt['privacy_mode'] != 'pixelate'):
        raise ValueError('Preview/source identity mismatch')
    cache = PREVIEWS / str(row_id) / 'privacy'
    privacy = read(cache / 'privacy_meta.json')
    tracks_name = privacy['tracks_file']
    if (Path(tracks_name).name != tracks_name or not tracks_name.endswith('.npz')
        or not privacy['complete'] or privacy['video']['sha256'] != row['source_sha256']
        or privacy['format'] != receipt['privacy_format'] or privacy['format'] != 'scene-search-privacy-v2.1'
        or privacy['covered_frames'] != privacy['video']['frame_count']
        or privacy['tracks_sha256'] != receipt['privacy_tracks_sha256']
        or digest(cache / tracks_name) != privacy['tracks_sha256']):
        raise ValueError('Privacy generation changed')
    if asset not in ('preview.mp4', 'poster.jpg'):
        raise ValueError('Invalid preview asset')
    path = PREVIEWS / str(row_id) / asset
    expected = receipt['preview_sha256' if asset == 'preview.mp4' else 'poster_sha256']
    if digest(path) != expected:
        raise ValueError('Preview checksum changed')
    return path, receipt


def available(service, index_id, actor):
    snapshot = context(service, index_id, actor)
    rows, _, catalog = snapshot
    entries = []
    for row in rows:
        if str(row['row']) not in catalog['entries']:
            continue
        try:
            _, receipt = resolve(service, index_id, row['row'], actor, snapshot=snapshot)
            entries.append(dict(row=row['row'], clip_id=row['clip_id'],
                start_sec=receipt['preview_start_sec'], end_sec=receipt['preview_end_sec']))
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return dict(index_id=index_id, clips=len(rows), prepared=len(entries), previews=entries)


def grant(index_id, row, actor):
    payload = dict(index_id=index_id, row=row, actor=actor.model_dump(), expires=int(time.time()) + TTL)
    data = base64.urlsafe_b64encode(json.dumps(payload, separators=(',', ':')).encode()).decode().rstrip('=')
    signature = hmac.new(SECRET, data.encode(), hashlib.sha256).hexdigest()
    return data + '.' + signature, payload['expires']


def granted_actor(token, index_id, row):
    try:
        data, signature = token.split('.')
        expected = hmac.new(SECRET, data.encode(), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, expected):
            raise ValueError('Invalid signature')
        payload = json.loads(base64.urlsafe_b64decode(data + '=' * (-len(data) % 4)))
        if payload['expires'] <= time.time() or payload['index_id'] != index_id or payload['row'] != row:
            raise ValueError('Expired or mismatched grant')
        return ActorContext.model_validate(payload['actor'])
    except (ValueError, KeyError, TypeError, UnicodeError):
        raise HTTPException(403, 'PREVIEW_GRANT_INVALID') from None
