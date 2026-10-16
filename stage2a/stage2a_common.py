#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_NAME = "google/siglip2-base-patch16-224"
DEFAULT_DATA_ROOT = "/home/thviet/Documents/Scence-Search/P-011/data/covla"
WEATHER_NAMES = ["sunny", "cloudy", "rainy", "unknown"]
ROAD_NAMES = ["narrow road", "wide road", "unknown"]
WEATHER_TO_ID = {x: i for i, x in enumerate(WEATHER_NAMES)}
ROAD_TO_ID = {x: i for i, x in enumerate(ROAD_NAMES)}
BINARY_KEYS = ["is_tunnel", "is_highway", "has_pedestrian", "has_carrier_car"]

EXTRA_PATTERNS = {
    "crosswalk": ("crosswalk", "pedestrian crossing"),
    "intersection": ("intersection",),
    "cyclist": ("cyclist", "bicycle"),
    "motorcycle": ("motorcycle", "motorbike"),
    "bus": (" bus ", "bus ahead", "bus in"),
    "truck": (" truck ", "truck ahead", "truck in"),
    "stop_sign": ("stop sign",),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_key_text(text: str) -> str:
    text = normalize_space(text.lower())
    text = re.sub(r"[^\w\s]", "", text)
    return normalize_space(text)


def token_set(text: str) -> set[str]:
    return set(normalize_key_text(text).split())


def token_jaccard(a: str, b: str) -> float:
    x, y = token_set(a), token_set(b)
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / max(1, len(x | y))


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk: int = 4 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def write_jsonl(records: Iterable[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def discover_ids(videos_dir: Path, captions_dir: Path) -> List[str]:
    vids = {p.stem for p in videos_dir.glob("*.mp4")}
    caps = {p.stem for p in captions_dir.glob("*.jsonl")}
    usable = sorted(vids & caps)
    if not usable:
        raise RuntimeError(f"No video/caption pairs: videos={videos_dir}, captions={captions_dir}")
    print(f"[dataset] videos={len(vids):,} captions={len(caps):,} usable={len(usable):,} orphan_captions={len(caps-vids):,}")
    return usable


def load_frozen_split(path: Path, usable_ids: Sequence[str]) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Stage-1 split is required and must NOT be regenerated for Stage 2A: {path}"
        )
    split = json.loads(path.read_text(encoding="utf-8"))
    a, b, c = map(set, (split["train"], split["val"], split["test"]))
    if a & b or a & c or b & c:
        raise RuntimeError("split.json contains leakage between train/val/test")
    if (a | b | c) != set(usable_ids):
        raise RuntimeError("split.json IDs do not exactly match current usable CoVLA IDs")
    return split


def load_caption_file(path: Path) -> List[Dict[str, Any]]:
    arr: List[Optional[Dict[str, Any]]] = [None] * 600
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if len(obj) != 1:
                raise ValueError(f"{path}:{ln}: expected one frame-key object")
            k, payload = next(iter(obj.items()))
            i = int(k)
            if i >= len(arr):
                arr.extend([None] * (i + 1 - len(arr)))
            arr[i] = payload
    if sum(x is not None for x in arr) < 590:
        raise RuntimeError(f"Too few caption entries in {path}")
    prev = None
    for i in range(len(arr)):
        if arr[i] is None:
            arr[i] = prev
        else:
            prev = arr[i]
    nxt = None
    for i in range(len(arr) - 1, -1, -1):
        if arr[i] is None:
            arr[i] = nxt
        else:
            nxt = arr[i]
    return [x for x in arr if x is not None]


def clean_plain(text: str) -> str:
    text = normalize_space(text)
    text = re.sub(r"\s*There is no traffic light in the scene\.?", "", text, flags=re.I)
    text = re.split(r"\bCaution\b", text, maxsplit=1, flags=re.I)[0]
    text = normalize_space(text)
    if text and text[-1] not in ".!?":
        text += "."
    return text


def affirmative(text: str, phrase: str) -> bool:
    lower = " " + normalize_space(text.lower()) + " "
    start = 0
    while True:
        idx = lower.find(phrase.lower(), start)
        if idx < 0:
            return False
        local = lower[max(0, idx - 28): idx + len(phrase)]
        if not re.search(r"\b(no|without|not)\b.{0,18}" + re.escape(phrase.lower()), local):
            return True
        start = idx + max(1, len(phrase))


def mine_extras(rich_texts: Sequence[str]) -> List[str]:
    out = []
    for concept, patterns in EXTRA_PATTERNS.items():
        if any(affirmative(t, p) for t in rich_texts for p in patterns):
            out.append(concept)
    return out


def mode(values: Sequence[str], default: str) -> str:
    xs = [x for x in values if x]
    return Counter(xs).most_common(1)[0][0] if xs else default


def ratio_mode(values: Sequence[str], chosen: str) -> float:
    xs = [x for x in values if x]
    return float(sum(x == chosen for x in xs) / max(1, len(xs)))


def structured_variant(r: Dict[str, Any]) -> str:
    parts: List[str] = []
    if r["weather"] != "unknown":
        parts.append(f"{r['weather']} weather")
    if r["is_highway"]:
        parts.append("driving on a highway")
    elif r["road"] != "unknown":
        parts.append(f"driving on a {r['road']}")
    if r["is_tunnel"]:
        parts.append("inside or entering a tunnel")
    if r["has_pedestrian"]:
        parts.append("a pedestrian is present")
    if r["has_carrier_car"]:
        parts.append("a car carrier truck is present")
    if r.get("extra_concepts"):
        parts.append("scene contains " + ", ".join(x.replace("_", " ") for x in r["extra_concepts"]))
    return normalize_space("; ".join(parts)).strip(" ;") or "driving scene"


def controlled_variant(r: Dict[str, Any]) -> str:
    clauses = []
    if r["is_highway"]:
        clauses.append("The ego vehicle is driving on a highway")
    elif r["road"] != "unknown":
        clauses.append(f"The ego vehicle is driving on a {r['road']}")
    else:
        clauses.append("The ego vehicle is driving on a road")
    if r["weather"] != "unknown":
        clauses.append(f"in {r['weather']} weather")
    text = " ".join(clauses) + "."
    if r["has_pedestrian"]:
        text += " A pedestrian is visible in the scene."
    if r["is_tunnel"]:
        text += " The scene is in or near a tunnel."
    return normalize_space(text)


def make_text_variants(r: Dict[str, Any], max_variants: int = 4) -> List[str]:
    candidates = [
        r["text"],
        r["dominant_plain_caption"],
        structured_variant(r),
        controlled_variant(r),
    ]
    out, seen = [], set()
    for x in candidates:
        x = normalize_space(x)
        k = normalize_key_text(x)
        if x and k not in seen:
            seen.add(k)
            out.append(x)
        if len(out) >= max_variants:
            break
    return out or ["Driving scene."]


def aggregate_window(video_id: str, entries: Sequence[Dict[str, Any]], sf: int, ef: int, fps: float) -> Dict[str, Any]:
    w = list(entries[sf:ef])
    plains = [normalize_space(str(x.get("plain_caption", ""))) for x in w]
    rich = [normalize_space(str(x.get("rich_caption", ""))) for x in w]
    pc = Counter(x for x in plains if x)
    base = clean_plain(pc.most_common(1)[0][0] if pc else "Driving scene.")

    weather_vals = [str(x.get("weather", "unknown")).lower() for x in w]
    road_vals = [str(x.get("road", "unknown")).lower() for x in w]
    weather = mode(weather_vals, "unknown")
    road = mode(road_vals, "unknown")
    weather = weather if weather in WEATHER_TO_ID else "unknown"
    road = road if road in ROAD_TO_ID else "unknown"

    tunnel_rate = float(np.mean([bool(x.get("is_tunnel", False)) for x in w]))
    highway_rate = float(np.mean([bool(x.get("is_highway", False)) for x in w]))
    pedestrian_rate = float(np.mean([bool(x.get("has_pedestrian", False)) for x in w]))
    carrier_rate = float(np.mean([bool(x.get("has_carrier_car", False)) for x in w]))
    tunnel = tunnel_rate >= 0.20
    highway = highway_rate >= 0.50
    pedestrian = pedestrian_rate >= 0.05
    carrier = carrier_rate >= 0.05
    extras = mine_extras(rich)

    parts = [base] if base else []
    if weather != "unknown":
        parts.append(f"The weather is {weather}.")
    if road != "unknown":
        parts.append(f"The ego vehicle is on a {road}.")
    if tunnel:
        parts.append("The scene is inside or entering a tunnel.")
    if highway:
        parts.append("The ego vehicle is driving on a highway.")
    if pedestrian:
        parts.append("A pedestrian is present in the scene.")
    if carrier:
        parts.append("A car carrier truck is present.")
    if extras:
        parts.append("Visible scene concepts include " + ", ".join(x.replace("_", " ") for x in extras) + ".")
    text = normalize_space(" ".join(parts))

    signature = {
        "base": normalize_key_text(base),
        "weather": weather,
        "road": road,
        "tunnel": bool(tunnel),
        "highway": bool(highway),
        "pedestrian": bool(pedestrian),
        "carrier": bool(carrier),
        "extras": sorted(extras),
    }
    key = sha1_text(json.dumps(signature, sort_keys=True))[:16]
    r = {
        "clip_id": f"{video_id}_{sf/fps:.2f}_{ef/fps:.2f}",
        "video_id": video_id,
        "start_frame": sf,
        "end_frame": ef,
        "start_sec": sf / fps,
        "end_sec": ef / fps,
        "text": text,
        "semantic_key": key,
        "dominant_plain_caption": base,
        "weather": weather,
        "road": road,
        "is_tunnel": bool(tunnel),
        "is_highway": bool(highway),
        "has_pedestrian": bool(pedestrian),
        "has_carrier_car": bool(carrier),
        "extra_concepts": extras,
        "weather_confidence": ratio_mode(weather_vals, weather),
        "road_confidence": ratio_mode(road_vals, road),
        "tunnel_positive_rate": tunnel_rate,
        "highway_positive_rate": highway_rate,
        "pedestrian_positive_rate": pedestrian_rate,
        "carrier_positive_rate": carrier_rate,
    }
    r["text_variants"] = make_text_variants(r)
    return r


def build_manifest(video_ids: Sequence[str], captions_dir: Path, window_sec: float, stride_sec: float, fps: float) -> List[Dict[str, Any]]:
    wf, st = int(round(window_sec * fps)), int(round(stride_sec * fps))
    total = int(round(30.0 * fps))
    starts = list(range(0, total - wf + 1, st))
    out = []
    from tqdm.auto import tqdm
    for vid in tqdm(video_ids, desc="build manifest"):
        entries = load_caption_file(captions_dir / f"{vid}.jsonl")
        if len(entries) < total:
            raise RuntimeError(f"{vid}: only {len(entries)} caption frames")
        for s in starts:
            out.append(aggregate_window(vid, entries, s, s + wf, fps))
    return out


def prep_frame(rgb: np.ndarray, size: int) -> torch.Tensor:
    # EXACT Stage-1 preprocessing.
    x = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(x).permute(2, 0, 1).contiguous()


def decode_frames(video_path: Path, sf: int, ef: int, n: int, size: int) -> torch.Tensor:
    ids = np.round(np.linspace(sf, ef - 1, n)).astype(np.int64)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    xs, last = [], None
    try:
        for i in ids.tolist():
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, bgr = cap.read()
            if ok and bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                last = rgb
            elif last is not None:
                rgb = last
            else:
                raise RuntimeError(f"Cannot decode frame {i} in {video_path}")
            xs.append(prep_frame(rgb, size))
    finally:
        cap.release()
    return torch.stack(xs)


class AttnPool(nn.Module):
    # Exact Stage-1 module/key names.
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.score(x).squeeze(-1), dim=1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)


class Stage2AHead(nn.Module):
    """Trainable Stage-1 head, without the frozen backbone.

    State-dict keys intentionally match Stage-1 trainable_state_dict exactly.
    """
    def __init__(
        self,
        hidden_dim: int,
        num_frames: int = 12,
        emb_dim: int = 512,
        layers: int = 3,
        heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        d = hidden_dim
        self.num_frames = num_frames
        self.hidden_dim = hidden_dim
        self.embedding_dim = emb_dim
        self.temporal_pos = nn.Parameter(torch.zeros(1, num_frames, d))
        nn.init.trunc_normal_(self.temporal_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d, heads, d * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=layers, norm=nn.LayerNorm(d)
        )
        self.pool = AttnPool(d)
        self.clip_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, emb_dim))
        self.frame_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, emb_dim))
        self.text_proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, emb_dim))
        self.weather_head = nn.Linear(d, len(WEATHER_NAMES))
        self.road_head = nn.Linear(d, len(ROAD_NAMES))
        self.binary_head = nn.Linear(d, 4)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def encode_video(self, frame_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        if frame_features.ndim != 3:
            raise ValueError(f"Expected [B,T,D], got {tuple(frame_features.shape)}")
        b, t, d = frame_features.shape
        if t != self.num_frames or d != self.hidden_dim:
            raise ValueError(
                f"Expected T={self.num_frames},D={self.hidden_dim}; got {t},{d}"
            )
        seq = self.temporal(frame_features + self.temporal_pos[:, :t])
        hidden = self.pool(seq)
        clip = F.normalize(self.clip_proj(hidden), dim=-1)
        frame = F.normalize(self.frame_proj(frame_features), dim=-1)
        return {
            "clip_emb": clip,
            "frame_emb": frame,
            "clip_hidden": hidden,
            "weather_logits": self.weather_head(hidden),
            "road_logits": self.road_head(hidden),
            "binary_logits": self.binary_head(hidden),
            "logit_scale": self.logit_scale.exp().clamp(max=100.0),
        }

    def encode_text(self, text_features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.text_proj(text_features), dim=-1)

    def forward(self, frame_features: torch.Tensor, text_features: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encode_video(frame_features)
        shape = text_features.shape
        if text_features.ndim == 3:
            b, k, d = shape
            txt = self.encode_text(text_features.reshape(b * k, d)).reshape(b, k, -1)
        elif text_features.ndim == 2:
            txt = self.encode_text(text_features)
        else:
            raise ValueError(f"text_features must be [B,D] or [B,K,D], got {shape}")
        out["text_emb"] = txt
        return out


def stage1_trainable_state(ckpt: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    if "trainable_state_dict" in ckpt:
        return ckpt["trainable_state_dict"]
    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        return {k: v for k, v in sd.items() if not k.startswith("backbone.")}
    raise KeyError("Checkpoint has neither trainable_state_dict nor model_state_dict")


def infer_head_config_from_state(
    state: Dict[str, torch.Tensor],
    fallback_num_frames: int = 12,
    fallback_emb_dim: int = 512,
) -> Dict[str, int]:
    tp = state["temporal_pos"]
    num_frames = int(tp.shape[1])
    hidden_dim = int(tp.shape[2])
    emb_dim = int(state["clip_proj.1.weight"].shape[0])
    # count transformer layer ids
    layer_ids = set()
    for k in state:
        m = re.match(r"temporal\.layers\.(\d+)\.", k)
        if m:
            layer_ids.add(int(m.group(1)))
    layers = len(layer_ids) or 3
    return {
        "hidden_dim": hidden_dim,
        "num_frames": num_frames or fallback_num_frames,
        "emb_dim": emb_dim or fallback_emb_dim,
        "layers": layers,
    }


def load_stage1_head(
    checkpoint: Path,
    heads: int = 8,
    dropout: float = 0.1,
    map_location: str = "cpu",
) -> tuple[Stage2AHead, Dict[str, Any]]:
    ckpt = torch.load(checkpoint, map_location=map_location)
    state = stage1_trainable_state(ckpt)
    cfg = infer_head_config_from_state(state)
    model = Stage2AHead(
        cfg["hidden_dim"], cfg["num_frames"], cfg["emb_dim"],
        cfg["layers"], heads, dropout,
    )
    missing, unexpected = model.load_state_dict(state, strict=True)
    if missing or unexpected:
        raise RuntimeError(f"Strict Stage-1 head load failed missing={missing} unexpected={unexpected}")
    return model, ckpt


def pick_precision(req: str, device: torch.device) -> str:
    if device.type != "cuda":
        return "fp32"
    if req == "auto":
        return "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    if req == "bf16" and not torch.cuda.is_bf16_supported():
        return "fp16"
    return req


def amp_ctx(precision: str, device: torch.device):
    if device.type != "cuda" or precision == "fp32":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast("cuda", dtype=dtype)


def metadata_weights(records: Sequence[Dict[str, Any]], device: torch.device):
    def inv(names, key):
        c = Counter(r[key] for r in records)
        n = sum(c.values())
        a = np.array([n / max(1, len(names) * c.get(x, 0)) for x in names], np.float32)
        a = np.clip(a / max(a.mean(), 1e-6), 0.25, 8.0)
        return torch.tensor(a, device=device)

    ww = inv(WEATHER_NAMES, "weather")
    rw = inv(ROAD_NAMES, "road")
    pws = []
    for k in BINARY_KEYS:
        p = sum(bool(r[k]) for r in records)
        n = len(records) - p
        pws.append(min(max(n / max(1, p), 1.0), 20.0))
    return ww, rw, torch.tensor(pws, dtype=torch.float32, device=device)


def make_scheduler(opt, total_steps: int, warmup_ratio: float):
    from torch.optim.lr_scheduler import LambdaLR
    warm = max(1, int(total_steps * warmup_ratio))

    def fn(step):
        if step < warm:
            return float(step + 1) / warm
        p = (step - warm) / max(1, total_steps - warm)
        return 0.5 * (1 + math.cos(math.pi * p))

    return LambdaLR(opt, fn)


def cuda_mem() -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    g = 1024 ** 3
    return {
        "gpu/allocated_gb": torch.cuda.memory_allocated() / g,
        "gpu/reserved_gb": torch.cuda.memory_reserved() / g,
        "gpu/max_allocated_gb": torch.cuda.max_memory_allocated() / g,
    }
