"""Reusable saved Stage3A reconstruction and bounded CPU/GPU index export."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

from hybrid_v2.local_runtime import device_for, load_checkpoint
from .stage3a_model import Stage3AFusion
from .stage3a_motion import FEATURE_DIM


def load_stage3a(checkpoint, device="cpu"):
    ck = load_checkpoint(checkpoint, "driveretriever-stage3a-motion-v1")
    state, cfg = ck["model_state_dict"], ck.get("config", {})
    feature_dim = state["motion.in_proj.1.weight"].shape[1]
    emb_dim = state["motion.out.1.weight"].shape[0]
    if (feature_dim, emb_dim) != (FEATURE_DIM, 512):
        raise ValueError(f"Unsupported Stage3A geometry: {(feature_dim, emb_dim)}")
    model = Stage3AFusion(
        feature_dim, emb_dim, int(cfg.get("model_dim", 256)),
        int(cfg.get("layers", 2)), int(cfg.get("heads", 8)),
        float(cfg.get("dropout", .1)), float(cfg.get("init_motion_scale", .005)),
    )
    model.load_state_dict(state, strict=True)
    return model.to(device_for(device)).eval()


@torch.inference_mode()
def export_fused(base, tracks, masks, checkpoint, output, device="cpu", batch_size=64):
    if batch_size <= 0 or base.ndim != 2 or base.shape[1] != 512:
        raise ValueError("Expected base (N,512) and positive batch size")
    if tracks.shape != (len(base), 24, FEATURE_DIM) or masks.shape != (len(base), 24):
        raise ValueError("Stage3A tracks/masks must be aligned (N,24,29)/(N,24)")
    if masks.dtype != np.bool_:
        raise ValueError("Stage3A track masks must be boolean")
    model = load_stage3a(checkpoint, device)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name + ".tmp")
    result = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float32, shape=base.shape)
    for s in range(0, len(base), batch_size):
        b = torch.tensor(np.asarray(base[s:s+batch_size]), device=device, dtype=torch.float32)
        t = torch.tensor(np.asarray(tracks[s:s+batch_size]), device=device, dtype=torch.float32)
        m = torch.tensor(np.asarray(masks[s:s+batch_size]), device=device, dtype=torch.bool)
        if not torch.isfinite(b).all() or not torch.isfinite(t).all():
            raise ValueError("Non-finite Stage3A input")
        if (torch.linalg.vector_norm(b, dim=-1) == 0).any():
            raise ValueError("Zero base embedding cannot be fused")
        out = model(b, t, m)
        expected = F.normalize(b, dim=-1)
        if not torch.allclose(out["fused_emb"][~m.any(1)], expected[~m.any(1)], atol=1e-6):
            raise ValueError("Stage3A no-track fallback differs from normalized base")
        result[s:s+len(b)] = out["fused_emb"].cpu().numpy()
    result.flush()
    del result, model
    tmp.replace(output)
    return output
