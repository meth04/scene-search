#!/usr/bin/env python3
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionEncoder(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        model_dim: int = 256,
        out_dim: int = 512,
        layers: int = 2,
        heads: int = 8,
        dropout: float = .1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, model_dim),
            nn.GELU(),
        )
        self.cls = nn.Parameter(torch.zeros(1, 1, model_dim))
        nn.init.normal_(self.cls, std=.02)
        block = nn.TransformerEncoderLayer(
            model_dim, heads, model_dim * 4, dropout,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block, layers, norm=nn.LayerNorm(model_dim)
        )
        self.out = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, out_dim),
        )

    def forward(self, x, mask):
        b = x.shape[0]
        z = self.in_proj(x)
        cls = self.cls.expand(b, -1, -1)
        z = torch.cat([cls, z], dim=1)
        pad = torch.cat([
            torch.zeros(b, 1, dtype=torch.bool, device=mask.device),
            ~mask.bool()
        ], dim=1)
        z = self.encoder(z, src_key_padding_mask=pad)
        return self.out(z[:, 0])


class Stage3AFusion(nn.Module):
    """Residual gated motion fusion that starts close to the frozen Stage-2B embedding."""
    def __init__(
        self,
        feature_dim: int,
        emb_dim: int = 512,
        model_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = .1,
        init_motion_scale: float = .005,
    ):
        super().__init__()
        self.motion = MotionEncoder(
            feature_dim, model_dim, emb_dim, layers, heads, dropout
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(emb_dim * 2),
            nn.Linear(emb_dim * 2, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, 1),
        )
        init_logit = torch.logit(torch.tensor(float(init_motion_scale))).item()
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, init_logit)
        self.motion_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, base_clip_emb, track_features, track_mask):
        base = F.normalize(base_clip_emb.float(), dim=-1)
        motion = self.motion(track_features.float(), track_mask)
        motion = self.motion_proj(motion)

        # Normalize motion so gate directly controls residual magnitude.
        motion_dir = F.normalize(motion, dim=-1)

        gate = torch.sigmoid(
            self.gate(torch.cat([base, motion_dir], dim=-1))
        )
        # If YOLO/ByteTrack found no usable track, Stage3A must fall back
        # exactly to Stage2B rather than injecting a learned "empty" motion.
        has_track = track_mask.any(dim=1, keepdim=True).float()
        gate = gate * has_track
        fused = F.normalize(base + gate * motion_dir, dim=-1)
        return {
            "fused_emb": fused,
            "motion_emb": motion_dir,
            "motion_gate": gate.squeeze(-1),
        }
