
from __future__ import annotations
import torch
import torch.nn as nn

class Stage5HybridReranker(nn.Module):
    def __init__(self,feature_dim,hidden=96,dropout=.15,residual_scale=2.0):
        super().__init__()
        self.residual_scale=float(residual_scale)
        self.net=nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim,hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden,48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48,1),
        )
        # Starts as exact fallback.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def residual(self,x):
        return self.residual_scale*torch.tanh(self.net(x).squeeze(-1))

    def forward(self,x,base_score,alpha=1.0):
        r=self.residual(x)
        return base_score+float(alpha)*r,r
