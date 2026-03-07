"""
Motion Module — AnimateDiff-style motion modeling.

Learns human motion dynamics for generating temporally coherent video sequences.
"""

import torch
import torch.nn as nn
from einops import rearrange

from src.video.temporal_attention import TemporalAttention


class MotionModule(nn.Module):
    """
    AnimateDiff-style motion module for learning motion dynamics.

    Inserted into the UNet to enable video generation by modeling
    temporal dynamics between frames.

    Args:
        dim: Feature channel dimension.
        num_attention_heads: Number of attention heads.
        num_layers: Number of temporal transformer layers.
        temporal_window: Max number of frames.
    """

    def __init__(self, dim: int, num_attention_heads: int = 8,
                 num_layers: int = 2, temporal_window: int = 16):
        super().__init__()
        self.layers = nn.ModuleList([
            MotionTransformerBlock(
                dim=dim,
                heads=num_attention_heads,
                temporal_window=temporal_window,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.GroupNorm(32, dim)

    def forward(self, x: torch.Tensor, num_frames: int) -> torch.Tensor:
        """
        Args:
            x: (B*T, C, H, W) batched video features.
            num_frames: T, number of frames per video.

        Returns:
            Motion-modulated features with same shape.
        """
        bt, c, h, w = x.shape
        b = bt // num_frames
        t = num_frames

        # Reshape to (B, T, C, H, W)
        x = rearrange(x, "(b t) c h w -> b t c h w", b=b, t=t)

        for layer in self.layers:
            x = layer(x)

        # Reshape back
        x = rearrange(x, "b t c h w -> (b t) c h w")
        return x


class MotionTransformerBlock(nn.Module):
    """Single motion transformer block with temporal attention and FFN."""

    def __init__(self, dim: int, heads: int = 8, temporal_window: int = 16,
                 ff_mult: int = 4):
        super().__init__()
        self.temporal_attn = TemporalAttention(
            dim=dim, heads=heads, temporal_window=temporal_window,
        )
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )
        nn.init.zeros_(self.ff[-1].weight)
        nn.init.zeros_(self.ff[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video features.
        """
        b, t, c, h, w = x.shape

        # Temporal attention
        x_flat = rearrange(x, "b t c h w -> b t (h w) c")
        x_flat = self.temporal_attn(x_flat)

        # FFN
        x_flat = x_flat + self.ff(x_flat)

        return rearrange(x_flat, "b t (h w) c -> b t c h w", h=h, w=w)
