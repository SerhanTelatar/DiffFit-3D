"""
Temporal Attention — Inter-frame attention for video consistency.

Ensures garment patterns remain consistent across video frames by
attending to features from neighboring frames.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TemporalAttention(nn.Module):
    """
    Temporal attention layer for maintaining consistency across video frames.

    Operates along the temporal dimension, allowing each frame's features
    to attend to features from all other frames in the sequence.

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        temporal_window: Number of frames to attend to.
    """

    def __init__(self, dim: int, heads: int = 8, dim_head: int = 64,
                 temporal_window: int = 16):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.temporal_window = temporal_window

        self.norm = nn.LayerNorm(dim)
        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(0.0))

        # Temporal position encoding
        self.pos_enc = nn.Parameter(torch.randn(1, temporal_window, dim) * 0.02)

        # Zero-init output for residual
        nn.init.zeros_(self.to_out[0].weight)
        nn.init.zeros_(self.to_out[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, H*W, C) video features [batch, frames, spatial, channels].

        Returns:
            Temporally-attended features with same shape.
        """
        b, t, n, c = x.shape
        residual = x

        # Reshape: treat each spatial position independently across time
        x = rearrange(x, "b t n c -> (b n) t c")

        # Add temporal position encoding
        if t <= self.temporal_window:
            x = x + self.pos_enc[:, :t, :]
        else:
            # Interpolate position encodings for longer sequences
            pos = F.interpolate(
                self.pos_enc.permute(0, 2, 1), size=t, mode="linear", align_corners=False
            ).permute(0, 2, 1)
            x = x + pos

        x = self.norm(x)

        # QKV
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = rearrange(q, "b t (h d) -> b h t d", h=self.heads)
        k = rearrange(k, "b t (h d) -> b h t d", h=self.heads)
        v = rearrange(v, "b t (h d) -> b h t d", h=self.heads)

        # Attention
        scale = self.dim_head ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h t d -> b t (h d)")
        out = self.to_out(out)

        # Reshape back
        out = rearrange(out, "(b n) t c -> b t n c", b=b, n=n)
        return residual + out


class TemporalConvBlock(nn.Module):
    """1D temporal convolution block for short-range temporal modeling."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, kernel_size, padding=kernel_size // 2),
        )
        nn.init.zeros_(self.conv[-1].weight)
        nn.init.zeros_(self.conv[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C, H, W) video features.
        """
        b, t, c, h, w = x.shape
        x_flat = rearrange(x, "b t c h w -> (b h w) c t")
        residual = x_flat
        x_flat = self.conv(x_flat)
        x_flat = x_flat + residual
        return rearrange(x_flat, "(b h w) c t -> b t c h w", b=b, h=h, w=w)
