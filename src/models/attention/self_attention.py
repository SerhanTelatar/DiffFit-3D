"""
Self-Attention Module for Feature Refinement.

Standard multi-head self-attention used within both the Person UNet and
Garment UNet branches for spatial feature refinement.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with optional relative position encoding.

    Args:
        dim: Input feature dimension.
        heads: Number of attention heads.
        dim_head: Dimension per attention head.
        dropout: Dropout rate.
        use_relative_position: Whether to use relative position bias.
        max_positions: Maximum spatial positions for relative encoding.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_relative_position: bool = False,
        max_positions: int = 64,
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.use_relative_position = use_relative_position

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, self.inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, dim),
            nn.Dropout(dropout),
        )

        if use_relative_position:
            self.relative_position_bias = nn.Embedding(
                (2 * max_positions - 1) ** 2, heads
            )
            # Build relative position index
            coords = torch.arange(max_positions)
            grid = torch.stack(
                torch.meshgrid(coords, coords, indexing="ij")
            ).reshape(2, -1)
            relative_coords = grid[:, :, None] - grid[:, None, :]
            relative_coords += max_positions - 1
            relative_coords = relative_coords[0] * (2 * max_positions - 1) + relative_coords[1]
            self.register_buffer("relative_position_index", relative_coords)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) input feature tokens.
            attention_mask: Optional (B, N, N) attention mask.

        Returns:
            Output features of shape (B, N, D).
        """
        residual = x
        x = self.norm(x)

        # Compute Q, K, V in one projection
        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape to multi-head
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.heads)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Add relative position bias if enabled
        if self.use_relative_position:
            n = x.shape[1]
            rel_pos_idx = self.relative_position_index[:n, :n].reshape(-1)
            rel_pos_bias = self.relative_position_bias(rel_pos_idx)
            rel_pos_bias = rel_pos_bias.reshape(n, n, -1).permute(2, 0, 1)
            attn = attn + rel_pos_bias.unsqueeze(0)

        if attention_mask is not None:
            attn = attn.masked_fill(~attention_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)

        # Attend
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        out = self.to_out(out)
        return residual + out


class WindowedSelfAttention(nn.Module):
    """
    Windowed self-attention for efficient processing of high-resolution
    feature maps. Splits input into non-overlapping windows and applies
    self-attention within each window.

    Args:
        dim: Feature dimension.
        window_size: Size of the attention window.
        heads: Number of attention heads.
        dim_head: Dimension per head.
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 8,
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.window_size = window_size
        self.attn = MultiHeadSelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            use_relative_position=True,
            max_positions=window_size,
        )

    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """
        Args:
            x: (B, H*W, D) flattened spatial features.
            h: Spatial height.
            w: Spatial width.

        Returns:
            Output features of shape (B, H*W, D).
        """
        b, _, d = x.shape
        ws = self.window_size

        # Reshape to spatial
        x = x.reshape(b, h, w, d)

        # Pad if needed
        pad_h = (ws - h % ws) % ws
        pad_w = (ws - w % ws) % ws
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))

        hp, wp = x.shape[1], x.shape[2]

        # Partition into windows: (B * num_windows, ws*ws, D)
        x = x.reshape(b, hp // ws, ws, wp // ws, ws, d)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws * ws, d)

        # Apply attention within windows
        x = self.attn(x)

        # Un-partition
        num_h, num_w = hp // ws, wp // ws
        x = x.reshape(b, num_h, num_w, ws, ws, d)
        x = x.permute(0, 1, 3, 2, 4, 5).reshape(b, hp, wp, d)

        # Remove padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :h, :w, :]

        return x.reshape(b, h * w, d)
