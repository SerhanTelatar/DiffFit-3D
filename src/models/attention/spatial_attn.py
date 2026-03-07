"""
Spatial Transformer Block.

Combines self-attention, cross-attention, and feed-forward layers into
a complete transformer block that operates on spatial feature maps.
Used within the UNet encoder/decoder stages.
"""

from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange

from src.models.attention.self_attention import MultiHeadSelfAttention
from src.models.attention.cross_attention import GarmentPersonCrossAttention


class FeedForward(nn.Module):
    """GEGLU feed-forward network."""

    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.0):
        super().__init__()
        inner_dim = dim * mult
        # GEGLU: split into two halves, one for gating
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, inner_dim * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GEGLU(nn.Module):
    """Gated GELU activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, gate = x.chunk(2, dim=-1)
        return x * torch.nn.functional.gelu(gate)


class SpatialTransformerBlock(nn.Module):
    """
    A complete spatial transformer block with:
    1. Self-attention for spatial feature refinement
    2. Cross-attention for garment ↔ person fusion
    3. Feed-forward network

    This is the fundamental building block used in both the Person UNet
    and Garment UNet at each resolution stage.

    Args:
        dim: Feature dimension (number of channels).
        heads: Number of attention heads.
        dim_head: Dimension per head.
        context_dim: Dimension of the cross-attention context (garment features).
        depth: Number of transformer layers in this block.
        dropout: Dropout rate.
        ff_mult: Feed-forward hidden dim multiplier.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        context_dim: Optional[int] = None,
        depth: int = 1,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth

        # Input projection: conv channels -> transformer dim
        self.norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.proj_in = nn.Linear(dim, dim)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            self.transformer_blocks.append(
                BasicTransformerLayer(
                    dim=dim,
                    heads=heads,
                    dim_head=dim_head,
                    context_dim=context_dim,
                    dropout=dropout,
                    ff_mult=ff_mult,
                )
            )

        # Output projection
        self.proj_out = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) spatial feature map from UNet conv layers.
            context: (B, M, D_context) cross-attention context (garment features).

        Returns:
            Transformed feature map of shape (B, C, H, W).
        """
        b, c, h, w = x.shape
        residual = x

        # Normalize and flatten spatial dims
        x = self.norm(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.proj_in(x)

        # Apply transformer layers
        for block in self.transformer_blocks:
            x = block(x, context=context)

        # Project back and reshape
        x = self.proj_out(x)
        x = rearrange(x, "b (h w) c -> b c h w", h=h, w=w)

        return x + residual


class BasicTransformerLayer(nn.Module):
    """
    Single transformer layer: Self-Attn → Cross-Attn → FF.

    Args:
        dim: Feature dimension.
        heads: Number of attention heads.
        dim_head: Dimension per head.
        context_dim: Cross-attention context dimension.
        dropout: Dropout rate.
        ff_mult: Feed-forward multiplier.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        context_dim: Optional[int] = None,
        dropout: float = 0.0,
        ff_mult: int = 4,
    ):
        super().__init__()
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout,
        )

        # Cross-attention (optional, only if context is provided)
        self.cross_attn = None
        if context_dim is not None:
            self.cross_attn = GarmentPersonCrossAttention(
                query_dim=dim,
                context_dim=context_dim,
                heads=heads,
                dim_head=dim_head,
                dropout=dropout,
                fusion_type="add",
            )

        # Feed-forward
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, N, D) tokens.
            context: (B, M, D_ctx) optional cross-attention context.
        """
        # Self-attention
        x = self.self_attn(x)

        # Cross-attention
        if self.cross_attn is not None and context is not None:
            x = self.cross_attn(x, context)

        # Feed-forward
        x = x + self.ff(x)

        return x
