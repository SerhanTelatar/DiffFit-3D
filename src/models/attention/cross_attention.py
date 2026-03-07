"""
Cross-Attention Module for Garment ↔ Person Feature Fusion.

Implements the core attention mechanism that fuses garment features onto the
person representation, enabling the garment to be "warped" and integrated
into the person's body geometry through learned attention patterns.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class GarmentPersonCrossAttention(nn.Module):
    """
    Cross-attention between garment and person feature maps.

    This module allows the person UNet to attend to garment features,
    enabling texture transfer that respects body geometry. The garment
    features serve as key/value, while the person features serve as query.

    Args:
        query_dim: Dimension of the query (person) features.
        context_dim: Dimension of the context (garment) features.
            If None, defaults to query_dim.
        heads: Number of attention heads.
        dim_head: Dimension per attention head.
        dropout: Dropout rate for attention weights.
        fusion_type: How to fuse attended features ('add', 'concat_proj', 'gated').
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        fusion_type: str = "concat_proj",
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.scale = dim_head ** -0.5
        self.fusion_type = fusion_type

        context_dim = context_dim or query_dim

        # Query from person features
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=False)
        # Key and Value from garment features
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=False)

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim),
            nn.Dropout(dropout),
        )

        # Fusion mechanism
        if fusion_type == "concat_proj":
            self.fusion_proj = nn.Linear(query_dim * 2, query_dim)
        elif fusion_type == "gated":
            self.gate = nn.Sequential(
                nn.Linear(query_dim * 2, query_dim),
                nn.Sigmoid(),
            )

        # Layer norm for stable training
        self.norm_q = nn.LayerNorm(query_dim)
        self.norm_k = nn.LayerNorm(context_dim)

    def forward(
        self,
        person_features: torch.Tensor,
        garment_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply cross-attention from person to garment features.

        Args:
            person_features: (B, N, D_person) person feature tokens.
            garment_features: (B, M, D_garment) garment feature tokens.
            attention_mask: Optional (B, N, M) mask for attention weights.

        Returns:
            Fused features with shape (B, N, D_person).
        """
        residual = person_features

        # Normalize
        person_features = self.norm_q(person_features)
        garment_features = self.norm_k(garment_features)

        # Project to Q, K, V
        q = self.to_q(person_features)
        k = self.to_k(garment_features)
        v = self.to_v(garment_features)

        # Reshape for multi-head attention: (B, heads, seq_len, dim_head)
        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.heads)
        v = rearrange(v, "b m (h d) -> b h m d", h=self.heads)

        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attention_mask is not None:
            # Expand mask for heads: (B, 1, N, M)
            attention_mask = attention_mask.unsqueeze(1)
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)

        # Attend to values
        attended = torch.matmul(attn_weights, v)
        attended = rearrange(attended, "b h n d -> b n (h d)")

        # Project output
        attended = self.to_out(attended)

        # Fuse with residual
        if self.fusion_type == "add":
            output = residual + attended
        elif self.fusion_type == "concat_proj":
            concat = torch.cat([residual, attended], dim=-1)
            output = self.fusion_proj(concat)
        elif self.fusion_type == "gated":
            concat = torch.cat([residual, attended], dim=-1)
            gate_values = self.gate(concat)
            output = residual + gate_values * attended
        else:
            output = residual + attended

        return output

    def get_attention_maps(
        self,
        person_features: torch.Tensor,
        garment_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract attention maps for visualization.

        Returns:
            Attention weights of shape (B, heads, N, M).
        """
        person_features = self.norm_q(person_features)
        garment_features = self.norm_k(garment_features)

        q = self.to_q(person_features)
        k = self.to_k(garment_features)

        q = rearrange(q, "b n (h d) -> b h n d", h=self.heads)
        k = rearrange(k, "b m (h d) -> b h m d", h=self.heads)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)

        return attn_weights


class MultiScaleCrossAttention(nn.Module):
    """
    Multi-scale cross-attention that attends to garment features at
    multiple resolutions for preserving both fine details and global structure.

    Args:
        query_dim: Dimension of the query features.
        context_dims: List of dimensions for each scale of garment features.
        heads: Number of attention heads per scale.
        dim_head: Dimension per head.
    """

    def __init__(
        self,
        query_dim: int,
        context_dims: list[int],
        heads: int = 8,
        dim_head: int = 64,
    ):
        super().__init__()
        self.scale_attns = nn.ModuleList([
            GarmentPersonCrossAttention(
                query_dim=query_dim,
                context_dim=ctx_dim,
                heads=heads,
                dim_head=dim_head,
                fusion_type="add",
            )
            for ctx_dim in context_dims
        ])
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(len(context_dims)) / len(context_dims))

    def forward(
        self,
        person_features: torch.Tensor,
        multi_scale_garment_features: list[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            person_features: (B, N, D) person tokens.
            multi_scale_garment_features: List of (B, M_i, D_i) garment features.
        """
        weights = F.softmax(self.scale_weights, dim=0)
        output = person_features

        for w, attn, garment_feat in zip(
            weights, self.scale_attns, multi_scale_garment_features
        ):
            attended = attn(output, garment_feat)
            output = output + w * (attended - output)

        return output
