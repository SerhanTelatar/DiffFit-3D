"""
Garment UNet — Extracts and warps garment features.

The Garment UNet processes the garment image in latent space, extracting
multi-scale features (texture, pattern, logo) that are warped to align
with the body geometry. Its intermediate features are consumed by the
Person UNet via cross-attention.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.person_unet import TimestepEmbedding, ResBlock
from src.models.attention.spatial_attn import SpatialTransformerBlock


class GarmentFeatureProjector(nn.Module):
    """
    Projects multi-scale garment encoder features into a unified token
    sequence for cross-attention with the Person UNet.
    """

    def __init__(
        self,
        feature_dims: list[int],
        output_dim: int = 768,
    ):
        super().__init__()
        self.projectors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, output_dim),
                nn.GELU(),
                nn.Linear(output_dim, output_dim),
            )
            for dim in feature_dims
        ])

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """
        Project and concatenate multi-scale features into token sequence.

        Args:
            features: List of (B, C_i, H_i, W_i) feature maps.

        Returns:
            (B, sum(H_i * W_i), output_dim) token sequence.
        """
        projected = []
        for proj, feat in zip(self.projectors, features):
            b, c, h, w = feat.shape
            feat = rearrange(feat, "b c h w -> b (h w) c")
            feat = proj(feat)
            projected.append(feat)
        return torch.cat(projected, dim=1)


class GarmentUNet(nn.Module):
    """
    Garment branch UNet for extracting and processing garment features.

    This network:
    1. Encodes the garment image into latent features at multiple scales
    2. Provides intermediate features to the Person UNet via cross-attention
    3. Can optionally apply TPS warping guided by body keypoints

    Args:
        in_channels: Input latent channels.
        model_channels: Base channel count.
        out_channels: Output channels (garment feature tokens).
        channel_mult: Channel multipliers per resolution stage.
        num_res_blocks: Number of ResBlocks per stage.
        attention_resolutions: Resolutions for self-attention.
        num_heads: Attention heads.
        context_dim: Context dim for optional text conditioning.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        in_channels: int = 4,
        model_channels: int = 320,
        out_channels: int = 4,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (4, 2, 1),
        num_heads: int = 8,
        context_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.model_channels = model_channels

        time_emb_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(model_channels)

        # Input
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        ch = model_channels
        feature_channels = []

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout=dropout)]
                ch = out_ch
                ds = 2 ** level
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformerBlock(
                            dim=ch, heads=num_heads,
                            dim_head=ch // num_heads,
                            context_dim=context_dim,
                        )
                    )
                self.encoder_blocks.append(nn.ModuleList(layers))

            feature_channels.append(ch)

            if level < len(channel_mult) - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )

        # Middle
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SpatialTransformerBlock(
            dim=ch, heads=num_heads, dim_head=ch // num_heads, context_dim=context_dim
        )
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)

        # Feature projector for cross-attention output
        self.feature_projector = GarmentFeatureProjector(
            feature_dims=feature_channels,
            output_dim=context_dim,
        )

        # Output projection (for garment latent reconstruction)
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        return_features: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) garment latent.
            timesteps: (B,) diffusion timesteps.
            context: (B, M, D) optional text/CLIP conditioning.
            return_features: Whether to return multi-scale features for
                cross-attention with the Person UNet.

        Returns:
            Dictionary with:
                - 'output': (B, C_out, H, W) reconstructed garment latent.
                - 'features': (B, N_tokens, D) multi-scale feature tokens
                    for cross-attention (if return_features=True).
                - 'feature_maps': List of (B, C_i, H_i, W_i) intermediate
                    feature maps.
        """
        t_emb = self.time_embed(timesteps)

        h = self.input_conv(x)

        # Collect multi-scale features
        multi_scale_features = []
        ds_idx = 0
        block_idx = 0

        for level, mult in enumerate(
            [1, 2, 4, 4]
        ):  # TODO: use channel_mult length dynamically
            for _ in range(2):  # num_res_blocks
                if block_idx < len(self.encoder_blocks):
                    for layer in self.encoder_blocks[block_idx]:
                        if isinstance(layer, ResBlock):
                            h = layer(h, t_emb)
                        elif isinstance(layer, SpatialTransformerBlock):
                            h = layer(h, context=context)
                    block_idx += 1

            multi_scale_features.append(h)

            if ds_idx < len(self.downsamples):
                h = self.downsamples[ds_idx](h)
                ds_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context=context)
        h = self.mid_block2(h, t_emb)

        # Output
        out = self.out_norm(h)
        out = F.silu(out)
        out = self.out_conv(out)

        result = {"output": out, "feature_maps": multi_scale_features}

        if return_features:
            result["features"] = self.feature_projector(multi_scale_features)

        return result

    def extract_features(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract garment feature tokens for cross-attention with Person UNet.

        Returns:
            (B, N_tokens, D) garment feature token sequence.
        """
        result = self.forward(x, timesteps, context, return_features=True)
        return result["features"]
