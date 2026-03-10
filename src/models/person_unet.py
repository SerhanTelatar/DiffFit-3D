"""
Person UNet — Preserves body contour, pose, and identity.

The Person UNet processes the person image (in latent space) along with
conditioning signals (pose skeleton, body segmentation, DensePose UV maps)
to produce features that maintain the person's identity while preparing
the spatial structure for garment integration.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.models.attention.spatial_attn import SpatialTransformerBlock


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embeddings projected through an MLP."""

    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period, dtype=torch.float32))
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / half_dim
        )
        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)


class ResBlock(nn.Module):
    """Residual block with timestep conditioning and optional up/down sampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )
        self.dropout = nn.Dropout(dropout)

        if in_channels != out_channels:
            self.skip_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_conv = nn.Identity()

        self.updown = None
        if up:
            self.updown = nn.Upsample(scale_factor=2, mode="nearest")
        elif down:
            self.updown = nn.AvgPool2d(2)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        if self.updown is not None:
            x = self.updown(x)
            h = self.updown(h)
        h = self.conv1(h)

        # Add timestep embedding
        time_emb = self.time_proj(time_emb)[:, :, None, None]
        h = h + time_emb

        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.skip_conv(x)


class ControlNetConditioningBlock(nn.Module):
    """
    ControlNet-style conditioning encoder that processes auxiliary signals
    (pose, segmentation, DensePose) into multi-scale conditioning features.
    """

    def __init__(
        self,
        conditioning_channels: int = 5,
        embedding_channels: list[int] = [16, 32, 96, 256],
        out_channels: int = 320,
    ):
        super().__init__()
        layers = []
        in_ch = conditioning_channels
        for i, out_ch in enumerate(embedding_channels):
            layers.append(nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=2 if i > 0 else 1))
            layers.append(nn.SiLU())
            in_ch = out_ch
        layers.append(nn.Conv2d(in_ch, out_channels, 3, padding=1))
        # Zero initialization for residual connection
        nn.init.zeros_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)
        self.blocks = nn.Sequential(*layers)

    def forward(self, conditioning: torch.Tensor) -> torch.Tensor:
        return self.blocks(conditioning)


class PersonUNet(nn.Module):
    """
    Person branch UNet with ControlNet-style conditioning.

    Processes the person's latent representation alongside structural
    conditioning (pose, segmentation, DensePose UV) to produce features
    that maintain body identity and pose. Cross-attention layers allow
    the garment branch to inject garment features.

    Args:
        in_channels: Input channels (latent + conditioning concat).
        model_channels: Base channel count.
        out_channels: Output latent channels.
        channel_mult: Channel multipliers for each resolution stage.
        num_res_blocks: Number of ResBlocks per stage.
        attention_resolutions: Resolutions at which to apply attention.
        num_heads: Number of attention heads.
        context_dim: Cross-attention context dimension (garment features).
        controlnet_channels: Number of ControlNet conditioning channels.
        dropout: Dropout rate.
        use_checkpoint: Enable gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int = 9,
        model_channels: int = 320,
        out_channels: int = 4,
        channel_mult: tuple[int, ...] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (4, 2, 1),
        num_heads: int = 8,
        context_dim: int = 768,
        controlnet_channels: int = 5,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
        
    ):
        super().__init__()
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.use_checkpoint = use_checkpoint
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult

        time_emb_dim = model_channels * 4

        # Timestep embedding
        self.time_embed = TimestepEmbedding(model_channels)

        # ControlNet conditioning
        self.controlnet_cond = ControlNetConditioningBlock(
            conditioning_channels=controlnet_channels,
            out_channels=model_channels,
        )

        # Input convolution
        self.input_conv = nn.Conv2d(in_channels, model_channels, 3, padding=1)

        # Encoder (downsampling path)
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downsamples = nn.ModuleList()
        channels = [model_channels]
        ch = model_channels

        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult
            for _ in range(num_res_blocks):
                layers = [ResBlock(ch, out_ch, time_emb_dim, dropout=dropout)]
                ch = out_ch
                # Add attention at specified resolutions
                ds = 2 ** level
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformerBlock(
                            dim=ch,
                            heads=num_heads,
                            dim_head=ch // num_heads,
                            context_dim=context_dim,
                        )
                    )
                self.encoder_blocks.append(nn.ModuleList(layers))
                channels.append(ch)

            if level < len(channel_mult) - 1:
                self.encoder_downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
                channels.append(ch)

        # Middle block
        self.mid_block1 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)
        self.mid_attn = SpatialTransformerBlock(
            dim=ch, heads=num_heads, dim_head=ch // num_heads, context_dim=context_dim
        )
        self.mid_block2 = ResBlock(ch, ch, time_emb_dim, dropout=dropout)

        # Decoder (upsampling path)
        self.decoder_blocks = nn.ModuleList()
        self.decoder_upsamples = nn.ModuleList()

        for level, mult in list(enumerate(channel_mult))[::-1]:
            out_ch = model_channels * mult
            for i in range(num_res_blocks + 1):
                skip_ch = channels.pop()
                layers = [ResBlock(ch + skip_ch, out_ch, time_emb_dim, dropout=dropout)]
                ch = out_ch
                ds = 2 ** level
                if ds in attention_resolutions:
                    layers.append(
                        SpatialTransformerBlock(
                            dim=ch,
                            heads=num_heads,
                            dim_head=ch // num_heads,
                            context_dim=context_dim,
                        )
                    )
                self.decoder_blocks.append(nn.ModuleList(layers))

            if level > 0:
                self.decoder_upsamples.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(ch, ch, 3, padding=1),
                    )
                )

        # Output
        self.out_norm = nn.GroupNorm(32, ch)
        self.out_conv = nn.Conv2d(ch, out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        controlnet_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, H, W) noisy latent + conditioning channels.
            timesteps: (B,) diffusion timesteps.
            context: (B, M, D) garment features for cross-attention.
            controlnet_cond: (B, C_ctrl, H, W) ControlNet conditioning.

        Returns:
            Predicted noise of shape (B, C_out, H, W).
        """
        # Timestep embedding
        t_emb = self.time_embed(timesteps)

        # Input convolution
        h = self.input_conv(x)

        # Add ControlNet conditioning
        if controlnet_cond is not None:
            ctrl = self.controlnet_cond(controlnet_cond)
            # Ensure spatial dims match
            if ctrl.shape[-2:] != h.shape[-2:]:
                ctrl = F.interpolate(ctrl, size=h.shape[-2:], mode="bilinear", align_corners=False)
            h = h + ctrl

        # Encoder 
        skip_connections = [h]
        ds_idx = 0
        blocks_per_level = self.num_res_blocks  

        block_idx = 0
        for level, mult in enumerate(self.channel_mult):
            for _ in range(blocks_per_level):
                block_layers = self.encoder_blocks[block_idx]
                for layer in block_layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, SpatialTransformerBlock):
                        h = layer(h, context=context)
                skip_connections.append(h)
                block_idx += 1

            # Downsample — her seviyenin HEMEN sonunda
            if level < len(self.channel_mult) - 1:
                h = self.encoder_downsamples[ds_idx](h)
                skip_connections.append(h)
                ds_idx += 1

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h, context=context)
        h = self.mid_block2(h, t_emb)

        # Decoder
        up_idx = 0
        num_levels = len(self.channel_mult)
        blocks_per_level_dec = self.num_res_blocks + 1
        for level_idx, (level, mult) in enumerate(list(enumerate(self.channel_mult))[::-1]):
            for j in range(blocks_per_level_dec):
                dec_block_idx = level_idx * blocks_per_level_dec + j
                block_layers = self.decoder_blocks[dec_block_idx]
                skip = skip_connections.pop()
                h = torch.cat([h, skip], dim=1)
                for layer in block_layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, SpatialTransformerBlock):
                        h = layer(h, context=context)

            # Upsample — her decoder seviyesinin sonunda
            if level > 0:
                h = self.decoder_upsamples[up_idx](h)
                up_idx += 1

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return h
