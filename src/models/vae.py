"""
VAE Wrapper for Latent Diffusion.

Wraps the Stable Diffusion VAE for encoding images to latent space and
decoding back to pixel space. Supports loading pretrained weights and
optional fine-tuning.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEWrapper(nn.Module):
    """
    Wrapper around the Stable Diffusion VAE (Variational Autoencoder).

    Handles encoding images to the latent space used by the diffusion model
    and decoding latent representations back to pixel space.

    Args:
        in_channels: Number of input image channels (3 for RGB).
        out_channels: Number of output image channels.
        latent_channels: Number of latent space channels.
        block_out_channels: Channel counts for encoder/decoder blocks.
        layers_per_block: Number of ResNet layers per block.
        norm_num_groups: Number of groups for GroupNorm.
        scaling_factor: Latent space scaling factor (SD 1.5 uses 0.18215).
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        scaling_factor: float = 0.18215,
    ):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.latent_channels = latent_channels

        # Encoder
        self.encoder = VAEEncoder(
            in_channels=in_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # Decoder
        self.decoder = VAEDecoder(
            out_channels=out_channels,
            latent_channels=latent_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
        )

        # Quant and post-quant conv (matching SD VAE)
        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

    def encode(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Encode image to latent distribution parameters.

        Args:
            x: (B, 3, H, W) input image, normalized to [-1, 1].

        Returns:
            Dict with 'latent', 'mean', 'logvar' tensors.
        """
        h = self.encoder(x)
        moments = self.quant_conv(h)
        mean, logvar = moments.chunk(2, dim=1)
        logvar = torch.clamp(logvar, -30.0, 20.0)

        # Reparameterize
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        latent = mean + std * eps

        # Scale
        latent = latent * self.scaling_factor

        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
        }

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to image.

        Args:
            z: (B, C_latent, H/8, W/8) latent representation.

        Returns:
            (B, 3, H, W) decoded image.
        """
        z = z / self.scaling_factor
        z = self.post_quant_conv(z)
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Encode and decode (for training the VAE)."""
        enc = self.encode(x)
        dec = self.decode(enc["latent"])
        return {
            "reconstruction": dec,
            "latent": enc["latent"],
            "mean": enc["mean"],
            "logvar": enc["logvar"],
        }

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "VAEWrapper":
        """Load pretrained VAE weights."""
        model = cls(**kwargs)
        try:
            from safetensors.torch import load_file
            state_dict = load_file(path)
        except Exception:
            state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        return model


class VAEResBlock(nn.Module):
    """Residual block for VAE encoder/decoder."""

    def __init__(self, in_ch: int, out_ch: int, norm_groups: int = 32):
        super().__init__()
        self.norm1 = nn.GroupNorm(norm_groups, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(norm_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return h + self.skip(x)


class VAEAttentionBlock(nn.Module):
    """Self-attention for the VAE bottleneck."""

    def __init__(self, channels: int, norm_groups: int = 32):
        super().__init__()
        self.norm = nn.GroupNorm(norm_groups, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        residual = x
        x = self.norm(x)

        q = self.q(x).reshape(b, c, -1)
        k = self.k(x).reshape(b, c, -1)
        v = self.v(x).reshape(b, c, -1)

        attn = torch.bmm(q.transpose(1, 2), k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)

        out = torch.bmm(v, attn.transpose(1, 2)).reshape(b, c, h, w)
        return residual + self.proj_out(out)


class VAEEncoder(nn.Module):
    """VAE Encoder: Image → Latent Distribution."""

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], 3, padding=1)

        blocks = []
        in_ch = block_out_channels[0]
        for i, out_ch in enumerate(block_out_channels):
            for _ in range(layers_per_block):
                blocks.append(VAEResBlock(in_ch, out_ch, norm_num_groups))
                in_ch = out_ch
            if i < len(block_out_channels) - 1:
                blocks.append(nn.Conv2d(in_ch, in_ch, 3, stride=2, padding=1))

        self.blocks = nn.Sequential(*blocks)

        # Mid
        self.mid = nn.Sequential(
            VAEResBlock(in_ch, in_ch, norm_num_groups),
            VAEAttentionBlock(in_ch, norm_num_groups),
            VAEResBlock(in_ch, in_ch, norm_num_groups),
        )

        # Out
        self.norm_out = nn.GroupNorm(norm_num_groups, in_ch)
        self.conv_out = nn.Conv2d(in_ch, 2 * latent_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.blocks(x)
        x = self.mid(x)
        x = self.norm_out(x)
        x = F.silu(x)
        x = self.conv_out(x)
        return x


class VAEDecoder(nn.Module):
    """VAE Decoder: Latent → Image."""

    def __init__(
        self,
        out_channels: int = 3,
        latent_channels: int = 4,
        block_out_channels: tuple[int, ...] = (128, 256, 512, 512),
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
    ):
        super().__init__()
        reversed_channels = list(reversed(block_out_channels))
        in_ch = reversed_channels[0]

        self.conv_in = nn.Conv2d(latent_channels, in_ch, 3, padding=1)

        # Mid
        self.mid = nn.Sequential(
            VAEResBlock(in_ch, in_ch, norm_num_groups),
            VAEAttentionBlock(in_ch, norm_num_groups),
            VAEResBlock(in_ch, in_ch, norm_num_groups),
        )

        blocks = []
        for i, out_ch in enumerate(reversed_channels):
            for j in range(layers_per_block + 1):
                blocks.append(VAEResBlock(in_ch, out_ch, norm_num_groups))
                in_ch = out_ch
            if i < len(reversed_channels) - 1:
                blocks.append(nn.Upsample(scale_factor=2, mode="nearest"))
                blocks.append(nn.Conv2d(in_ch, in_ch, 3, padding=1))

        self.blocks = nn.Sequential(*blocks)

        self.norm_out = nn.GroupNorm(norm_num_groups, in_ch)
        self.conv_out = nn.Conv2d(in_ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.conv_in(z)
        z = self.mid(z)
        z = self.blocks(z)
        z = self.norm_out(z)
        z = F.silu(z)
        z = self.conv_out(z)
        return z
