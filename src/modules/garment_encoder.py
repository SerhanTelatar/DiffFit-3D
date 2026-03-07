"""
Garment Encoder — Feature extraction from garment images.

Uses pretrained CLIP ViT-L/14 or DINOv2 as backbone to extract
multi-scale feature maps from garment images for conditioning.
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GarmentEncoder(nn.Module):
    """
    Extracts multi-scale features from garment images using a pretrained
    vision encoder (CLIP or DINOv2).

    Args:
        encoder_type: 'clip' or 'dinov2'.
        model_name: HuggingFace model identifier.
        output_dims: Output dimensions for each projection layer.
        freeze: Whether to freeze the backbone weights.
    """

    def __init__(self, encoder_type: str = "clip",
                 model_name: str = "openai/clip-vit-large-patch14",
                 output_dims: list[int] = [256, 512, 1024, 1024],
                 freeze: bool = True):
        super().__init__()
        self.encoder_type = encoder_type
        self.output_dims = output_dims

        if encoder_type == "clip":
            self._build_clip_encoder(model_name)
        elif encoder_type == "dinov2":
            self._build_dinov2_encoder(model_name)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # Multi-scale projection heads
        backbone_dim = self._get_backbone_dim()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Linear(dim, dim),
            )
            for dim in output_dims
        ])

        # Learnable scale tokens for multi-scale extraction
        self.scale_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, 1, backbone_dim) * 0.02)
            for _ in output_dims
        ])

        if freeze:
            self._freeze_backbone()

    def _build_clip_encoder(self, model_name: str):
        try:
            from transformers import CLIPVisionModel, CLIPImageProcessor
            self.backbone = CLIPVisionModel.from_pretrained(model_name)
            self.processor = CLIPImageProcessor.from_pretrained(model_name)
        except ImportError:
            # Fallback: simple vision encoder
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 768, 16, stride=16),
                nn.LayerNorm([768]),
            )
            self.processor = None

    def _build_dinov2_encoder(self, model_name: str):
        try:
            self.backbone = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 768, 14, stride=14),
                nn.LayerNorm([768]),
            )
        self.processor = None

    def _get_backbone_dim(self) -> int:
        if self.encoder_type == "clip":
            return getattr(self.backbone.config, "hidden_size", 768) if hasattr(self.backbone, "config") else 768
        return 768

    def _freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, garment_image: torch.Tensor) -> list[torch.Tensor]:
        """
        Extract multi-scale features from garment image.

        Args:
            garment_image: (B, 3, H, W) garment image tensor.

        Returns:
            List of feature tensors, one per scale.
        """
        if hasattr(self.backbone, "vision_model"):
            outputs = self.backbone(pixel_values=garment_image, output_hidden_states=True)
            hidden = outputs.last_hidden_state  # (B, N+1, D) including CLS
            features = hidden[:, 1:, :]  # Remove CLS token
        else:
            x = self.backbone(garment_image)
            if x.dim() == 4:
                b, c, h, w = x.shape
                features = x.reshape(b, c, -1).permute(0, 2, 1)
            else:
                features = x

        # Project to multi-scale features
        multi_scale = []
        for proj, scale_token in zip(self.projections, self.scale_tokens):
            scale_t = scale_token.expand(features.shape[0], -1, -1)
            attended = features + scale_t
            projected = proj(attended)
            multi_scale.append(projected)

        return multi_scale

    def encode_garment(self, garment_image: torch.Tensor) -> torch.Tensor:
        """Get a single concatenated feature representation."""
        features = self.forward(garment_image)
        return torch.cat(features, dim=1)
