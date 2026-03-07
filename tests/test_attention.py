"""Tests for cross-attention and spatial transformer blocks."""

import pytest
import torch
from src.models.attention.cross_attention import GarmentPersonCrossAttention, MultiScaleCrossAttention
from src.models.attention.self_attention import MultiHeadSelfAttention, WindowedSelfAttention
from src.models.attention.spatial_attn import SpatialTransformerBlock, BasicTransformerLayer


class TestCrossAttention:
    def test_output_shape(self):
        attn = GarmentPersonCrossAttention(query_dim=256, context_dim=512, heads=4)
        person = torch.randn(2, 64, 256)
        garment = torch.randn(2, 32, 512)
        out = attn(person, garment)
        assert out.shape == person.shape

    def test_fusion_types(self):
        for fusion in ["add", "concat_proj", "gated"]:
            attn = GarmentPersonCrossAttention(query_dim=64, context_dim=64, heads=2, fusion_type=fusion)
            p = torch.randn(1, 16, 64)
            g = torch.randn(1, 8, 64)
            out = attn(p, g)
            assert out.shape == p.shape

    def test_attention_maps(self):
        attn = GarmentPersonCrossAttention(query_dim=64, context_dim=64, heads=4)
        p = torch.randn(2, 16, 64)
        g = torch.randn(2, 8, 64)
        maps = attn.get_attention_maps(p, g)
        assert maps.shape == (2, 4, 16, 8)
        assert torch.allclose(maps.sum(dim=-1), torch.ones(2, 4, 16), atol=1e-5)


class TestMultiScaleCrossAttention:
    def test_output_shape(self):
        attn = MultiScaleCrossAttention(query_dim=128, context_dims=[64, 128, 256], heads=4)
        person = torch.randn(2, 32, 128)
        garment_multi = [
            torch.randn(2, 32, 64),
            torch.randn(2, 16, 128),
            torch.randn(2, 8, 256),
        ]
        out = attn(person, garment_multi)
        assert out.shape == person.shape


class TestSelfAttention:
    def test_output_shape(self):
        attn = MultiHeadSelfAttention(dim=128, heads=4)
        x = torch.randn(2, 64, 128)
        out = attn(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        attn = MultiHeadSelfAttention(dim=32, heads=2)
        x = torch.randn(1, 8, 32)
        out = attn(x)
        # Should be different from input (not just identity)
        assert not torch.allclose(out, x)


class TestSpatialTransformerBlock:
    def test_conv_format(self):
        """Test that it accepts (B, C, H, W) and returns same shape."""
        block = SpatialTransformerBlock(dim=64, heads=4, dim_head=16, context_dim=32)
        x = torch.randn(2, 64, 8, 8)
        context = torch.randn(2, 16, 32)
        out = block(x, context)
        assert out.shape == x.shape

    def test_without_context(self):
        block = SpatialTransformerBlock(dim=32, heads=2, dim_head=16)
        x = torch.randn(1, 32, 4, 4)
        out = block(x)
        assert out.shape == x.shape
