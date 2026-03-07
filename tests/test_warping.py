"""Tests for TPS and flow-based warping modules."""

import pytest
import torch
from src.modules.warping import TPSWarping, FlowWarping


class TestTPSWarping:
    def test_output_shape(self):
        tps = TPSWarping(num_control_points=5, input_dim=64)
        source = torch.randn(2, 64, 16, 16)
        condition = torch.randn(2, 64, 16, 16)
        result = tps(source, condition)
        assert result["warped"].shape == source.shape
        assert result["grid"].shape == (2, 16, 16, 2)

    def test_identity_approx(self):
        """With zero offsets, output should approximate input."""
        tps = TPSWarping(num_control_points=3, input_dim=32)
        # Zero-init the predictor to get near-zero offsets
        for p in tps.cp_predictor.parameters():
            torch.nn.init.zeros_(p)
        source = torch.randn(1, 32, 8, 8)
        condition = torch.zeros(1, 32, 8, 8)
        result = tps(source, condition)
        assert result["warped"].shape == source.shape

    def test_batch_independence(self):
        tps = TPSWarping(num_control_points=4, input_dim=32)
        source = torch.randn(4, 32, 8, 8)
        condition = torch.randn(4, 32, 8, 8)
        result = tps(source, condition)
        assert result["warped"].shape[0] == 4


class TestFlowWarping:
    def test_output_shape(self):
        flow = FlowWarping(in_channels=64, mid_channels=32)
        source = torch.randn(2, 32, 16, 16)
        target = torch.randn(2, 32, 16, 16)
        result = flow(source, target)
        assert result["warped"].shape == source.shape
        assert result["flow"].shape == (2, 2, 16, 16)

    def test_flow_field_range(self):
        flow = FlowWarping(in_channels=64, mid_channels=32)
        source = torch.randn(1, 32, 8, 8)
        target = torch.randn(1, 32, 8, 8)
        result = flow(source, target)
        # Flow should be finite
        assert torch.isfinite(result["flow"]).all()
