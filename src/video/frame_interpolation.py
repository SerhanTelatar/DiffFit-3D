"""
Frame Interpolation — Intermediate frame generation.

Generates intermediate frames for smooth video output using learned
optical flow and synthesis networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameInterpolator(nn.Module):
    """
    Generates intermediate frames between two input frames.

    Uses a flow estimation + synthesis approach for high-quality interpolation.

    Args:
        channels: Feature channels for the flow network.
    """

    def __init__(self, channels: int = 64):
        super().__init__()
        # Bidirectional flow estimator
        self.flow_net = FlowEstimator(in_channels=6, channels=channels)
        # Synthesis network
        self.synthesis = SynthesisNet(in_channels=9 + 4, channels=channels)  # 2 frames + warped + flows

    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor,
                t: float = 0.5) -> torch.Tensor:
        """
        Interpolate a frame at time t between frame0 and frame1.

        Args:
            frame0: (B, 3, H, W) first frame.
            frame1: (B, 3, H, W) second frame.
            t: Interpolation time (0=frame0, 1=frame1).

        Returns:
            Interpolated frame (B, 3, H, W).
        """
        # Estimate bidirectional flow
        flow_01, flow_10 = self.flow_net(frame0, frame1)

        # Scale flows for intermediate time
        flow_t0 = -(1 - t) * t * flow_01 + t * t * flow_10
        flow_t1 = (1 - t) * (1 - t) * flow_01 - t * (1 - t) * flow_10

        # Warp frames to time t
        warped_0 = self._warp(frame0, flow_t0)
        warped_1 = self._warp(frame1, flow_t1)

        # Synthesize output
        synth_input = torch.cat([
            frame0, frame1, warped_0,
            flow_t0, flow_t1,
        ], dim=1)
        output = self.synthesis(synth_input)

        return output

    def interpolate_sequence(self, frames: list[torch.Tensor],
                              factor: int = 2) -> list[torch.Tensor]:
        """Interpolate between all consecutive frame pairs."""
        result = [frames[0]]
        for i in range(len(frames) - 1):
            for j in range(1, factor):
                t = j / factor
                interp = self.forward(frames[i], frames[i + 1], t)
                result.append(interp)
            result.append(frames[i + 1])
        return result

    def _warp(self, image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Backward warp image using optical flow."""
        b, c, h, w = image.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=image.device),
            torch.linspace(-1, 1, w, device=image.device), indexing="ij",
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        flow_grid = grid + flow.permute(0, 2, 3, 1) * 2 / torch.tensor([w, h], device=image.device).float()
        return F.grid_sample(image, flow_grid, mode="bilinear", padding_mode="border", align_corners=True)


class FlowEstimator(nn.Module):
    """Bidirectional optical flow estimation."""

    def __init__(self, in_channels: int = 6, channels: int = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, 7, stride=2, padding=3), nn.ReLU(),
            nn.Conv2d(channels, channels * 2, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(channels * 2, channels * 4, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(channels * 4, channels * 4, 3, padding=1), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(channels * 4, channels * 2, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(channels, 4, 4, stride=2, padding=1),  # 4 = 2 flows × 2 dims
        )

    def forward(self, frame0: torch.Tensor, frame1: torch.Tensor):
        x = torch.cat([frame0, frame1], dim=1)
        features = self.encoder(x)
        flows = self.decoder(features)
        # Resize to input resolution
        flows = F.interpolate(flows, size=frame0.shape[-2:], mode="bilinear", align_corners=False)
        flow_01, flow_10 = flows[:, :2], flows[:, 2:]
        return flow_01, flow_10


class SynthesisNet(nn.Module):
    """Frame synthesis network."""

    def __init__(self, in_channels: int = 13, channels: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(channels, 3, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
