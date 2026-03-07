"""
Warping Module — Geometric garment deformation.

Implements Thin-Plate Spline (TPS) and flow-based warping for
geometrically deforming garments to match body shape.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class TPSWarping(nn.Module):
    """
    Thin-Plate Spline warping for garment deformation.

    Learns control point displacements to warp the garment image/features
    to align with the target body pose.

    Args:
        num_control_points: Number of TPS control points per axis.
        input_dim: Feature dimension for control point regression.
        regularization_weight: TPS smoothness regularization.
    """

    def __init__(self, num_control_points: int = 10, input_dim: int = 256,
                 regularization_weight: float = 0.01):
        super().__init__()
        self.num_cp = num_control_points
        self.reg_weight = regularization_weight
        num_points = num_control_points ** 2

        # Control point predictor
        self.cp_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(num_control_points),
            nn.Flatten(),
            nn.Linear(input_dim * num_points, 512),
            nn.ReLU(),
            nn.Linear(512, num_points * 2),  # (x, y) offsets
        )

        # Initialize grid of control points
        grid_x = torch.linspace(-1, 1, num_control_points)
        grid_y = torch.linspace(-1, 1, num_control_points)
        grid = torch.stack(torch.meshgrid(grid_x, grid_y, indexing="ij"), dim=-1)
        self.register_buffer("base_grid", grid.reshape(-1, 2))

    def forward(self, source: torch.Tensor, condition: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Warp source features using TPS transformation predicted from condition.

        Args:
            source: (B, C, H, W) source feature map (garment).
            condition: (B, C, H, W) condition feature map (body shape).

        Returns:
            Dict with 'warped' transformed features and 'grid' sampling grid.
        """
        b = source.shape[0]
        # Predict control point offsets
        offsets = self.cp_predictor(condition)
        offsets = offsets.reshape(b, -1, 2) * 0.1  # Scale offsets

        # Displaced control points
        target_points = self.base_grid.unsqueeze(0).expand(b, -1, -1) + offsets

        # Compute TPS grid
        grid = self._compute_tps_grid(
            self.base_grid.unsqueeze(0).expand(b, -1, -1),
            target_points, source.shape[-2], source.shape[-1],
        )

        # Apply warping
        warped = F.grid_sample(source, grid, mode="bilinear", padding_mode="border", align_corners=True)

        return {"warped": warped, "grid": grid}

    def _compute_tps_grid(self, source_pts: torch.Tensor, target_pts: torch.Tensor,
                          h: int, w: int) -> torch.Tensor:
        """Compute TPS sampling grid from source→target control points."""
        b = source_pts.shape[0]
        device = source_pts.device
        n = source_pts.shape[1]

        # Build TPS system matrix
        diff = source_pts.unsqueeze(2) - source_pts.unsqueeze(1)
        dist = torch.sum(diff ** 2, dim=-1)
        dist = torch.clamp(dist, min=1e-6)
        kernel = dist * torch.log(dist + 1e-6)

        ones = torch.ones(b, n, 1, device=device)
        p_matrix = torch.cat([ones, source_pts], dim=-1)

        top = torch.cat([kernel, p_matrix], dim=-1)
        zeros = torch.zeros(b, 3, 3, device=device)
        bottom = torch.cat([p_matrix.transpose(1, 2), zeros], dim=-1)
        system = torch.cat([top, bottom], dim=1)

        # Add regularization
        system = system + self.reg_weight * torch.eye(n + 3, device=device).unsqueeze(0)

        # RHS
        rhs = torch.cat([target_pts, torch.zeros(b, 3, 2, device=device)], dim=1)

        # Solve
        params = torch.linalg.solve(system, rhs)

        # Generate output grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=device),
            torch.linspace(-1, 1, w, device=device), indexing="ij",
        )
        grid_pts = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        grid_pts = grid_pts.unsqueeze(0).expand(b, -1, -1)

        # Evaluate TPS at grid points
        diff = grid_pts.unsqueeze(2) - source_pts.unsqueeze(1)
        dist = torch.sum(diff ** 2, dim=-1)
        dist = torch.clamp(dist, min=1e-6)
        kernel_vals = dist * torch.log(dist + 1e-6)

        ones_g = torch.ones(b, h * w, 1, device=device)
        p_grid = torch.cat([ones_g, grid_pts], dim=-1)
        features = torch.cat([kernel_vals, p_grid], dim=-1)

        result = torch.matmul(features, params)
        return result.reshape(b, h, w, 2)


class FlowWarping(nn.Module):
    """
    Optical flow-based warping using a flow prediction network.

    Args:
        in_channels: Combined input channels (source + target features).
        mid_channels: Intermediate channel count.
    """

    def __init__(self, in_channels: int = 512, mid_channels: int = 256):
        super().__init__()
        self.flow_net = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1), nn.ReLU(),
            nn.Conv2d(mid_channels // 2, 2, 3, padding=1),  # 2-channel flow (dx, dy)
        )

    def forward(self, source: torch.Tensor, target: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Predict flow and warp source to align with target.

        Args:
            source: (B, C, H, W) source features.
            target: (B, C, H, W) target features.

        Returns:
            Dict with 'warped' result and 'flow' field.
        """
        combined = torch.cat([source, target], dim=1)
        flow = self.flow_net(combined)

        b, _, h, w = source.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=source.device),
            torch.linspace(-1, 1, w, device=source.device), indexing="ij",
        )
        base_grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
        sampling_grid = base_grid + flow.permute(0, 2, 3, 1)

        warped = F.grid_sample(source, sampling_grid, mode="bilinear",
                               padding_mode="border", align_corners=True)

        return {"warped": warped, "flow": flow}
