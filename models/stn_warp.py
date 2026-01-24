import torch
import torch.nn as nn
import torch.nn.functional as F

class TPSGridGen(nn.Module):
    def __init__(self, height, width, num_control_points):
        super(TPSGridGen, self).__init__()
        self.height = height
        self.width = width
        self.num_control_points = num_control_points

        self.grid_size = int(num_control_points ** 0.5)

        target_control_points = self._create_control_points()
        self.register_buffer("target_control_points", target_control_points)

        target_coordinate = self._create_coordinate_grid()
        self.register_buffer("target_coordinate", target_coordinate)

    def _create_control_points(self):
        axis = torch.linspace(-1, 1, 5)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        control_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return control_points

    def _create_coordinate_grid(self):
        y = torch.linspace(-1, 1, self.height)
        x = torch.linspace(-1, 1, self.width)
        grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
        coordinates = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
        return coordinates
        
    def _compute_radial_basis(self, p1, p2):
        diff = p1.unsqueeze(-2) - p2.unsqueeze(-3)
        diff_sq = (diff ** 2).sum(dim=-1)
        diff_sq = torch.clamp(diff_sq, min=1e-6)
        return diff_sq * torch.log(diff_sq) * 0.5

    def _precompute_tps_matrices(self):
        N = self.num_control_points # 25
        cp = self.target_control_points # (25, 2)

        K = self._compute_radial_basis(cp, cp) # (25, 25)
        
        ones = torch.ones(N, 1)
        P = torch.cat([ones, cp], dim=1) # (25, 3)

        zeros = torch.zeros(3,3)

        L_top = torch.cat([K, P], dim=1) # (N, N + 3)
        L_bottom = torch.cat([P.t(), zeros], dim=1) # (3, N + 3)

        L = torch.cat([L_top, L_bottom], dim=0) # (N + 3, N + 3)

        L_inv = torch.invers(L + torch.eye(N + 3) * 1e-6) # (N + 3, N + 3)
        self.register_buffer("L_inv", L_inv)

        U = self._compute_radial_basis(self.target_coordinate, cp) # (H*W, N)
        self.register_buffer("U", U)

        ones = torch.ones(self.height * self.width, 1)
        P_target = torch.cat([ones, self.target_coordinate], dim=1) # (H*W, 3)
        
        Phi = torch.cat([U, P_target], dim=1) # (H*W, N + 3)
        self.register_buffer("Phi", Phi)

    def forward(self, source_control_points):
        B = source_control_points.size[0]
        N = self.num_control_points # 25
        zeros = torch.zeros(B, 3, 2, device=source_control_points.device, dtype=source_control_points.dtype) # (B, 3, 2)

        Y = torch.cat([source_control_points, zeros], dim=1) # (B, N + 3, 2)

        L_inv_expanded = self.L_inv.unsqueeze(0).expand(B, -1, -1) # (B, N + 3, N + 3)
        W = torch.bmm(L_inv_expanded, Y) # (B, N + 3, 2)

        Phi_expanded = self.Phi.unsqueeze(0).expand(B, -1, -1) # (B, H*W, N + 3)
        grid = torch.bmm(Phi_expanded, W) # (B, H*W, 2)

        grid = grid.view(B, self.height, self.width, 2) # (B, H, W, 2)

        return grid
    
class ConvBlock(nn.Module):
    def __init__(self, in_chanels, out_channels, kernel_size = 3, stride =1 , padding = 1, use_bn = True, use_relu = True):
        super(ConvBlock, self).__init__()

        layers = [nn.Conv2d(in_chanels, out_channels, kernel_size, stride , padding)]

        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))

        if use_relu:
            layers.append(nn.ReLU(inplace=True))

    def forward(self, x):
        return self.conv(x)

class FeatureExtractor(nn.Module):
    def __init__(self, in_channels, base_channels = 64):
        super(FeatureExtractor, self).__init__()

        channels = [base_channels * (2 ** i) for i in range(4)] + [base_channels * 8]
        
        self.enc1 = self.encoder_block(in_channels, channels[0])
        self.enc2 = self.encoder_block(channels[0], channels[1])
        self.enc3 = self.encoder_block(channels[1], channels[2])
        self.enc4 = self.encoder_block(channels[2], channels[3])
        self.enc5 = self.encoder_block(channels[3], channels[4])

        self.pool = nn.MaxPool2d(2, 2)
    

    def _encoder_block(self, in_cahnnels, out_channels):
        return nn.Sequential(
            ConvBlock(in_cahnnels, out_channels),
            ConvBlock(out_channels, out_channels),
        )
    
    def forward(self, x):

        features = []

        x = self.enc1(x)
        features.append(x)
        x = self.pool(x)

        x = self.enc2(x)
        features.append(x)
        x = self.pool(x)

        x = self.enc3(x)
        features.append(x)
        x = self.pool(x)

        x = self.enc4(x)
        features.append(x)
        x = self.pool(x)

        x = self.enc5(x)
        features.append(x)
        x = self.pool(x)

        return x, features

class FeatureCorrelation(nn.Module):
    def __init__(self):
        super(FeatureCorrelation, self).__init__()

    def forward(feature_A, feature_B):
        B, C, H, W =feature_A.size()
        
        feature_A = F.normalize(feature_A, p=2, dim=1)
        feature_B = F.normalize(feature_B, p=2, dim=1)

        feature_A = feature_A.view(B, C, -1)
        feature_B = feature_B.view(B, C, -1)

        correlation = torch.bmm(feature_A.transpose(1, 2), feature_B) # (B, H*W, H*W)
        correlation = correlation.view(B, H * W, H, W)

        return correlation
    
