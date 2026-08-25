#!/usr/bin/env python3
"""
icnet.py — Illumination Correction Network (IC-Net Grid Map)
Sub-network siêu nhẹ kết hợp:
  - Nhánh Global: Dự đoán tham số photometric toàn cục (gamma_g, beta_g, alpha_g)
  - Nhánh Local Grid (8x8): Dự đoán bản đồ phần dư không gian (d_gamma, d_beta, d_alpha)
  - Bilinear Upsampling lên full-resolution để hiệu chỉnh thích nghi cả toàn cục lẫn cục bộ.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F


class ICNet(nn.Module):
    """
    IC-Net: Dual-Branch Spatially-Adaptive Illumination Correction Sub-network
    - Input: Ảnh gốc full-resolution (B, 3, H, W) trong dải [0, 1]
    - Parameter Estimation: Chạy trên thumbnail 64x64
    - Global Branch: Dự đoán (gamma_g, beta_g, alpha_g)
    - Local Grid Branch (8x8): Dự đoán (d_gamma, d_beta, d_alpha)
    - Output: Bản đồ tham số (B, 1, H, W) nội suy bilinear mượt mà
    - Transformation Formula:
        I_corrected = clamp(alpha(x,y) * (I_original.clamp(min=1e-4) ** gamma(x,y)) + beta(x,y), 0.0, 1.0)
    """
    def __init__(self, thumbnail_size: int = 64, grid_size: int = 8):
        super().__init__()
        self.thumbnail_size = thumbnail_size
        self.grid_size = grid_size

        # Backbone trích xuất đặc trưng: 64x64x3 -> 8x8x32
        self.backbone = nn.Sequential(
            # 64x64x3 -> 32x32x16
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            # 32x32x16 -> 16x16x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # 16x16x32 -> 8x8x32
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 1. Global Head: 8x8x32 -> 1x1 -> 3 tham số toàn cục (gamma_g, beta_g, alpha_g)
        self.global_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 3)  # [gamma_raw, beta_raw, alpha_raw]
        )

        # 2. Local Grid Head: 8x8x32 -> 8x8x3 bản đồ phần dư cục bộ (d_gamma, d_beta, d_alpha)
        self.local_head = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=True)  # [d_gamma_raw, d_beta_raw, d_alpha_raw]
        )

        self._init_weights()

    def _init_weights(self):
        """Khởi tạo trọng số sao cho ban đầu IC-Net đạt 100% Identity Transform."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.constant_(m.bias, 0.0)

        # 1. Khởi tạo Global gamma bias để đạt đúng gamma_g = 1.0 khi sigmoid(raw)
        gamma_target_sigmoid = (1.0 - 0.4) / 2.6
        gamma_bias = math.log(gamma_target_sigmoid / (1.0 - gamma_target_sigmoid))
        with torch.no_grad():
            self.global_head[-1].bias[0] = gamma_bias
            self.global_head[-1].bias[1] = 0.0  # beta = 0
            self.global_head[-1].bias[2] = 0.0  # alpha = 1.0

            # 2. Khởi tạo Local Head weight rất nhỏ & bias = 0 để phần dư ban đầu ~ 0.0
            if self.local_head[-1].bias is not None:
                nn.init.constant_(self.local_head[-1].bias, 0.0)
            nn.init.normal_(self.local_head[-1].weight, std=0.001)

    def predict_param_maps(self, x: torch.Tensor):
        """
        Dự đoán bản đồ tham số không gian (B, 1, H, W) cho gamma, beta, alpha.
        x: (B, 3, H, W) normalized [0, 1]
        """
        B, C, H, W = x.shape
        if x.shape[-2:] != (self.thumbnail_size, self.thumbnail_size):
            thumbnail = F.interpolate(x, size=(self.thumbnail_size, self.thumbnail_size),
                                      mode='bilinear', align_corners=False)
        else:
            thumbnail = x

        feats = self.backbone(thumbnail)  # (B, 32, 8, 8)

        # 1. Nhánh Global
        g_raw = self.global_head(feats)  # (B, 3)
        gamma_g = (0.4 + 2.6 * torch.sigmoid(g_raw[:, 0:1])).view(B, 1, 1, 1)  # [0.4, 3.0]
        beta_g  = (0.3 * torch.tanh(g_raw[:, 1:2])).view(B, 1, 1, 1)           # [-0.3, 0.3]
        alpha_g = (0.5 + 1.0 * torch.sigmoid(g_raw[:, 2:3])).view(B, 1, 1, 1)  # [0.5, 1.5]

        # 2. Nhánh Local Grid (8x8)
        l_raw = self.local_head(feats)   # (B, 3, 8, 8)
        d_gamma = 0.8 * torch.tanh(l_raw[:, 0:1])  # [-0.8, 0.8]
        d_beta  = 0.2 * torch.tanh(l_raw[:, 1:2])  # [-0.2, 0.2]
        d_alpha = 0.3 * torch.tanh(l_raw[:, 2:3])  # [-0.3, 0.3]

        # 3. Kết hợp Global Base + Local Delta và kẹp biên an toàn
        gamma_grid = torch.clamp(gamma_g + d_gamma, 0.4, 3.0)
        beta_grid  = torch.clamp(beta_g + d_beta, -0.3, 0.3)
        alpha_grid = torch.clamp(alpha_g + d_alpha, 0.5, 1.5)

        # 4. Nội suy Bilinear phóng to lên kích thước ảnh đầy đủ (H, W)
        gamma_map = F.interpolate(gamma_grid, size=(H, W), mode='bilinear', align_corners=False)
        beta_map  = F.interpolate(beta_grid, size=(H, W), mode='bilinear', align_corners=False)
        alpha_map = F.interpolate(alpha_grid, size=(H, W), mode='bilinear', align_corners=False)

        return gamma_map, beta_map, alpha_map

    def forward(self, x: torch.Tensor, return_params: bool = False):
        """
        Hiệu chỉnh ánh sáng ảnh full-resolution.
        x: (B, 3, H, W) normalized [0.0, 1.0]
        """
        gamma_map, beta_map, alpha_map = self.predict_param_maps(x)
        
        # Biến đổi photometric thích nghi không gian
        x_safe = x.clamp(min=1e-4)
        x_corrected = torch.clamp(alpha_map * torch.pow(x_safe, gamma_map) + beta_map, 0.0, 1.0)

        if return_params:
            return x_corrected, (gamma_map, beta_map, alpha_map)
        return x_corrected

    def count_parameters(self) -> int:
        """Đếm tổng số tham số có thể huấn luyện."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def measure_latency(self, device: str = 'cuda', input_shape=(1, 3, 640, 640), n_runs: int = 100, n_warmup: int = 10) -> float:
        """Đo độ trễ (latency ms) của IC-Net."""
        self.eval()
        self.to(device)
        dummy_input = torch.rand(input_shape, device=device)

        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self(dummy_input)

            if 'cuda' in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                for _ in range(n_runs):
                    _ = self(dummy_input)
                torch.cuda.synchronize()
                end_time = time.perf_counter()
            else:
                start_time = time.perf_counter()
                for _ in range(n_runs):
                    _ = self(dummy_input)
                end_time = time.perf_counter()

        avg_latency_ms = ((end_time - start_time) / n_runs) * 1000.0
        return avg_latency_ms


if __name__ == "__main__":
    net = ICNet()
    n_params = net.count_parameters()
    print("IC-Net Grid (Dual-Branch) Initialized Successfully.")
    print(f"Total Trainable Parameters: {n_params:,} (~{n_params/1000:.1f}K)")
    
    # Test forward pass
    dummy = torch.rand(2, 3, 640, 640)
    out, (g, b, a) = net(dummy, return_params=True)
    print(f"Output Shape: {out.shape}, Range: [{out.min().item():.3f}, {out.max().item():.3f}]")
    print(f"Mean Gamma: {g.mean().item():.4f}, Beta: {b.mean().item():.4f}, Alpha: {a.mean().item():.4f}")