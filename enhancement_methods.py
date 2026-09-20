#!/usr/bin/env python3
"""
enhancement_methods.py — Registry of Illumination / Low-Light Correction methods
used for benchmark comparisons against MQTone
in 5-fold cross-validation benchmarks.

Supported methods:
  - 'none'        : Uncorrected Baseline (standard YOLOv8n/YOLO11n)
  - 'mqtone'      : MQTone (learnable dual-branch tone mapping, see mqtone.py) — LEARNABLE
  - 'gamma'       : Adaptive Gamma Correction — NON-LEARNABLE (0 parameters)
  - 'clahe'       : CLAHE on L channel of Lab — NON-LEARNABLE (0 parameters)
  - 'iat'         : IAT (Illumination-Adaptive Transformer) — LEARNABLE
  - 'zerodce'     : Zero-DCE (CVPR 2020) — LEARNABLE
  - 'zerodce_pp'  : Zero-DCE++ (TPAMI 2021) — LEARNABLE
  - 'retinexnet'  : RetinexNet Enhance-Net (BMVC 2018) — LEARNABLE
                    trained end-to-end with detection and photometric losses
  - 'enlightengan': EnlightenGAN Generator U-Net (IEEE TIP 2021) — LEARNABLE
                    ~8.6M params, GAN discriminator replaced by L_photo supervised loss
                    for fair comparison with other methods in this benchmark
"""

import math
import sys
import time

# Fix UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Method registry & display names
# ─────────────────────────────────────────────────────────────────────────────
CORRECTION_METHODS = [
    'none', 'mqtone', 'iat', 'zerodce', 'zerodce_pp',
    'gamma', 'clahe', 'retinexnet', 'enlightengan', 'afifi',
]

# Non-learnable methods: no standalone training, no photometric loss,
# applied as a fixed deterministic transformation during forward passes.
NON_LEARNABLE_METHODS = {'gamma', 'clahe'}

METHOD_DISPLAY_NAMES = {
    'none':         'Baseline (No Correction)',
    'mqtone':       'MQTone (Proposed)',
    'icnet':        'MQTone (Proposed)',
    'iat':          'IAT (Transformer)',
    'zerodce':      'Zero-DCE',
    'zerodce_pp':   'Zero-DCE++',
    'gamma':        'Gamma Correction',
    'clahe':        'CLAHE',
    'retinexnet':   'RetinexNet',
    'enlightengan': 'EnlightenGAN',
    'afifi':        'Afifi et al. (Exposure Corr.)',
}


def is_learnable_method(method: str) -> bool:
    method = method.lower()
    return method != 'none' and method not in NON_LEARNABLE_METHODS


# ─────────────────────────────────────────────────────────────────────────────
# Base class: unified interface for all enhancement modules
# (implements count_parameters() and measure_latency() for benchmark consistency)

# ─────────────────────────────────────────────────────────────────────────────
class BaseCorrectionModule(nn.Module):
    """Common interface: forward(x) returns enhanced frame (B, 3, H, W) in [0, 1]."""

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def measure_latency(self, device: str = 'cuda', input_shape=(1, 3, 640, 640),
                         n_runs: int = 100, n_warmup: int = 10) -> float:
        self.eval()
        self.to(device)
        dummy_input = torch.rand(input_shape, device=device)
        with torch.no_grad():
            for _ in range(n_warmup):
                _ = self(dummy_input)
            if 'cuda' in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    _ = self(dummy_input)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
            else:
                t0 = time.perf_counter()
                for _ in range(n_runs):
                    _ = self(dummy_input)
                t1 = time.perf_counter()
        return ((t1 - t0) / n_runs) * 1000.0


# ─────────────────────────────────────────────────────────────────────────────
# 1. Zero-DCE (CVPR 2020, LEARNABLE)
# ─────────────────────────────────────────────────────────────────────────────
class ZeroDCECorrection(BaseCorrectionModule):
    """
    Zero-DCE (Zero-Reference Deep Curve Estimation, CVPR 2020):
    7-layer CNN with skip connections predicting 8 curve maps (24 channels),
    applying an 8-iteration Light Enhancement curve to correct illumination.
    """
    def __init__(self, number_f: int = 32):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.e_conv1 = nn.Conv2d(3, number_f, 3, 1, 1, bias=True)
        self.e_conv2 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv3 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv4 = nn.Conv2d(number_f, number_f, 3, 1, 1, bias=True)
        self.e_conv5 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv6 = nn.Conv2d(number_f * 2, number_f, 3, 1, 1, bias=True)
        self.e_conv7 = nn.Conv2d(number_f * 2, 24, 3, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))
        r1, r2, r3, r4, r5, r6, r7, r8 = torch.split(x_r, 3, dim=1)

        x_cur = x + r1 * (torch.pow(x, 2) - x)
        x_cur = x_cur + r2 * (torch.pow(x_cur, 2) - x_cur)
        x_cur = x_cur + r3 * (torch.pow(x_cur, 2) - x_cur)
        x_cur = x_cur + r4 * (torch.pow(x_cur, 2) - x_cur)
        x_cur = x_cur + r5 * (torch.pow(x_cur, 2) - x_cur)
        x_cur = x_cur + r6 * (torch.pow(x_cur, 2) - x_cur)
        x_cur = x_cur + r7 * (torch.pow(x_cur, 2) - x_cur)
        enhance_image = x_cur + r8 * (torch.pow(x_cur, 2) - x_cur)
        return enhance_image.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Zero-DCE++ (TPAMI 2021, LEARNABLE)
# ─────────────────────────────────────────────────────────────────────────────
class CSDN_Tem(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_ch,
            bias=True
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1,
            bias=True
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


class ZeroDCEppCorrection(BaseCorrectionModule):
    """
    Zero-DCE++ (TPAMI 2021):
    Lightweight version with Depthwise Separable Convolutions and parameter sharing,
    predicting a single 3-channel curve parameter map (rather than 24 channels).
    """
    def __init__(self, number_f: int = 32, scale_factor: int = 1):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

        self.e_conv1 = CSDN_Tem(3, number_f)
        self.e_conv2 = CSDN_Tem(number_f, number_f)
        self.e_conv3 = CSDN_Tem(number_f, number_f)
        self.e_conv4 = CSDN_Tem(number_f, number_f)
        self.e_conv5 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv6 = CSDN_Tem(number_f * 2, number_f)
        self.e_conv7 = CSDN_Tem(number_f * 2, 3)

    def enhance(self, x: torch.Tensor, x_r: torch.Tensor) -> torch.Tensor:
        x_cur = x
        for _ in range(8):
            x_cur = x_cur + x_r * (torch.pow(x_cur, 2) - x_cur)
        return x_cur

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale_factor == 1:
            x_down = x
        else:
            x_down = F.interpolate(x, scale_factor=1.0 / self.scale_factor, mode='bilinear', align_corners=False)

        x1 = self.relu(self.e_conv1(x_down))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3, x4], dim=1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2, x5], dim=1)))
        x_r = torch.tanh(self.e_conv7(torch.cat([x1, x6], dim=1)))

        if self.scale_factor != 1:
            x_r = F.interpolate(x_r, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

        enhance_image = self.enhance(x, x_r)
        return enhance_image.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 3. IAT (Illumination-Adaptive Transformer, BMVC 2022, LEARNABLE)
# ─────────────────────────────────────────────────────────────────────────────
class IATCorrection(BaseCorrectionModule):
    """
    IAT (Illumination-Adaptive Transformer, BMVC 2022):
    Architecture combining Local Net (CNN/Transformer blocks) for multiplier/addition maps
    and Global Net (Transformer-based) for Color Correction Matrix (CCM)
    and adaptive gamma estimation.
    """
    def __init__(self, in_dim: int = 3, with_global: bool = True, type: str = 'lol'):
        super().__init__()
        try:
            from IAT_enhance.IAT_main import IAT
        except ImportError:
            from .IAT_enhance.IAT_main import IAT
        self.iat = IAT(in_dim=in_dim, with_global=with_global, type=type)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # IAT forward returns (mul, add, img_high)
        _, _, img_high = self.iat(x)
        return img_high.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Gamma Correction (Non-learnable classical baseline)
# ─────────────────────────────────────────────────────────────────────────────
class GammaCorrection(BaseCorrectionModule):
    """
    Classical adaptive gamma correction (non-learnable):
    Estimates gamma per image such that post-correction mean luminance
    approximates `target_mean` (default: 0.45):
        gamma = log(target_mean) / log(mean_luminance)
        I_out = clamp(I_in, eps, 1) ^ gamma
    """
    def __init__(self, target_mean: float = 0.45, gamma_min: float = 0.3, gamma_max: float = 3.0):
        super().__init__()
        self.target_mean = target_mean
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        # Dummy buffer ensuring .to(device) works consistently without parameters
        self.register_buffer('_dummy', torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        eps = 1e-4
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        mean_lum = gray.mean(dim=[1, 2, 3], keepdim=True).clamp(min=eps, max=1.0 - eps)
        log_target = math.log(self.target_mean)
        gamma = (log_target / torch.log(mean_lum)).clamp(self.gamma_min, self.gamma_max)
        out = torch.pow(x.clamp(min=eps), gamma)
        return out.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLAHE (Non-learnable classical baseline, non-differentiable)
# ─────────────────────────────────────────────────────────────────────────────
class CLAHECorrection(BaseCorrectionModule):
    """
    Contrast Limited Adaptive Histogram Equalization applied on L channel of Lab space
    (preserves hue, enhances local luminance contrast).
    Non-learnable OpenCV CPU implementation (non-differentiable).
    Latency is measured on CPU per image without GPU batch acceleration.
    
    """
    def __init__(self, clip_limit: float = 2.0, tile_grid_size: int = 8):
        super().__init__()
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.register_buffer('_dummy', torch.zeros(1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        B = x.shape[0]
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit,
                                 tileGridSize=(self.tile_grid_size, self.tile_grid_size))
        x_np = (x.detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)  # (B,3,H,W)
        out_np = np.empty_like(x_np)
        for b in range(B):
            img_rgb = np.transpose(x_np[b], (1, 2, 0))  # (H,W,3) RGB
            lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
            l, a, bb = cv2.split(lab)
            l_eq = clahe.apply(l)
            lab_eq = cv2.merge((l_eq, a, bb))
            rgb_eq = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)
            out_np[b] = np.transpose(rgb_eq, (2, 0, 1))
        out = torch.from_numpy(out_np).float().to(device) / 255.0
        return out.clamp(0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# 6. RetinexNet Enhance-Net (BMVC 2018, Wei et al., LEARNABLE)
# ─────────────────────────────────────────────────────────────────────────────
class RetinexNetCorrection(BaseCorrectionModule):
    """
    RetinexNet Enhance-Net style (BMVC 2018, Wei et al.
    'Deep Retinex Decomposition for Low-Light Enhancement').

    Implements the Enhance-Net component as a self-contained image-to-image
    network: 3-level multi-scale encoder-decoder with skip connections and a
    multi-scale feature bridge — the characteristic design of RetinexNet's
    Enhance-Net, applied directly to full RGB input.

    Fair comparison conditions (identical to all other methods):
      - Train from scratch end-to-end with L_det + lambda_photo * L_photo
      - No Decom-Net pretrained weights required
      - Same Protocol B 5-fold cross-validation splits
      - Same YOLOv8n / YOLO11n detector backbone

    Original paper: https://arxiv.org/abs/1808.04560
    Reference PyTorch: https://github.com/aasharma90/RetinexNet_PyTorch
    ~533K parameters.
    """

    def __init__(self, n_ch: int = 32):
        super().__init__()

        # ── Encoder: 3-level double-conv blocks with BatchNorm ──
        self.enc1 = self._double_conv(3,       n_ch,   stride=1)   # (B, 32,  H,   W  )
        self.enc2 = self._double_conv(n_ch,    n_ch*2, stride=2)   # (B, 64,  H/2, W/2)
        self.enc3 = self._double_conv(n_ch*2,  n_ch*4, stride=2)   # (B, 128, H/4, W/4)

        # ── Multi-scale bridge (Enhance-Net characteristic):
        #    pool enc1 & enc2 down to enc3 resolution, concatenate,
        #    then fuse with two conv layers for global context ──
        bridge_in = n_ch + n_ch*2 + n_ch*4   # 32 + 64 + 128 = 224
        self.bridge = nn.Sequential(
            nn.Conv2d(bridge_in, n_ch*4, kernel_size=1, bias=False),
            nn.BatchNorm2d(n_ch*4),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_ch*4, n_ch*4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_ch*4),
            nn.ReLU(inplace=True),
        )

        # ── Decoder: ConvTranspose upsampling + skip connections ──
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(n_ch*4, n_ch*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_ch*2),
            nn.ReLU(inplace=True),
        )                                                            # (B, 64,  H/2, W/2)
        # dec2 input = dec3 output (64) + enc2 skip (64) = 128 channels
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(n_ch*2 + n_ch*2, n_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_ch),
            nn.ReLU(inplace=True),
        )                                                            # (B, 32,  H,   W  )
        # dec1 input = dec2 output (32) + enc1 skip (32) = 64 channels
        self.dec1 = nn.Sequential(
            nn.Conv2d(n_ch + n_ch, n_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(n_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_ch, 3, kernel_size=3, padding=1, bias=True),
            nn.Sigmoid(),
        )                                                            # (B, 3,   H,   W  )

    @staticmethod
    def _double_conv(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
        """Two consecutive Conv-BN-ReLU layers; first conv uses given stride."""
        return nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1,      padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)   # (B, 32,  H,   W  )
        e2 = self.enc2(e1)  # (B, 64,  H/2, W/2)
        e3 = self.enc3(e2)  # (B, 128, H/4, W/4)

        # Multi-scale bridge: pool enc1 & enc2 to enc3 spatial size
        e1_ds = F.adaptive_avg_pool2d(e1, output_size=e3.shape[2:])
        e2_ds = F.adaptive_avg_pool2d(e2, output_size=e3.shape[2:])
        bridge = self.bridge(torch.cat([e3, e2_ds, e1_ds], dim=1))  # (B, 128, H/4, W/4)

        # Decoder with skip connections
        d3  = self.dec3(bridge)                           # (B, 64,  H/2, W/2)
        d2  = self.dec2(torch.cat([d3, e2], dim=1))      # (B, 32,  H,   W  )
        out = self.dec1(torch.cat([d2, e1], dim=1))      # (B, 3,   H,   W  )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# 7. EnlightenGAN Generator U-Net (IEEE TIP 2021, LEARNABLE)
# ─────────────────────────────────────────────────────────────────────────────
class EnlightenGANCorrection(BaseCorrectionModule):
    """
    EnlightenGAN Generator-only variant (VITA-Group,
    'EnlightenGAN: Deep Light Enhancement without Paired Supervision',
    IEEE TIP 2021).

    Faithfully implements the Unet_resize_conv generator from the official
    VITA-Group/EnlightenGAN repository with:
      - Self-attention luminance guide: luminance channel (ITU-R BT.601)
        appended to RGB input → 4-channel input (as in opt.self_attention=True)
      - 7-level U-Net encoder (MaxPool2d downsampling)
      - Resize-convolution decoder (bilinear upsample + conv, avoids
        ConvTranspose2d checkerboard artifacts as in the original code)
      - InstanceNorm throughout, LeakyReLU encoder / ReLU decoder

    Fair comparison conditions (identical to all other methods):
      - GAN discriminator loss REPLACED by L_photo = ||I_corr - I_clean||_1
        (same supervised photometric loss used by MQTone, Zero-DCE, IAT,
        RetinexNet in this benchmark)
      - Train from scratch end-to-end with L_det + lambda_photo * L_photo
      - No pretrained VGG perceptual loss or paired LOL dataset weights
      - Same Protocol B 5-fold cross-validation splits
      - Same YOLOv8n / YOLO11n detector backbone

    Original code: https://github.com/VITA-Group/EnlightenGAN
    ~8.6M parameters (Generator only, ngf=32 as in the official implementation).
    """

    def __init__(self, ngf: int = 32):
        """
        Args:
            ngf: Base number of generator filters. Default 32 matches the
                 official Unet_resize_conv implementation (conv1_1 outputs 32
                 channels). The 7-level U-Net with ngf=32 yields ~8.6M parameters.
        """
        super().__init__()

        IN   = nn.InstanceNorm2d  # InstanceNorm as in the original
        p    = 1                  # padding for all 3×3 convolutions

        def enc_block(in_ch: int, out_ch: int, use_norm: bool = True) -> nn.Sequential:
            """Conv3×3 → [InstanceNorm] → LeakyReLU(0.2) — encoder building block."""
            layers: list = [nn.Conv2d(in_ch, out_ch, 3, 1, p, bias=not use_norm)]
            if use_norm:
                layers.append(IN(out_ch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def dec_block(in_ch: int, out_ch: int, use_norm: bool = True) -> nn.Sequential:
            """Conv3×3 → [InstanceNorm] → ReLU — decoder building block (after upsample)."""
            layers: list = [nn.Conv2d(in_ch, out_ch, 3, 1, p, bias=not use_norm)]
            if use_norm:
                layers.append(IN(out_ch, affine=True))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        # Channel widths: 32 → 64 → 128 → 256 → 256 → 256 → 256 (cap at ngf*8)
        ch = [min(ngf * (2**i), ngf * 8) for i in range(8)]
        # e.g. ngf=32 → [32, 64, 128, 256, 256, 256, 256, 256]

        # ── Encoder: 7 levels ──────────────────────────────────────────────
        # Level 1: 4-channel input (RGB + luminance), no IN on very first conv
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, ch[0], 3, 1, p, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            enc_block(ch[0], ch[0]),                              # (B, 32,  H,    W   )
        )
        self.pool1 = nn.MaxPool2d(2)                              # → H/2

        self.enc2 = nn.Sequential(
            enc_block(ch[0], ch[1]),
            enc_block(ch[1], ch[1]),                              # (B, 64,  H/2,  W/2 )
        )
        self.pool2 = nn.MaxPool2d(2)                              # → H/4

        self.enc3 = nn.Sequential(
            enc_block(ch[1], ch[2]),
            enc_block(ch[2], ch[2]),
            enc_block(ch[2], ch[2]),                              # (B, 128, H/4,  W/4 )
        )
        self.pool3 = nn.MaxPool2d(2)                              # → H/8

        self.enc4 = nn.Sequential(
            enc_block(ch[2], ch[3]),
            enc_block(ch[3], ch[3]),
            enc_block(ch[3], ch[3]),                              # (B, 256, H/8,  W/8 )
        )
        self.pool4 = nn.MaxPool2d(2)                              # → H/16

        self.enc5 = nn.Sequential(
            enc_block(ch[3], ch[4]),
            enc_block(ch[4], ch[4]),
            enc_block(ch[4], ch[4]),                              # (B, 256, H/16, W/16)
        )
        self.pool5 = nn.MaxPool2d(2)                              # → H/32

        self.enc6 = nn.Sequential(
            enc_block(ch[4], ch[5]),
            enc_block(ch[5], ch[5]),                              # (B, 256, H/32, W/32)
        )
        self.pool6 = nn.MaxPool2d(2)                              # → H/64

        # Level 7 — bottleneck (innermost, no IN as in official code)
        self.enc7 = nn.Sequential(
            enc_block(ch[5], ch[6], use_norm=False),
            enc_block(ch[6], ch[6], use_norm=False),              # (B, 256, H/64, W/64)
        )

        # ── Decoder: 7 levels (resize-conv to avoid checkerboard) ──────────
        # Each dec_block is applied AFTER bilinear upsample + skip concat.
        self.dec7 = dec_block(ch[6] + ch[5], ch[5])              # (B, 256, H/32, W/32)
        self.dec6 = dec_block(ch[5] + ch[4], ch[4])              # (B, 256, H/16, W/16)
        self.dec5 = dec_block(ch[4] + ch[3], ch[3])              # (B, 256, H/8,  W/8 )
        self.dec4 = dec_block(ch[3] + ch[2], ch[2])              # (B, 128, H/4,  W/4 )
        self.dec3 = dec_block(ch[2] + ch[1], ch[1])              # (B, 64,  H/2,  W/2 )
        self.dec2 = dec_block(ch[1] + ch[0], ch[0])              # (B, 32,  H,    W   )
        self.dec1 = nn.Sequential(
            nn.Conv2d(ch[0], 3, 3, 1, p, bias=True),
            nn.Tanh(),                                             # output ∈ [-1, 1]
        )

    @staticmethod
    def _upsample_cat(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Bilinear upsample x to skip's spatial dims, then concat along channel."""
        x_up = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([x_up, skip], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention: luminance map as 4th input channel (ITU-R BT.601)
        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x4  = torch.cat([x, lum], dim=1)   # (B, 4, H, W)

        # Encoder
        e1 = self.enc1(x4)               # (B, 32,  H,    W   )
        e2 = self.enc2(self.pool1(e1))   # (B, 64,  H/2,  W/2 )
        e3 = self.enc3(self.pool2(e2))   # (B, 128, H/4,  W/4 )
        e4 = self.enc4(self.pool3(e3))   # (B, 256, H/8,  W/8 )
        e5 = self.enc5(self.pool4(e4))   # (B, 256, H/16, W/16)
        e6 = self.enc6(self.pool5(e5))   # (B, 256, H/32, W/32)
        e7 = self.enc7(self.pool6(e6))   # (B, 256, H/64, W/64)

        # Decoder (bilinear upsample → concat skip → conv)
        d7  = self.dec7(self._upsample_cat(e7, e6))   # (B, 256, H/32, W/32)
        d6  = self.dec6(self._upsample_cat(d7, e5))   # (B, 256, H/16, W/16)
        d5  = self.dec5(self._upsample_cat(d6, e4))   # (B, 256, H/8,  W/8 )
        d4  = self.dec4(self._upsample_cat(d5, e3))   # (B, 128, H/4,  W/4 )
        d3  = self.dec3(self._upsample_cat(d4, e2))   # (B, 64,  H/2,  W/2 )
        d2  = self.dec2(self._upsample_cat(d3, e1))   # (B, 32,  H,    W   )
        out = self.dec1(d2)                             # (B, 3,   H,    W  ) ∈ [-1,1]

        # Remap Tanh [-1, 1] → [0, 1]
        return (out + 1.0) * 0.5


# ─────────────────────────────────────────────────────────────────────────────
# 8. Afifi et al. Coarse-to-Fine Multi-Scale Exposure Correction (CVPR 2021)
# ─────────────────────────────────────────────────────────────────────────────
class _ScaleUNet(nn.Module):
    """
    Single-scale U-Net sub-network used by AfifiCorrection.

    Architecture (from the paper, Section 3.1 and official code):
      - 3-level encoder: Conv-BN-ReLU blocks with stride-2 MaxPool
      - Bottleneck: 2x Conv-BN-ReLU at coarsest scale
      - 3-level decoder: bilinear upsample + skip concat + Conv-BN-ReLU
      - Final 1x1 Conv + Sigmoid for [0,1] output

    Args:
        in_ch : input channels (3 for coarsest scale; 6 for finer scales,
                because coarser output is concatenated as guidance)
        ngf   : base filter count (default 24, as in the paper)
    """

    def __init__(self, in_ch: int = 3, ngf: int = 24):
        super().__init__()

        def _block(ic, oc, stride=1):
            return nn.Sequential(
                nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
                nn.Conv2d(oc, oc, 3, padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = _block(in_ch,   ngf)        # (B, ngf,   H,   W  )
        self.enc2 = _block(ngf,     ngf*2)      # (B, ngf*2, H/2, W/2)
        self.enc3 = _block(ngf*2,   ngf*4)      # (B, ngf*4, H/4, W/4)

        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = _block(ngf*4, ngf*8)  # (B, ngf*8, H/4, W/4)

        # Decoder (bilinear upsample + skip)
        self.dec3 = _block(ngf*8 + ngf*4, ngf*4)
        self.dec2 = _block(ngf*4 + ngf*2, ngf*2)
        self.dec1 = _block(ngf*2 + ngf,   ngf)

        # Output head
        self.out_conv = nn.Sequential(
            nn.Conv2d(ngf, 3, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)                    # (B, ngf,   H,   W  )
        e2 = self.enc2(self.pool(e1))        # (B, ngf*2, H/2, W/2)
        e3 = self.enc3(self.pool(e2))        # (B, ngf*4, H/4, W/4)
        b  = self.bottleneck(e3)             # (B, ngf*8, H/4, W/4)  [no extra pool]

        d3 = self.dec3(torch.cat([
            F.interpolate(b,  e3.shape[2:], mode='bilinear', align_corners=True), e3], 1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=True), e2], 1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=True), e1], 1))
        return self.out_conv(d1)


class AfifiCorrection(BaseCorrectionModule):
    """
    Afifi et al. 'Learning Multi-Scale Photo Exposure Correction' (CVPR 2021).

    Coarse-to-fine multi-scale architecture with 3 sequential sub-networks
    operating on Laplacian pyramid levels (1/4, 1/2, full resolution).
    The unique strength of this method is handling BOTH over- and under-exposed
    images, making it the only SOTA method in this benchmark that directly targets
    the overexposure domain of the polymer banknote challenge.

    Architecture (Section 3 and Fig. 2 of the paper):
      1. sub_s3: processes I @ 1/4 scale (in_ch=3)         → Y_s3 ∈ [0,1]
      2. sub_s2: processes I @ 1/2 scale with Y_s3 guidance  → Y_s2 ∈ [0,1]
         (upsample Y_s3 to 1/2 resolution, concat with I @1/2 → 6 channels)
      3. sub_s1: processes I @ full scale with Y_s2 guidance → Y_s1 ∈ [0,1]
         (upsample Y_s2 to full resolution, concat with I      → 6 channels)
      Final output: Y_s1 (full-resolution corrected image)

    Fair comparison conditions (identical to all other methods in this benchmark):
      - Trained from scratch end-to-end with L_det + lambda_photo * L_photo
      - No pretrained exposure-correction weights
      - Same Protocol B 5-fold cross-validation splits
      - Same YOLOv8n / YOLO11n detector backbone
      Note: This method was originally trained on 24,330 paired images; results
      under the constrained mobile deployment setting (336 imgs/fold) reflect
      data efficiency rather than peak architectural capability.

    Original paper: https://arxiv.org/abs/2003.11596
    Official code:  https://github.com/mahmoudnafifi/Exposure_Correction
    ~4M parameters (3 x ScaleUNet with ngf=24).
    """

    def __init__(self, ngf: int = 24):
        """
        Args:
            ngf: Base filter count per sub-network (default 24 as in the paper).
                 3 sub-networks at scales 1/4, 1/2, 1x yield ~4M total params.
        """
        super().__init__()
        # Scale 1/4 — coarsest, processes RGB only
        self.sub_s3 = _ScaleUNet(in_ch=3, ngf=ngf)
        # Scale 1/2 — receives RGB + upsampled coarse output (6 channels)
        self.sub_s2 = _ScaleUNet(in_ch=6, ngf=ngf)
        # Scale 1/1 — receives RGB + upsampled mid output (6 channels)
        self.sub_s1 = _ScaleUNet(in_ch=6, ngf=ngf)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        H, W = x.shape[2], x.shape[3]

        # Scale 1/4 — coarse colour correction
        x_s3 = F.interpolate(x, size=(H // 4, W // 4), mode='bilinear', align_corners=True)
        y_s3 = self.sub_s3(x_s3)                                 # (B, 3, H/4, W/4)

        # Scale 1/2 — detail refinement, guided by coarse output
        x_s2 = F.interpolate(x, size=(H // 2, W // 2), mode='bilinear', align_corners=True)
        y_s3_up = F.interpolate(y_s3, size=x_s2.shape[2:], mode='bilinear', align_corners=True)
        y_s2 = self.sub_s2(torch.cat([x_s2, y_s3_up], dim=1))   # (B, 3, H/2, W/2)

        # Scale 1/1 — full-resolution fine correction, guided by mid output
        y_s2_up = F.interpolate(y_s2, size=(H, W), mode='bilinear', align_corners=True)
        y_s1 = self.sub_s1(torch.cat([x, y_s2_up], dim=1))      # (B, 3, H, W)

        return y_s1   # already in [0, 1] via Sigmoid in _ScaleUNet.out_conv


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_correction_module(method: str):
    """
    Return nn.Module corresponding to `method` (set to .train() by default),
    or None if method == 'none' (uncorrected baseline).
    """
    method = method.lower()
    if method == 'none':
        return None
    if method in ('mqtone', 'icnet'):
        from mqtone import MQTone
        return MQTone()
    if method == 'iat':
        return IATCorrection()
    if method == 'zerodce':
        return ZeroDCECorrection()
    if method == 'zerodce_pp':
        return ZeroDCEppCorrection()
    if method == 'gamma':
        return GammaCorrection()
    if method == 'clahe':
        return CLAHECorrection()
    if method == 'retinexnet':
        return RetinexNetCorrection()
    if method == 'enlightengan':
        return EnlightenGANCorrection()
    if method == 'afifi':
        return AfifiCorrection()
    raise ValueError(
        f"Invalid correction method: '{method}'. "
        f"Valid choices: {CORRECTION_METHODS}"
    )


if __name__ == "__main__":
    print(f"\n{'Method':<16} {'Display Name':<30} {'Params':>12} {'Learnable':>10} {'GPU Lat (ms)':>14}")
    print("-" * 87)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for m in CORRECTION_METHODS:
        if m == 'none':
            print(f"{'none':<16} {'Baseline (No Correction)':<30} {'0':>12} {'No':>10} {'0.00':>14}")
            continue
        mod = build_correction_module(m)
        n_params = mod.count_parameters()
        learnable = "Yes" if is_learnable_method(m) else "No"
        try:
            lat_ms = mod.measure_latency(device=device, input_shape=(1, 3, 640, 640),
                                          n_runs=50, n_warmup=5)
            lat_str = f"{lat_ms:.2f}"
        except Exception as e:
            lat_str = f"ERR"
        print(
            f"{m:<16} {METHOD_DISPLAY_NAMES[m]:<30} {n_params:>12,} {learnable:>10} {lat_str:>14}"
        )