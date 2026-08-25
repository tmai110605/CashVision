#!/usr/bin/env python3
"""
enhancement_methods.py — Registry các phương pháp Illumination / Low-Light
Correction dùng để so sánh với IC-Net (đề xuất chính của C2) trong bảng
so sánh cuối cùng (final_all_methods_all_folds → summary_final_5fold_all_methods_headline.csv).

Các phương pháp:
  - 'none'   : Không hiệu chỉnh (Baseline thuần YOLOv8n)
  - 'icnet'  : IC-Net (đề xuất của C2, xem icnet.py) — HỌC ĐƯỢC
  - 'gamma'  : Gamma Correction thích nghi cổ điển — KHÔNG học được (0 tham số)
  - 'clahe'  : CLAHE cổ điển trên kênh L (Lab) — KHÔNG học được (0 tham số)

╔══════════════════════════════════════════════════════════════════════════╗
║ GHI CHÚ — VỀ VIỆC LOẠI ZERO-DCE++ / SCI / RETINEXFORMER KHỎI SO SÁNH     ║
╠══════════════════════════════════════════════════════════════════════════╣
║ Phiên bản trước của file này có kèm 3 module HỌC ĐƯỢC bổ sung            ║
║ (Zero-DCE++, SCI, Retinexformer) để so sánh với IC-Net. Các module đó   ║
║ đã bị GỠ BỎ HOÀN TOÀN khỏi registry/so sánh của bài báo này vì:          ║
║   - Đó là bản REIMPLEMENTATION rút gọn tự viết theo ý tưởng kiến trúc   ║
║     cốt lõi của từng paper gốc (Zero-DCE++ 2021, SCI CVPR'22,           ║
║     Retinexformer ICCV'23) -- KHÔNG phải code/checkpoint pretrained     ║
║     chính chủ của tác giả, và quy mô tham số nhỏ hơn nhiều so với kiến  ║
║     trúc gốc trong paper (vd Retinexformer gốc là Transformer nhiều    ║
║     triệu tham số, còn bản rút gọn chỉ ~15K).                          ║
║   - Vì không phải bản gốc/pretrained chính chủ, so sánh trực tiếp với  ║
║     các con số này (accuracy lẫn cost) không đại diện công bằng cho    ║
║     Zero-DCE++/SCI/Retinexformer thật -- nên bài báo này CHỈ báo cáo    ║
║     so sánh IC-Net với 2 baseline cổ điển KHÔNG học được (Gamma, CLAHE) ║
║     và baseline thuần (none), tránh nhầm lẫn cho người đọc.            ║
║ Nếu về sau muốn khôi phục so sánh với bản gốc, cần nạp checkpoint      ║
║ pretrained chính chủ qua module.load_state_dict(...) sau khi           ║
║ build_correction_module(), thay vì dùng lại 3 class rút gọn đã gỡ.     ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import math
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Registry & tên hiển thị
# ─────────────────────────────────────────────────────────────────────────────
CORRECTION_METHODS = ['none', 'icnet', 'gamma', 'clahe']

# Method không có tham số học được -> không train riêng, không có Photometric Loss,
# áp dụng như một phép biến đổi cố định ở MỌI epoch (giống hệt mọi lần forward).
NON_LEARNABLE_METHODS = {'gamma', 'clahe'}

METHOD_DISPLAY_NAMES = {
    'none': 'Baseline (No Correction)',
    'icnet': 'IC-Net (Proposed)',
    'gamma': 'Gamma Correction',
    'clahe': 'CLAHE',
}


def is_learnable_method(method: str) -> bool:
    method = method.lower()
    return method != 'none' and method not in NON_LEARNABLE_METHODS


# ─────────────────────────────────────────────────────────────────────────────
# Base class chung: interface thống nhất cho mọi module hiệu chỉnh
# (giống hệt icnet.ICNet.count_parameters()/measure_latency() để dùng lẫn nhau
# trong run_cost_analysis() của run_c2.py)
# ─────────────────────────────────────────────────────────────────────────────
class BaseCorrectionModule(nn.Module):
    """Interface chung: forward(x) trả ảnh đã hiệu chỉnh (B,3,H,W) trong [0,1]."""

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
# 1. Gamma Correction (cổ điển, KHÔNG có tham số học được)
# ─────────────────────────────────────────────────────────────────────────────
class GammaCorrection(BaseCorrectionModule):
    """
    Gamma Correction thích nghi cổ điển (KHÔNG có tham số học được):
    Với mỗi ảnh trong batch, ước lượng gamma sao cho độ sáng trung bình sau
    hiệu chỉnh tiến gần `target_mean` (mặc định 0.45):
        gamma = log(target_mean) / log(mean_luminance)
        I_out = clamp(I_in, eps, 1) ^ gamma
    """
    def __init__(self, target_mean: float = 0.45, gamma_min: float = 0.3, gamma_max: float = 3.0):
        super().__init__()
        self.target_mean = target_mean
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        # Buffer "giả" để .to(device) hoạt động nhất quán dù không có tham số thật
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
# 2. CLAHE (cổ điển, KHÔNG có tham số học được, KHÔNG khả vi)
# ─────────────────────────────────────────────────────────────────────────────
class CLAHECorrection(BaseCorrectionModule):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization), áp dụng trên kênh L
    của không gian màu Lab (giữ nguyên màu sắc, chỉ tăng tương phản cục bộ).
    KHÔNG có tham số học được, xử lý qua OpenCV per-image trên CPU (không khả vi)
    -- vì vậy latency đo được thường CAO hơn hẳn các module dựa trên CNN/GPU khác,
    đây là đặc điểm thực tế đúng của CLAHE khi triển khai, không phải lỗi.
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
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def build_correction_module(method: str):
    """
    Trả về nn.Module tương ứng với `method` (đã .train() theo mặc định của
    nn.Module), hoặc None nếu method == 'none' (baseline không hiệu chỉnh).
    """
    method = method.lower()
    if method == 'none':
        return None
    if method == 'icnet':
        from icnet import ICNet
        return ICNet()
    if method == 'gamma':
        return GammaCorrection()
    if method == 'clahe':
        return CLAHECorrection()
    raise ValueError(f"Method hiệu chỉnh không hợp lệ: '{method}'. Các lựa chọn hợp lệ: {CORRECTION_METHODS}")


if __name__ == "__main__":
    for m in CORRECTION_METHODS:
        if m == 'none':
            print(f"{m:<14} -> Baseline (không có module)")
            continue
        mod = build_correction_module(m)
        n_params = mod.count_parameters()
        learnable = "Có" if is_learnable_method(m) else "Không"
        print(f"{m:<14} -> {METHOD_DISPLAY_NAMES[m]:<28} | Params: {n_params:>7,} | Học được: {learnable}")