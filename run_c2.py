#!/usr/bin/env python3
"""
run_c2.py — Main CLI Orchestrator cho C2:
  - Illumination-Robust Module (IC-Net) — đề xuất chính của C2
    (Consistency Loss trên cặp ảnh {light, dark} đã bị loại bỏ hoàn toàn khỏi pipeline)
  - So sánh IC-Net với các phương pháp illumination/low-light correction cổ điển,
    KHÔNG học được: Gamma Correction, CLAHE (xem enhancement_methods.py).
    Zero-DCE++/SCI/Retinexformer (bản reimplementation rút gọn) đã bị loại bỏ
    hoàn toàn khỏi so sánh -- xem ghi chú lý do trong enhancement_methods.py.
  - Quality-Gate (huấn luyện độc lập, sweep threshold, session simulation)
  - Ablation nhanh (Fold 1) + Kết quả chính thức 5-Fold (mean±std) cho TỪNG method
  - Protocol B (5-Fold Stratified + Locked Test Set)
  - Đo chi phí tính toán (Params, Latency ms) cho mọi method
  - LR Schedule: Linear Warmup + Cosine Annealing (thay cho LR cố định)

Chạy:
  1. Ablation Stage (Fold 1, kiểm tra nhanh toàn bộ method):
     python run_c2.py --data_dir . --metadata metadata.csv --stage ablation --config all --device 0 --epochs 100
  2. Final Stage (5-Fold, mean±std, bảng so sánh cuối cùng — IC-Net vs Gamma/CLAHE):
     python run_c2.py --data_dir . --metadata metadata.csv --stage final --methods all --device 0 --epochs 100
     # hoặc chỉ định 1 tập con: --methods none,icnet,gamma,clahe
"""

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

# Đảm bảo đường dẫn module luôn trỏ đúng thư mục dự án
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fix UTF-8 encoding trên Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
import yaml
from torch.utils.data import DataLoader

from quality_gate import QualityGate, train_quality_gate, sweep_quality_gate_threshold, simulate_session
from train_c2 import (
    CashVisionDataset,
    cashvision_collate_fn,
    C2DetectionPipeline,
    train_one_epoch_c2,
    validate_one_epoch_c2,
    evaluate_c2_pipeline,
    CLASS_NAMES,
    DENOM_CLASSES,
    TORN_CLASS_ID
)
from enhancement_methods import (
    build_correction_module,
    CORRECTION_METHODS,
    METHOD_DISPLAY_NAMES,
    is_learnable_method
)
from ultralytics import YOLO
from ultralytics.utils.loss import v8DetectionLoss
from augment import run_augmentation


# ─────────────────────────────────────────────────────────────────────────────
# Cố định Seed
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ─────────────────────────────────────────────────────────────────────────────
# Chuẩn hoá Đường dẫn
# ─────────────────────────────────────────────────────────────────────────────
def normalize_device(device_str: str) -> str:
    """
    Chuẩn hoá device string cho PyTorch:
    - 'cpu'       -> 'cpu'
    - '0', '1'... -> 'cuda:0', 'cuda:1'... (PyTorch không chấp nhận chuỗi số trần)
    - 'cuda:0'    -> giữ nguyên
    """
    d = str(device_str).strip().lower()
    if d == 'cpu':
        return 'cpu'
    if d.isdigit():
        return f'cuda:{d}'
    return device_str


def resolve_data_path(data_dir: str):
    if sys.platform != "win32" and ":" in str(data_dir):
        clean_drive = data_dir.replace("\\", "/")
        if Path(clean_drive).exists():
            return Path(clean_drive).resolve()
        elif Path("train").exists():
            return Path(".").resolve()
        else:
            wsl_path = re.sub(r"^([a-zA-Z]):", r"/mnt/\1", clean_drive).lower()
            return Path(wsl_path).resolve()
    return Path(data_dir).resolve()


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: Generalization Dev Set cho việc chọn checkpoint (tách khỏi Locked Test)
# ─────────────────────────────────────────────────────────────────────────────
def carve_generalization_val_set(excluded_df: pd.DataFrame, frac: float = 0.15,
                                  min_per_cond: int = 5, max_per_cond: int = 40,
                                  seed: int = 42):
    """
    Trước đây (bug task 4): tiêu chí chọn best-checkpoint mỗi epoch (monitor_loss
    trong train_and_eval_config) chỉ tính trên val augmented từ ĐÚNG phân phối train
    hẹp (vd chỉ indoor/torn_clean nếu restrict_train_conditions mặc định). Nghĩa là
    "epoch tốt nhất" được chọn là tốt nhất cho phân phối đã thấy lúc train, không
    nhất thiết là tốt nhất để generalize sang điều kiện chưa từng thấy -- trong khi
    đó lại chính là mục tiêu thật sự của thí nghiệm. Module linh hoạt như IC-Net vì
    vậy dễ bị chọn nhầm checkpoint bị overfit vào phân phối train hẹp.

    Hàm này tách một phần NHỎ (mặc định 15%, tối đa max_per_cond ảnh/condition) của
    excluded_df (ảnh THẬT thuộc các condition bị loại khỏi train, vd outdoor/
    backlight/overexposed/torn_bright) ra làm "Generalization Dev Set" -- dùng
    RIÊNG để chọn checkpoint tốt nhất mỗi epoch, KHÔNG bao giờ dùng để báo cáo kết
    quả cuối cùng. Phần còn lại (đa số) vẫn được gộp vào locked_test_df như cũ,
    nên không có leakage vào con số final -- locked_test_df chưa từng bị dùng cho
    bất kỳ quyết định huấn luyện/chọn checkpoint nào.
    """
    empty = excluded_df.iloc[0:0].copy()
    if len(excluded_df) == 0:
        return empty, excluded_df

    rng = np.random.RandomState(seed)
    gen_val_parts, remaining_parts = [], []
    for _, sub in excluded_df.groupby('condition'):
        n = len(sub)
        n_take = int(np.clip(round(n * frac), min(min_per_cond, n), min(max_per_cond, n)))
        if n_take <= 0:
            remaining_parts.append(sub)
            continue
        take_idx = rng.choice(sub.index.values, size=n_take, replace=False)
        gen_val_parts.append(sub.loc[take_idx])
        remaining_parts.append(sub.drop(index=take_idx))

    gen_val_df = pd.concat(gen_val_parts, ignore_index=True) if gen_val_parts else empty
    remaining_df = pd.concat(remaining_parts, ignore_index=True) if remaining_parts else empty
    return gen_val_df, remaining_df


# ─────────────────────────────────────────────────────────────────────────────
# Chuẩn bị Dữ liệu Protocol B
# ─────────────────────────────────────────────────────────────────────────────
def prepare_protocol_b_splits(df: pd.DataFrame, data_dir: Path, results_dir: Path, n_splits: int = 5, seed: int = 42, force_reaugment: bool = False, restrict_train_conditions: list = None):
    """
    Protocol B:
    - Pool K-Fold = toàn bộ thư mục train/ (từ metadata)
    - Test khoá kín (Locked Test) = gộp val/ + test/

    restrict_train_conditions: nếu chỉ định (vd ['indoor', 'torn_clean']), CHỈ giữ lại
    những ảnh thuộc các condition này trong train pool để huấn luyện. Phần ảnh THẬT
    của các condition còn lại (vốn cũng nằm trong train/) sẽ được gộp thêm vào
    locked_test_df thay vì bỏ phí -- mục đích: kiểm tra khả năng model generalize
    sang các điều kiện hoàn toàn KHÔNG được thấy lúc train.
    """
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    # 1. Tách Train Pool vs Locked Test Set dựa trên prefix tên file (chuẩn xác 100%)
    is_train = df['filename'].astype(str).str.startswith('train_')
    train_pool_df = df[is_train].copy().reset_index(drop=True)
    locked_test_df = df[~is_train].copy().reset_index(drop=True)

    print(f"\n[PROTOCOL B DATA SPLIT]")
    print(f"  - K-Fold Train Pool (train/) trước khi lọc: {len(train_pool_df)} ảnh")
    print(f"  - Locked Test Set (val/ + test/) trước khi gộp: {len(locked_test_df)} ảnh")

    # 1b. (Tuỳ chọn) Giới hạn train chỉ còn 1 số condition cụ thể -- phần bị loại
    # được gộp thêm vào locked_test_df thay vì bỏ đi, để test generalization sang
    # điều kiện chưa từng thấy trên tập test lớn nhất có thể.
    gen_val_df = df.iloc[0:0].copy()  # rỗng mặc định nếu không restrict condition nào

    if restrict_train_conditions:
        restrict_set = set(c.strip().lower() for c in restrict_train_conditions)
        cond_lower = train_pool_df['condition'].astype(str).str.lower()
        keep_mask = cond_lower.isin(restrict_set)
        excluded_df = train_pool_df[~keep_mask].copy()
        train_pool_df = train_pool_df[keep_mask].copy().reset_index(drop=True)

        # Task 4: tách 1 phần nhỏ excluded_df làm Generalization Dev Set (chỉ dùng
        # để chọn checkpoint), phần còn lại (đa số) mới gộp vào locked_test_df như cũ.
        gen_val_df, remaining_excluded_df = carve_generalization_val_set(excluded_df, seed=seed)
        locked_test_df = pd.concat([locked_test_df, remaining_excluded_df], ignore_index=True)

        print(f"  ⚠️  [RESTRICT] Giới hạn train chỉ còn condition: {sorted(restrict_set)}")
        print(f"      -> {len(excluded_df)} ảnh (condition khác) bị loại khỏi train")
        print(f"      -> {len(gen_val_df)} ảnh trong số đó tách làm Generalization Dev Set "
              f"(chỉ dùng chọn checkpoint, KHÔNG tính vào kết quả cuối)")
        print(f"      -> {len(remaining_excluded_df)} ảnh còn lại gộp vào locked_test_df")

    print(f"  - K-Fold Train Pool SAU CÙNG: {len(train_pool_df)} ảnh")
    print(f"  - Generalization Dev Set (chọn checkpoint): {len(gen_val_df)} ảnh")
    print(f"  - Locked Test Set SAU CÙNG: {len(locked_test_df)} ảnh")

    # Stratified K-Fold trên Train Pool (stratify theo denomination * condition)
    train_pool_df['strat_key'] = train_pool_df['denomination_class'].astype(str) + "_" + train_pool_df['condition'].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(train_pool_df, train_pool_df['strat_key']), start=1):
        fold_dir = results_path / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_df = train_pool_df.iloc[train_idx].reset_index(drop=True)
        fold_test_df = train_pool_df.iloc[test_idx].reset_index(drop=True)

        # Xuất danh sách train_filelist.txt
        train_list_file = fold_dir / "train_filelist.txt"
        with open(train_list_file, 'w', encoding='utf-8') as f:
            for fname in fold_train_df['filename']:
                f.write(f"{fname}\n")

        # Tự động gọi augmentation cho riêng fold này
        # -> Bỏ qua nếu đã augment sẵn từ lần chạy trước (trừ khi force_reaugment=True)
        aug_out_dir = fold_dir / "augmented"
        combined_meta_csv = aug_out_dir / "combined_metadata_augmented.csv"
        if combined_meta_csv.exists() and not force_reaugment:
            print(f"[FOLD {fold_idx}] Đã có sẵn augmented data tại {combined_meta_csv} -> bỏ qua augment lại.")
        else:
            combined_meta_csv = run_augmentation(
                data_dir=str(data_dir),
                output_dir=str(aug_out_dir),
                image_list=fold_train_df['filename'].tolist(),
                metadata_path=str(results_path.parent / "metadata.csv" if (results_path.parent / "metadata.csv").exists() else "metadata.csv")
            )

        # Augment riêng 20% còn lại của fold (fold_test_df) để dùng làm VAL SET
        # -- CHỈ dùng để chọn checkpoint tốt nhất trong lúc train (early-stopping
        # style), KHÔNG dùng để báo cáo kết quả cuối cùng. Kết quả cuối luôn đến
        # từ locked_test_df (cố định, KHÔNG augment) ở bên ngoài hàm này.
        val_aug_out_dir = fold_dir / "val_augmented"
        val_combined_meta_csv = val_aug_out_dir / "combined_metadata_augmented.csv"
        if val_combined_meta_csv.exists() and not force_reaugment:
            print(f"[FOLD {fold_idx}] Đã có sẵn augmented VAL data tại {val_combined_meta_csv} -> bỏ qua augment lại.")
        else:
            val_combined_meta_csv = run_augmentation(
                data_dir=str(data_dir),
                output_dir=str(val_aug_out_dir),
                image_list=fold_test_df['filename'].tolist(),
                metadata_path=str(results_path.parent / "metadata.csv" if (results_path.parent / "metadata.csv").exists() else "metadata.csv")
            )

        fold_data.append({
            'fold': fold_idx,
            'fold_dir': fold_dir,
            'train_augmented_meta': combined_meta_csv,
            'val_augmented_meta': val_combined_meta_csv,
            'fold_test_df': fold_test_df  # bản RAW (không augment) -- vẫn dùng riêng cho Quality-Gate như cũ
        })

    return fold_data, locked_test_df, gen_val_df


# ─────────────────────────────────────────────────────────────────────────────
# Đo Chi phí Tính toán (Params + Latency) cho TẤT CẢ phương pháp so sánh
# ─────────────────────────────────────────────────────────────────────────────
def run_cost_analysis(device: str, results_dir: Path, methods: list = None, yolo_weights: str = "yolov8n.pt"):
    """
    Đo Params & Latency của TỪNG correction module (IC-Net, Gamma, CLAHE) so
    với baseline YOLO thuần -- dùng cho cột "Params"/"Latency" trong bảng
    so sánh cuối cùng (summary_final_5fold_all_methods_headline.csv). Trả về
    dict {method: (added_params, added_latency_ms)}, trong đó 'none' luôn là
    (0, 0.0).

    Lưu ý: Zero-DCE++/SCI/Retinexformer đã bị loại bỏ hoàn toàn khỏi
    CORRECTION_METHODS (xem enhancement_methods.py) nên không còn xuất hiện
    trong bảng cost analysis này -- không cần lọc/loại thủ công ở đây.
    """
    if methods is None:
        methods = [m for m in CORRECTION_METHODS if m != 'none']

    base_model = YOLO(yolo_weights).model.to(device)
    base_params = sum(p.numel() for p in base_model.parameters())

    base_model.eval()
    dummy = torch.rand(1, 3, 640, 640, device=device)
    with torch.no_grad():
        for _ in range(10):
            _ = base_model(dummy)
        if 'cuda' in str(device) and torch.cuda.is_available():
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                _ = base_model(dummy)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        else:
            t0 = time.perf_counter()
            for _ in range(100):
                _ = base_model(dummy)
            t1 = time.perf_counter()
    base_latency_ms = ((t1 - t0) / 100.0) * 1000.0

    rows = []
    cost_map = {'none': (0, 0.0)}
    for method in methods:
        module = build_correction_module(method)
        if module is None:
            cost_map[method] = (0, 0.0)
            continue
        module = module.to(device)
        n_params = sum(p.numel() for p in module.parameters())
        latency_ms = module.measure_latency(device=device)
        param_pct = (n_params / base_params) * 100.0 if base_params > 0 else 0.0
        latency_pct = (latency_ms / base_latency_ms) * 100.0 if base_latency_ms > 0 else 0.0

        rows.append({
            'module': METHOD_DISPLAY_NAMES.get(method, method),
            'added_parameters': n_params,
            'added_params_pct': f"+{param_pct:.2f}%",
            'added_latency_ms': round(latency_ms, 3),
            'added_latency_pct': f"+{latency_pct:.2f}%",
            'base_model_params': base_params,
            'base_model_latency_ms': round(base_latency_ms, 3)
        })
        cost_map[method] = (n_params, latency_ms)
        del module

    cost_df = pd.DataFrame(rows)
    cost_path = results_dir / "cost_analysis_all_methods.csv"
    cost_df.to_csv(cost_path, index=False)

    print(f"\n[COST ANALYSIS] Đã lưu vào: {cost_path}")
    print(f"  Base {yolo_weights}: {base_params:,} params | {base_latency_ms:.3f} ms/ảnh")
    for _, r in cost_df.iterrows():
        print(f"  - {r['module']:<22}: {r['added_parameters']:>7,} params ({r['added_params_pct']:>8}) | "
              f"{r['added_latency_ms']:>7.3f} ms ({r['added_latency_pct']})")
    print()
    return cost_map


# ─────────────────────────────────────────────────────────────────────────────
# Huấn luyện 1 Method (baseline 'none' hoặc 1 trong các correction module)
# ─────────────────────────────────────────────────────────────────────────────
def _format_mean_std(series: pd.Series) -> str:
    """Định dạng 'mean ± std' qua 5 Fold cho 1 cột chỉ số, bỏ qua NaN (vd
    mAP50_tear ở các condition không có ảnh rách thật). Nếu chỉ có 1 giá trị
    hợp lệ (hiếm, khi force_retrain lệch fold), std = 0.0 thay vì NaN."""
    s = series.dropna()
    if len(s) == 0:
        return "—"
    m = s.mean()
    sd = s.std() if len(s) > 1 else 0.0
    return f"{m:.2f} ± {sd:.2f}"


def train_and_eval_config(
    method: str,
    train_meta_csv: Path,
    eval_df: pd.DataFrame,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    output_dir: Path,
    seed: int = 42,
    force_retrain: bool = False,
    val_meta_csv: Path = None,
    extra_val_df: pd.DataFrame = None,
    lambda_photo: float = 1.0,
    early_stopping_patience: int = 0,
    early_stopping_min_delta: float = 0.0,
    yolo_weights: str = "yolov8n.pt"
):
    output_dir.mkdir(parents=True, exist_ok=True)

    method = method.lower()
    use_correction = (method != 'none')
    is_learnable = is_learnable_method(method)

    # 0. KIỂM TRA ĐÃ CÓ KẾT QUẢ HUẤN LUYỆN CHƯA
    metrics_file = output_dir / "metrics_per_condition.csv"
    weights_file = output_dir / "full_pipeline_weights.pt"
    if metrics_file.exists() and weights_file.exists() and not force_retrain:
        print(f"\n⚡ [CHECKPOINT FOUND] Đã có sẵn kết quả tại: {output_dir} -> Bỏ qua, không train lại!")
        metrics_df = pd.read_csv(metrics_file)
        pipeline = C2DetectionPipeline(weights_path=yolo_weights, correction_method=method).to(device)
        pipeline.load_state_dict(torch.load(weights_file, map_location=device))
        return metrics_df, pipeline

    print(f"\n{'='*75}")
    print(f"🚀 Huấn luyện Method: {METHOD_DISPLAY_NAMES.get(method, method)} "
          f"(Correction: {use_correction} | Học được: {is_learnable})")
    print(f"{'='*75}")

    # 1. Dataset & DataLoader
    # load_clean_ref=is_learnable: chỉ nạp thêm ảnh tham chiếu sạch khi method này
    # có Photometric Loss (chỉ IC-Net, method HỌC ĐƯỢC duy nhất còn lại) --
    # 'none'/gamma/clahe không cần, tránh I/O thừa.
    train_df = pd.read_csv(train_meta_csv)
    train_dataset = CashVisionDataset(train_df, img_size=640, load_clean_ref=is_learnable)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cashvision_collate_fn)

    # 1b. Val DataLoader (tuỳ chọn) -- chỉ dùng để chọn checkpoint tốt nhất mỗi epoch,
    # KHÔNG shuffle (không cần thiết, chỉ để tính loss trung bình).
    #
    # Task 4: gộp thêm extra_val_df (Generalization Dev Set -- ảnh THẬT từ các
    # condition bị loại khỏi train, vd outdoor/backlight/overexposed/torn_bright)
    # vào val_df, để monitor_loss phản ánh đúng mục tiêu generalize thay vì chỉ đo
    # trên đúng phân phối train hẹp. extra_val_df không augment (ảnh thật sẵn có),
    # nên cần đủ các cột giống val_df gốc (img_path/lbl_path/condition/... đã có sẵn
    # từ metadata.csv qua main()).
    val_loader = None
    if val_meta_csv is not None:
        val_df = pd.read_csv(val_meta_csv)
        if extra_val_df is not None and len(extra_val_df) > 0:
            n_before = len(val_df)
            val_df = pd.concat([val_df, extra_val_df], ignore_index=True)
            print(f"  ℹ️  [VAL SET] Gộp thêm {len(extra_val_df)} ảnh Generalization Dev Set "
                  f"vào val set ({n_before} -> {len(val_df)} ảnh) để chọn checkpoint.")
        val_dataset = CashVisionDataset(val_df, img_size=640, load_clean_ref=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=cashvision_collate_fn)

    # 2. Khởi tạo Pipeline
    pipeline = C2DetectionPipeline(weights_path=yolo_weights, correction_method=method).to(device)

    # Optimizer với Param Groups riêng (chỉ thêm param group cho correction module
    # nếu nó thực sự có tham số học được -- Gamma/CLAHE có 0 tham số nên bỏ qua)
    param_groups = []
    param_groups.append({'params': pipeline.detector.parameters(), 'lr': lr, 'weight_decay': 5e-4})
    if is_learnable and pipeline.corrector is not None:
        param_groups.append({'params': pipeline.corrector.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    loss_fn = v8DetectionLoss(pipeline.detector)

    # Lưu LR gốc của từng param group để dùng cho Warmup + Cosine Annealing
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    warmup_epochs = min(3, max(1, epochs // 10))  # Ultralytics mặc định warmup ~3 epoch
    # LR sàn cuối chu kỳ cosine, tính theo % base_lr (không giảm về hẳn 0 để tránh
    # param group của IC-Net/Detector "chết" hoàn toàn ở các epoch cuối).
    min_lr_frac = 0.01

    training_curves = []
    val_curves = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    # Nếu có val_loader: chọn checkpoint theo VAL loss (đúng thông lệ, tránh chọn
    # nhầm epoch bị overfit train). Nếu không truyền val_meta_csv (vd code cũ gọi lại
    # hàm này mà không sửa call site): fallback về hành vi cũ (chọn theo train loss).
    monitor_metric_name = "val_loss" if val_loader is not None else "train_loss"

    # Early Stopping: dừng huấn luyện sớm nếu monitor_loss (val_loss nếu có val_loader,
    # ngược lại train_loss) không cải thiện sau `early_stopping_patience` epoch liên tiếp.
    # - early_stopping_patience <= 0: TẮT early stopping (hành vi mặc định/cũ, chạy đủ
    #   `epochs`), để không phá vỡ các lần gọi hàm này đã có từ trước.
    # - Checkpoint tốt nhất (best_state) vẫn được theo dõi & khôi phục như cũ dù có
    #   dừng sớm hay không -- early stopping chỉ cắt bớt các epoch "vô ích" phía sau,
    #   không ảnh hưởng tới việc chọn checkpoint.
    early_stopping_enabled = early_stopping_patience > 0
    epochs_no_improve = 0
    stopped_early = False

    # 3. Custom Training Loop
    for epoch in range(1, epochs + 1):
        # LR Schedule: Linear Warmup (warmup_epochs đầu) -> Cosine Annealing cho phần
        # còn lại của quá trình huấn luyện (thay cho LR cố định như trước).
        # - Warmup: tuyến tính từ 10% -> 100% base_lr trong warmup_epochs đầu.
        #   Ultralytics Trainer gốc luôn làm bước này tự động; custom loop này thiếu nó,
        #   nhiều khả năng là nguyên nhân gây mất ổn định huấn luyện ở vài epoch đầu.
        # - Cosine Annealing: sau warmup, LR giảm dần theo nửa chu kỳ cosine từ 100%
        #   base_lr xuống còn min_lr_frac * base_lr tại epoch cuối cùng -- giúp hội tụ
        #   mượt hơn LR cố định, đặc biệt quan trọng khi tăng số epochs huấn luyện.
        if epoch <= warmup_epochs:
            warmup_factor = 0.1 + 0.9 * (epoch / warmup_epochs)
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g['lr'] = base_lr * warmup_factor
        else:
            cosine_progress = (epoch - warmup_epochs) / max(1, (epochs - warmup_epochs))
            cosine_factor = 0.5 * (1.0 + math.cos(math.pi * cosine_progress))
            lr_factor = min_lr_frac + (1.0 - min_lr_frac) * cosine_factor
            for g, base_lr in zip(optimizer.param_groups, base_lrs):
                g['lr'] = base_lr * lr_factor

        t0 = time.perf_counter()
        avg_total, avg_det, avg_photo = train_one_epoch_c2(
            pipeline=pipeline,
            dataloader=train_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epoch=epoch,
            lambda_photo=lambda_photo
        )
        t_epoch = time.perf_counter() - t0

        training_curves.append({
            'epoch': epoch,
            'loss_total': round(avg_total, 4),
            'loss_detection': round(avg_det, 4),
            'loss_photo': round(avg_photo, 4) if is_learnable else 0.0,
            'lr_detector': round(optimizer.param_groups[0]['lr'], 8),
            'time_sec': round(t_epoch, 2)
        })

        # Tính Val Loss (nếu có val_loader) -- KHÔNG cập nhật trọng số, chỉ để
        # chọn checkpoint tốt nhất. Đây là tiêu chí "monitor_loss" dùng để so sánh
        # thay vì avg_total (train loss) như code cũ.
        if val_loader is not None:
            val_loss = validate_one_epoch_c2(pipeline, val_loader, loss_fn, device)
            val_curves.append({'epoch': epoch, 'val_loss_detection': round(val_loss, 4)})
            monitor_loss = val_loss
        else:
            monitor_loss = avg_total

        # Lưu lại checkpoint nếu đây là epoch có monitor_loss thấp nhất từ trước đến giờ
        # (val loss nếu có val_loader, ngược lại fallback về train loss như hành vi cũ).
        # QUAN TRỌNG: dùng .detach().clone() cho từng tensor (không phải .copy() nông
        # trên dict, vì .copy() chỉ copy cấu trúc dict, các tensor bên trong vẫn là
        # cùng 1 object -- optimizer.step() sau đó sẽ tiếp tục sửa in-place các tensor
        # này, khiến "best_state" bị ghi đè theo, không còn là snapshot thật sự).
        if monitor_loss < (best_loss - early_stopping_min_delta):
            best_loss = monitor_loss
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in pipeline.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch % 5 == 0 or epoch == epochs:
            photo_str = f", Loss_Photo: {avg_photo:.4f}" if is_learnable else ""
            val_str = f" | Val_Loss: {monitor_loss:.4f}" if val_loader is not None else ""
            es_str = f" | NoImprove: {epochs_no_improve}/{early_stopping_patience}" if early_stopping_enabled else ""
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss_Det: {avg_det:.4f}{photo_str} | Total: {avg_total:.4f}{val_str}{es_str} | LR: {cur_lr:.2e} ({t_epoch:.1f}s)")

        # Dừng sớm nếu monitor_loss không cải thiện đủ số epoch liên tiếp quy định.
        # Đặt SAU log epoch để epoch cuối cùng (epoch dừng) vẫn được in ra bình thường,
        # kể cả khi nó không rơi đúng vào bội số 5.
        if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
            if not (epoch % 5 == 0 or epoch == epochs):
                # Epoch dừng chưa được log ở trên (không phải bội số 5) -> log riêng để
                # người dùng biết chính xác lúc nào dừng, không chỉ suy luận từ số dòng log.
                photo_str = f", Loss_Photo: {avg_photo:.4f}" if is_learnable else ""
                val_str = f" | Val_Loss: {monitor_loss:.4f}" if val_loader is not None else ""
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss_Det: {avg_det:.4f}{photo_str} | Total: {avg_total:.4f}{val_str} | NoImprove: {epochs_no_improve}/{early_stopping_patience} | LR: {cur_lr:.2e} ({t_epoch:.1f}s)")
            print(f"  ⏹️  [EARLY STOPPING] Dừng huấn luyện tại epoch {epoch}/{epochs}: "
                  f"{monitor_metric_name} không cải thiện sau {early_stopping_patience} epoch liên tiếp "
                  f"(tốt nhất: epoch {best_epoch}, {monitor_metric_name}={best_loss:.4f}).")
            stopped_early = True
            break

    # Khôi phục checkpoint tốt nhất ({monitor_metric_name} thấp nhất) trước khi lưu &
    # đánh giá, thay vì luôn dùng epoch cuối cùng (có thể đang ở trạng thái bị nhiễu tạm thời).
    if best_state is not None:
        pipeline.load_state_dict(best_state)
        print(f"  ✅ [BEST CHECKPOINT] Khôi phục epoch {best_epoch} ({monitor_metric_name}={best_loss:.4f}) để lưu & đánh giá.")

    # Lưu training curves (+ val curves nếu có)
    pd.DataFrame(training_curves).to_csv(output_dir / "training_curves.csv", index=False)
    if val_curves:
        pd.DataFrame(val_curves).to_csv(output_dir / "val_curves.csv", index=False)

    # Lưu metadata Early Stopping (để tra cứu sau này: có dừng sớm không, dừng ở đâu,
    # checkpoint tốt nhất là epoch nào) -- không ảnh hưởng logic huấn luyện/đánh giá.
    with open(output_dir / "training_meta.json", "w", encoding="utf-8") as f:
        json.dump({
            "epochs_requested": epochs,
            "epochs_run": len(training_curves),
            "early_stopping_enabled": early_stopping_enabled,
            "early_stopping_patience": early_stopping_patience,
            "stopped_early": stopped_early,
            "monitor_metric": monitor_metric_name,
            "best_epoch": best_epoch,
            "best_loss": round(best_loss, 6) if best_loss != float('inf') else None
        }, f, ensure_ascii=False, indent=2)

    # Lưu weights
    if use_correction and pipeline.corrector is not None:
        torch.save(pipeline.corrector.state_dict(), output_dir / f"{method}_weights.pt")
    torch.save(pipeline.state_dict(), output_dir / "full_pipeline_weights.pt")

    # 4. Đánh giá trên từng Condition
    test_conditions = ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']
    eval_records = []

    print(f"\n🔍 [EVALUATION] Đánh giá Method '{METHOD_DISPLAY_NAMES.get(method, method)}' trên {len(test_conditions)} điều kiện kiểm thử:")
    for cond in test_conditions:
        sub_df = eval_df[eval_df['condition'].str.lower() == cond].reset_index(drop=True)
        if len(sub_df) == 0:
            continue
        m = evaluate_c2_pipeline(pipeline, sub_df, cond, device)
        eval_records.append(m)
        map_str = f"{m['mAP50_tear']:.1f}%" if m['mAP50_tear'] is not None else "N/A"
        print(f"  ▶ {cond:<14} ({len(sub_df):>3} ảnh) | Acc Denom: {m['accuracy_denom']:>5.1f}% | mAP50 Tear: {map_str:>5} | Torn F1: {m['torn_f1']:>5.1f}% | False Alarm: {m['false_alarm_rate']:.1f}%")

    metrics_df = pd.DataFrame(eval_records)
    metrics_df.to_csv(output_dir / "metrics_per_condition.csv", index=False)

    return metrics_df, pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm C2: so sánh IC-Net (đề xuất) với Gamma Correction, "
                                                   "CLAHE + Quality-Gate.")
    parser.add_argument("--data_dir", type=str, default=".", help="Thư mục gốc chứa dữ liệu")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Đường dẫn file metadata.csv")
    parser.add_argument("--e1_config", type=str, default="results/yolov8n/config_used.yaml", help="Config YAML của E1")
    parser.add_argument("--stage", type=str, choices=["ablation", "final"], default="ablation",
                         help="Giai đoạn chạy: ablation (Fold 1, nhanh) hoặc final (5-Fold, mean±std, bảng so sánh cuối cùng)")
    parser.add_argument("--config", type=str, default="all", choices=CORRECTION_METHODS + ["all"],
                         help="Method chạy ở stage ablation (Fold 1). 'all' = chạy toàn bộ "
                              f"{CORRECTION_METHODS} (mặc định: all)")
    parser.add_argument("--methods", type=str, default="all",
                         help="Danh sách method (phân cách dấu phẩy) chạy ở stage final (5-Fold, mean±std). "
                              f"'all' = chạy toàn bộ: {','.join(CORRECTION_METHODS)}. "
                              "Ví dụ: --methods none,icnet,gamma,clahe")
    parser.add_argument("--lambda_photo", type=float, default=1.0, help="Hệ số trọng số Photometric Supervision Loss cho các module HỌC ĐƯỢC (mặc định: 1.0)")
    parser.add_argument("--gate_threshold_sweep", action="store_true", default=True, help="Chạy huấn luyện và sweep ngưỡng cho Quality-Gate")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (đã tăng mặc định 50 -> 100, dùng cùng Cosine Annealing LR Schedule)")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                         help="Số epoch liên tiếp không cải thiện val_loss (hoặc train_loss nếu không có val_loader) "
                              "trước khi dừng huấn luyện sớm. <=0 = TẮT early stopping (mặc định, chạy đủ --epochs).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                         help="Ngưỡng cải thiện tối thiểu để tính là 'có cải thiện' khi early stopping đang bật (mặc định: 0.0)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed cố định")
    parser.add_argument("--results_dir", type=str, default="results_c2", help="Thư mục lưu kết quả")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                         help="YOLO model weights (vd: yolov8n.pt, yolov8s.pt, runs/detect/train/weights/best.pt)")
    parser.add_argument("--force_reaugment", action="store_true", default=False, help="Bắt buộc augment lại dù đã có sẵn dữ liệu augmented cho fold")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Bắt buộc huấn luyện lại dù đã có sẵn weights và metrics của config/fold đó")
    parser.add_argument("--restrict_train_conditions", type=str, default="indoor,torn_clean",
                         help="Danh sách condition (phân cách bởi dấu phẩy) được PHÉP dùng để train, "
                              "vd: 'indoor,torn_clean'. Mặc định: 'indoor,torn_clean' để chỉ huấn luyện trên ảnh "
                              "chuẩn (indoor + torn_clean) và kiểm tra khả năng tổng quát hóa (generalization) "
                              "sang toàn bộ các điều kiện ánh sáng khó (outdoor, backlight, overexposed, torn_bright). "
                              "Truyền 'all' hoặc '' nếu muốn dùng toàn bộ 1260 ảnh trong train/.")

    args = parser.parse_args()
    set_seed(args.seed)

    data_path = resolve_data_path(args.data_dir)
    results_path = Path(args.results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    device = '0' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    device = normalize_device(device)
    print(f"🚀 [INIT] Khởi động C2 Pipeline trên thiết bị: {device}")

    # Đọc metadata
    meta_file = Path(args.metadata).resolve()
    if not meta_file.exists() and (data_path / args.metadata).exists():
        meta_file = (data_path / args.metadata).resolve()
    df_meta = pd.read_csv(meta_file)

    # Map đường dẫn thực tế cho ảnh (chỉ quét các thư mục gốc train/valid/test)
    img_map = {}
    lbl_map = {}
    valid_split_dirs = [data_path / s for s in ['train', 'valid', 'test', 'val'] if (data_path / s).exists()]
    if not valid_split_dirs:
        valid_split_dirs = [data_path]

    for s_dir in valid_split_dirs:
        for root, _, files in os.walk(s_dir):
            for f in files:
                p = Path(root) / f
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img_map[f] = str(p)
                    img_map[p.name] = str(p)
                elif p.suffix.lower() == '.txt' and f != 'classes.txt':
                    lbl_map[p.stem] = str(p)

    df_meta['img_path'] = df_meta['filename'].map(img_map)
    df_meta['lbl_path'] = df_meta['filename'].apply(lambda x: lbl_map.get(Path(x).stem))

    # Kiểm tra sớm: báo lỗi rõ ràng nếu có ảnh trong metadata.csv không tìm thấy trên đĩa,
    # thay vì để lỗi mập mờ xảy ra sau ở prepare_protocol_b_splits().
    n_missing = df_meta['img_path'].isna().sum()
    if n_missing > 0:
        missing_examples = df_meta.loc[df_meta['img_path'].isna(), 'filename'].head(5).tolist()
        raise FileNotFoundError(
            f"[LỖI DATA] {n_missing}/{len(df_meta)} ảnh trong metadata.csv không tìm thấy file thực tế "
            f"trong data_dir='{data_path}'. Ví dụ filename bị thiếu: {missing_examples}. "
            f"Kiểm tra lại --data_dir có đúng thư mục chứa train/valid/test không."
        )

    # 1. Chuẩn bị splits Protocol B & chạy Augmentation tự động cho từng fold
    restrict_conditions = None
    if args.restrict_train_conditions and args.restrict_train_conditions.lower() not in ['all', 'none', '']:
        restrict_conditions = [c.strip() for c in args.restrict_train_conditions.split(",") if c.strip()]

    fold_data, locked_test_df, gen_val_df = prepare_protocol_b_splits(
        df_meta, data_path, results_path, n_splits=5, seed=args.seed,
        force_reaugment=args.force_reaugment,
        restrict_train_conditions=restrict_conditions
    )

    # 2. Đo Chi phí Tính toán (Params + Latency) cho TẤT CẢ phương pháp so sánh
    cost_map = run_cost_analysis(device, results_path, yolo_weights=args.model)

    # 3. Quality-Gate: Huấn luyện, Sweep Threshold & Mô phỏng phiên
    if args.gate_threshold_sweep:
        q_weights = results_path / "quality_gate_weights.pt"
        q_sweep_csv = results_path / "quality_gate_threshold_sweep.csv"
        if q_weights.exists() and q_sweep_csv.exists() and not args.force_retrain:
            print("\n⚡ [QUALITY-GATE FOUND] Đã có sẵn weights & threshold sweep của Quality Gate -> Bỏ qua, không train lại.")
        else:
            print("\n" + "=" * 75)
            print("🛡️ QUALITY-GATE: HUẤN LUYỆN & QUÉT NGƯỠNG TAU")
            print("=" * 75)
            gate_train_df = fold_data[0]['fold_test_df']  # Dùng train của fold 1
            q_model = train_quality_gate(gate_train_df, fold_data[0]['fold_test_df'], device=device, epochs=20)
            torch.save(q_model.state_dict(), q_weights)

            sweep_quality_gate_threshold(q_model, fold_data[0]['fold_test_df'], device=device,
                                         output_csv=str(q_sweep_csv))
            simulate_session(q_model, locked_test_df, tau=0.6, device=device,
                             output_csv=str(results_path / "quality_gate_session_simulation.csv"))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: ABLATION nhanh trên Fold 1 (kiểm tra sơ bộ trước khi chạy 5-Fold)
    # ─────────────────────────────────────────────────────────────────────────
    if args.stage == "ablation":
        print("\n" + "=" * 85)
        print("📊 GIAI ĐOẠN 1: ABLATION CÁC METHOD HIỆU CHỈNH (Chạy trên Fold 1)")
        print("=" * 85)

        methods_to_run = CORRECTION_METHODS if args.config == "all" else [args.config]
        fold1 = fold_data[0]
        ablation_summary = []

        for method in methods_to_run:
            cfg_out_dir = results_path / f"config_{method}"
            m_df, _ = train_and_eval_config(
                method=method,
                train_meta_csv=fold1['train_augmented_meta'],
                val_meta_csv=fold1['val_augmented_meta'],
                eval_df=locked_test_df,
                epochs=args.epochs,
                batch_size=args.batch,
                lr=args.lr,
                device=device,
                output_dir=cfg_out_dir,
                seed=args.seed,
                force_retrain=args.force_retrain,
                extra_val_df=gen_val_df,
                lambda_photo=args.lambda_photo,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
                yolo_weights=args.model
            )

            params_m, latency_m = cost_map.get(method, (0, 0.0))
            for _, row in m_df.iterrows():
                ablation_summary.append({
                    'method': METHOD_DISPLAY_NAMES.get(method, method),
                    'correction': "Yes" if method != 'none' else "No",
                    'condition': row['condition'],
                    'accuracy_denom': row['accuracy_denom'],
                    'mAP50_tear': row['mAP50_tear'] if row['mAP50_tear'] is not None else "—",
                    'mAP50_95_tear': row['mAP50_95_tear'] if row['mAP50_95_tear'] is not None else "—",
                    'torn_f1': row['torn_f1'],
                    'avg_IoU_tear': row['avg_IoU_tear'] if row['avg_IoU_tear'] is not None else "—",
                    'added_params': params_m,
                    'added_latency_ms': round(latency_m, 3)
                })

        summary_ablation_df = pd.DataFrame(ablation_summary)
        ablation_csv = results_path / "summary_ablation_all_methods.csv"
        summary_ablation_df.to_csv(ablation_csv, index=False)

        print("\n" + "=" * 130)
        print("📋 BẢNG TỔNG HỢP ABLATION CÁC METHOD (STAGE 1 - FOLD 1)")
        print("=" * 130)
        print(f"{'Method':<26} | {'Condition':<14} | {'Acc Denom (%)':<15} | {'mAP50 Tear':<12} | {'Torn F1':<9} | {'Params':<10} | {'Latency'}")
        print("-" * 130)
        for _, r in summary_ablation_df.iterrows():
            print(f"{r['method']:<26} | {r['condition']:<14} | {r['accuracy_denom']:<15} | {str(r['mAP50_tear']):<12} | {r['torn_f1']:<9} | {str(r['added_params']):<10} | {r['added_latency_ms']} ms")
        print("=" * 130)
        print(f"📁 Đã lưu bảng Ablation vào: {ablation_csv}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: KẾT QUẢ CHÍNH THỨC 5-FOLD — SO SÁNH TOÀN BỘ METHOD (MEAN ± STD)
    # (Dánh giá trên Locked Test Set; đây là bảng cuối cùng
    #  so sánh IC-Net với Gamma Correction, CLAHE)
    # ─────────────────────────────────────────────────────────────────────────
    elif args.stage == "final":
        methods_to_run = CORRECTION_METHODS if args.methods.lower() == "all" \
            else [m.strip().lower() for m in args.methods.split(",") if m.strip()]

        unknown_methods = [m for m in methods_to_run if m not in CORRECTION_METHODS]
        if unknown_methods:
            raise ValueError(f"[LỖI] --methods chứa method không hợp lệ: {unknown_methods}. "
                              f"Các lựa chọn hợp lệ: {CORRECTION_METHODS}")

        print("\n" + "=" * 85)
        print(f"🏆 GIAI ĐOẠN 2: KẾT QUẢ CHÍNH THỨC 5-FOLD — SO SÁNH {len(methods_to_run)} METHOD")
        print(f"   {', '.join(METHOD_DISPLAY_NAMES.get(m, m) for m in methods_to_run)}")
        print(f"   (Đánh giá trên Locked Test Set: {len(locked_test_df)} ảnh)")
        print("=" * 85)

        all_methods_fold_metrics = []

        for method in methods_to_run:
            print(f"\n{'#'*90}")
            print(f"# METHOD: {METHOD_DISPLAY_NAMES.get(method, method)} ('{method}')")
            print(f"{'#'*90}")

            all_fold_metrics = []
            for f_item in fold_data:
                f_idx = f_item['fold']
                print(f"\n--- [{METHOD_DISPLAY_NAMES.get(method, method)} | Fold {f_idx}/5] ---")
                cfg_out_dir = results_path / f"final_config_{method}" / f"fold_{f_idx}"
                m_df, _ = train_and_eval_config(
                    method=method,
                    train_meta_csv=f_item['train_augmented_meta'],
                    val_meta_csv=f_item['val_augmented_meta'],
                    eval_df=locked_test_df,
                    epochs=args.epochs,
                    batch_size=args.batch,
                    lr=args.lr,
                    device=device,
                    output_dir=cfg_out_dir,
                    seed=args.seed + f_idx,
                    force_retrain=args.force_retrain,
                    extra_val_df=gen_val_df,
                    lambda_photo=args.lambda_photo,
                    early_stopping_patience=args.early_stopping_patience,
                    early_stopping_min_delta=args.early_stopping_min_delta,
                    yolo_weights=args.model
                )
                m_df['fold'] = f_idx
                m_df['method'] = method
                all_fold_metrics.append(m_df)

            method_full_df = pd.concat(all_fold_metrics, ignore_index=True)
            method_full_df.to_csv(results_path / f"final_{method}_all_folds_raw.csv", index=False)
            all_methods_fold_metrics.append(method_full_df)

        full_df = pd.concat(all_methods_fold_metrics, ignore_index=True)
        full_df.to_csv(results_path / "final_all_methods_all_folds_raw.csv", index=False)

        # ─── Tổng hợp Mean ± Std qua 5 Folds, cho MỖI method × MỖI condition ───
        # Cột gồm: các chỉ số ĐÃ CÓ SẴN trong evaluate_c2_pipeline (accuracy_denom,
        # banknote_avg_iou/mAP50/mAP50-95/miss_rate, binary_tear_acc, false_alarm_rate,
        # avg_IoU_tear) + các chỉ số MỚI theo yêu cầu: Params, Latency, Torn F1, Torn mAP
        # (Torn mAP = mAP50_tear / mAP50_95_tear, đổi tên hiển thị cho rõ nghĩa).
        headline_summary = []
        for method in methods_to_run:
            params_m, latency_m = cost_map.get(method, (0, 0.0))
            sub_method = full_df[full_df['method'] == method]

            for cond in ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']:
                sub = sub_method[sub_method['condition'] == cond]
                if len(sub) == 0:
                    continue

                has_tears = bool(sub['has_real_tears'].iloc[0])

                headline_summary.append({
                    'method': METHOD_DISPLAY_NAMES.get(method, method),
                    'condition': cond,
                    'accuracy_denom (%)': _format_mean_std(sub['accuracy_denom']),
                    'banknote_avg_iou (%)': _format_mean_std(sub['banknote_avg_iou']),
                    'banknote_mAP50 (%)': _format_mean_std(sub['banknote_map50']),
                    'banknote_mAP50-95 (%)': _format_mean_std(sub['banknote_map50_95']),
                    'banknote_miss_rate (%)': _format_mean_std(sub['banknote_miss_rate']),
                    'binary_tear_acc (%)': _format_mean_std(sub['binary_tear_acc']),
                    'false_alarm_rate (%)': _format_mean_std(sub['false_alarm_rate']),
                    'Torn_F1 (%)': _format_mean_std(sub['torn_f1']),
                    'Torn_mAP50 (%)': _format_mean_std(sub['mAP50_tear']) if has_tears else "—",
                    'Torn_mAP50-95 (%)': _format_mean_std(sub['mAP50_95_tear']) if has_tears else "—",
                    'Torn_avg_IoU (%)': _format_mean_std(sub['avg_IoU_tear']) if has_tears else "—",
                    'Params (added)': f"{params_m:,}" if method != 'none' else "0 (baseline)",
                    'Latency_ms (added)': f"{latency_m:.3f}" if method != 'none' else "0.000 (baseline)",
                })

        headline_df = pd.DataFrame(headline_summary)
        headline_csv = results_path / "summary_final_5fold_all_methods_headline.csv"
        headline_df.to_csv(headline_csv, index=False)

        print("\n" + "=" * 150)
        print("📊 BẢNG KẾT QUẢ CHÍNH THỨC 5-FOLD — SO SÁNH IC-NET VỚI CÁC PHƯƠNG PHÁP KHÁC (MEAN ± STD)")
        print("=" * 150)
        print(f"{'Method':<24} | {'Condition':<12} | {'Acc Denom (%)':<15} | {'Torn F1 (%)':<15} | {'Torn mAP50 (%)':<17} | {'False Alarm (%)':<17} | {'Params':<10} | {'Latency (ms)'}")
        print("-" * 150)
        for _, r in headline_df.iterrows():
            print(f"{r['method']:<24} | {r['condition']:<12} | {r['accuracy_denom (%)']:<15} | {r['Torn_F1 (%)']:<15} | "
                  f"{r['Torn_mAP50 (%)']:<17} | {r['false_alarm_rate (%)']:<17} | {r['Params (added)']:<10} | {r['Latency_ms (added)']}")
        print("=" * 150)
        print(f"📁 Đã lưu bảng đầy đủ (tất cả cột) vào: {headline_csv}")
        print(f"📁 Dữ liệu thô từng fold/method vào: {results_path / 'final_all_methods_all_folds_raw.csv'}\n")


if __name__ == "__main__":
    main()