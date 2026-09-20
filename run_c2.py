#!/usr/bin/env python3
"""
run_c2.py — Main CLI Orchestrator for C2:
  - Illumination-Robust Module (MQTone) — primary proposed module for C2
    (Consistency Loss on {light, dark} pairs has been completely removed)
  - Comparison with: IAT (Illumination-Adaptive Transformer), Zero-DCE,
    Zero-DCE++, Gamma Correction, CLAHE, RetinexNet (Enhance-Net style),
    EnlightenGAN (Generator-only) — see enhancement_methods.py.
  - Quality-Gate (independent training, threshold sweep, session simulation)
  - Fast Ablation (Fold 1) + Official 5-Fold benchmark (mean±std) for each method
  - Protocol B (5-Fold Stratified + Locked Test Set)
  - Computational cost profiling (Params, Latency ms) for all methods
  - LR Schedule: Linear Warmup + Cosine Annealing (replacing constant LR)

Usage:
  1. Ablation Stage (Fold 1, rapid verification across methods):
     python run_c2.py --data_dir . --metadata metadata.csv --stage ablation --config mqtone --device 0 --epochs 100
  2. Final Stage (5-Fold, mean±std, benchmark comparison — MQTone vs Baselines):
     python run_c2.py --data_dir . --metadata metadata.csv --stage final --methods mqtone,iat,none --device 0 --epochs 100
     # or run all registered methods: --methods all
     # including retinexnet and enlightengan: --methods retinexnet,enlightengan
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

# Ensure module paths point to project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Fix UTF-8 encoding on Windows
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
# Fix random seed
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
# Normalize paths
# ─────────────────────────────────────────────────────────────────────────────
def normalize_device(device_str: str) -> str:
    """
    Normalize device string for PyTorch:
    - 'cpu'       -> 'cpu'
    - '0', '1'... -> 'cuda:0', 'cuda:1'... (PyTorch requires valid device string)
    - 'cuda:0'    -> unchanged
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
# Task 4: Generalization Dev Set for checkpoint selection (separated from Locked Test)
# ─────────────────────────────────────────────────────────────────────────────
def carve_generalization_val_set(excluded_df: pd.DataFrame, frac: float = 0.15,
                                  min_per_cond: int = 5, max_per_cond: int = 40,
                                  seed: int = 42):
    """
    Checkpoint selection criterion (monitor_loss) on a narrow training distribution
    risks selecting a checkpoint that overfits to seen conditions rather than
    generalizing to unseen adverse conditions.

    This function splits a SMALL subset (default 15%, max max_per_cond images/condition)
    from excluded_df (real images from adverse conditions excluded from training,
    e.g., outdoor/backlight/overexposed/torn_bright) as a "Generalization Dev Set"
    -- used SOLELY to select the best checkpoint per epoch, and NEVER used to report
    final results. The remaining images merge into locked_test_df, ensuring zero
    leakage into final reported metrics.
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
# Prepare Data for Protocol B
# ─────────────────────────────────────────────────────────────────────────────
def prepare_protocol_b_splits(df: pd.DataFrame, data_dir: Path, results_dir: Path, n_splits: int = 5, seed: int = 42, force_reaugment: bool = False, restrict_train_conditions: list = None):
    """
    Protocol B:
    - K-Fold Pool = all images from train/ directory (from metadata)
    - Locked Test = merged val/ + test/ sets

    restrict_train_conditions: if specified (e.g. ['indoor', 'torn_clean']), ONLY images
    belonging to these conditions are kept in the training pool. Real images from other
    conditions are merged into locked_test_df -- allowing evaluation of out-of-distribution
    generalization to unseen adverse conditions.
    """
    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    # 1. Split Train Pool vs Locked Test Set based on filename prefix
    is_train = df['filename'].astype(str).str.startswith('train_')
    train_pool_df = df[is_train].copy().reset_index(drop=True)
    locked_test_df = df[~is_train].copy().reset_index(drop=True)

    print(f"\n[PROTOCOL B DATA SPLIT]")
    print(f"  - K-Fold Train Pool (train/) before filtering: {len(train_pool_df)} images")
    print(f"  - Locked Test Set (val/ + test/) before merging: {len(locked_test_df)} images")

    # 1b. (Optional) Restrict training to specified conditions -- excluded real images
    # are merged into locked_test_df to test generalization across unseen conditions.
    gen_val_df = df.iloc[0:0].copy()  # empty by default if no restriction

    if restrict_train_conditions:
        restrict_set = set(c.strip().lower() for c in restrict_train_conditions)
        cond_lower = train_pool_df['condition'].astype(str).str.lower()
        keep_mask = cond_lower.isin(restrict_set)
        excluded_df = train_pool_df[~keep_mask].copy()
        train_pool_df = train_pool_df[keep_mask].copy().reset_index(drop=True)

        # Split small subset of excluded_df for Generalization Dev Set (used only
        # for checkpoint selection), while the remaining majority merges into locked_test_df.
        gen_val_df, remaining_excluded_df = carve_generalization_val_set(excluded_df, seed=seed)
        locked_test_df = pd.concat([locked_test_df, remaining_excluded_df], ignore_index=True)

        print(f"  ⚠️  [RESTRICT] Restricting training pool to conditions: {sorted(restrict_set)}")
        print(f"      -> {len(excluded_df)} images (other conditions) excluded from train")
        print(f"      -> {len(gen_val_df)} images reserved for Generalization Dev Set "
              f"(checkpoint selection only, NOT included in final test metrics)")
        print(f"      -> {len(remaining_excluded_df)} remaining images merged into locked_test_df")

    print(f"  - Final K-Fold Train Pool: {len(train_pool_df)} images")
    print(f"  - Generalization Dev Set (checkpoint selection): {len(gen_val_df)} images")
    print(f"  - Final Locked Test Set: {len(locked_test_df)} images")

    # Stratified K-Fold on Train Pool (stratified by denomination * condition)
    train_pool_df['strat_key'] = train_pool_df['denomination_class'].astype(str) + "_" + train_pool_df['condition'].astype(str)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_data = []

    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(train_pool_df, train_pool_df['strat_key']), start=1):
        fold_dir = results_path / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        fold_train_df = train_pool_df.iloc[train_idx].reset_index(drop=True)
        fold_test_df = train_pool_df.iloc[test_idx].reset_index(drop=True)

        # Export train_filelist.txt
        train_list_file = fold_dir / "train_filelist.txt"
        with open(train_list_file, 'w', encoding='utf-8') as f:
            for fname in fold_train_df['filename']:
                f.write(f"{fname}\n")

        # Run data augmentation for this specific fold
        # -> Skip if already augmented (unless force_reaugment=True)
        aug_out_dir = fold_dir / "augmented"
        combined_meta_csv = aug_out_dir / "combined_metadata_augmented.csv"
        if combined_meta_csv.exists() and not force_reaugment:
            print(f"[FOLD {fold_idx}] Augmented data already exists at {combined_meta_csv} -> skipping augmentation.")
        else:
            combined_meta_csv = run_augmentation(
                data_dir=str(data_dir),
                output_dir=str(aug_out_dir),
                image_list=fold_train_df['filename'].tolist(),
                metadata_path=str(results_path.parent / "metadata.csv" if (results_path.parent / "metadata.csv").exists() else "metadata.csv")
            )

        # Augment remaining 20% validation split (fold_test_df) for VAL SET
        # -- Used ONLY for checkpoint selection during training (early stopping style),
        # NOT for reporting final results. Final metrics always come from locked_test_df.
        val_aug_out_dir = fold_dir / "val_augmented"
        val_combined_meta_csv = val_aug_out_dir / "combined_metadata_augmented.csv"
        if val_combined_meta_csv.exists() and not force_reaugment:
            print(f"[FOLD {fold_idx}] Augmented VAL data already exists at {val_combined_meta_csv} -> skipping augmentation.")
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
            'fold_test_df': fold_test_df  # RAW split (unaugmented) -- used for Quality-Gate
        })

    return fold_data, locked_test_df, gen_val_df


# ─────────────────────────────────────────────────────────────────────────────
# Measure Computational Cost (Params + Latency) for ALL Methods
# ─────────────────────────────────────────────────────────────────────────────
def run_cost_analysis(device: str, results_dir: Path, methods: list = None, yolo_weights: str = "yolov8n.pt"):
    """
    Measure added Params & Latency of each correction module relative to
    plain YOLO baseline -- used for Params and Latency columns in final tables.
    Returns dict {method: (added_params, added_latency_ms)}, where 'none' is (0, 0.0).
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
    results_dir.mkdir(parents=True, exist_ok=True)
    cost_path = results_dir / "cost_analysis_all_methods.csv"
    cost_df.to_csv(cost_path, index=False)

    print(f"\n[COST ANALYSIS] Saved to: {cost_path}")
    print(f"  Base {yolo_weights}: {base_params:,} params | {base_latency_ms:.3f} ms/image")
    for _, r in cost_df.iterrows():
        print(f"  - {r['module']:<22}: {r['added_parameters']:>7,} params ({r['added_params_pct']:>8}) | "
              f"{r['added_latency_ms']:>7.3f} ms ({r['added_latency_pct']})")
    print()
    return cost_map


# ─────────────────────────────────────────────────────────────────────────────
# Train 1 Method (baseline 'none' or one of the correction modules)
# ─────────────────────────────────────────────────────────────────────────────
def _format_mean_std(series: pd.Series) -> str:
    """Format 'mean ± std' across 5 Folds for a metric column, ignoring NaN
    (e.g., mAP50_tear for conditions without real tears)."""
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

    # 0. CHECK IF TRAINING RESULTS ALREADY EXIST
    metrics_file = output_dir / "metrics_per_condition.csv"
    weights_file = output_dir / "full_pipeline_weights.pt"
    if metrics_file.exists() and weights_file.exists() and not force_retrain:
        print(f"\n⚡ [CHECKPOINT FOUND] Results already exist at: {output_dir} -> Skipping training!")
        metrics_df = pd.read_csv(metrics_file)
        pipeline = C2DetectionPipeline(weights_path=yolo_weights, correction_method=method).to(device)
        pipeline.load_state_dict(torch.load(weights_file, map_location=device))
        return metrics_df, pipeline

    print(f"\n{'='*75}")
    print(f"🚀 Training Method: {METHOD_DISPLAY_NAMES.get(method, method)} "
          f"(Correction: {use_correction} | Learnable: {is_learnable})")
    print(f"{'='*75}")

    # 1. Dataset & DataLoader
    # load_clean_ref=is_learnable: only loads clean reference images when method
    # has Photometric Loss (MQTone, IAT, Zero-DCE, Zero-DCE++) --
    # 'none'/gamma/clahe do not need it, avoiding extra I/O.
    train_df = pd.read_csv(train_meta_csv)
    train_dataset = CashVisionDataset(train_df, img_size=640, load_clean_ref=is_learnable)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=cashvision_collate_fn)

    # 1b. Val DataLoader (optional) -- used only for selecting best checkpoint per epoch,
    # un-shuffled (computes average validation loss).
    #
    # Task 4: concatenates extra_val_df (Generalization Dev Set -- real images from
    # conditions excluded from training) into val_df for generalization tracking.
    # extra_val_df is unaugmented real images, containing required columns
    # (img_path/lbl_path/condition) passed from metadata.csv.
    val_loader = None
    if val_meta_csv is not None:
        val_df = pd.read_csv(val_meta_csv)
        if extra_val_df is not None and len(extra_val_df) > 0:
            n_before = len(val_df)
            val_df = pd.concat([val_df, extra_val_df], ignore_index=True)
            print(f"  ℹ️  [VAL SET] Added {len(extra_val_df)} Generalization Dev Set images "
                  f"into val set ({n_before} -> {len(val_df)} images) for checkpoint selection.")
        val_dataset = CashVisionDataset(val_df, img_size=640, load_clean_ref=False)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=cashvision_collate_fn)

    # 2. Initialize Pipeline
    pipeline = C2DetectionPipeline(weights_path=yolo_weights, correction_method=method).to(device)

    # Optimizer with separate Param Groups (only includes correction module
    # if learnable -- non-learnable modules like Gamma/CLAHE have 0 params)
    param_groups = []
    param_groups.append({'params': pipeline.detector.parameters(), 'lr': lr, 'weight_decay': 5e-4})
    if is_learnable and pipeline.corrector is not None:
        param_groups.append({'params': pipeline.corrector.parameters(), 'lr': 1e-3, 'weight_decay': 1e-4})

    optimizer = torch.optim.AdamW(param_groups)
    loss_fn = v8DetectionLoss(pipeline.detector)

    # Store base LR of each param group for Warmup + Cosine Annealing
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    warmup_epochs = min(3, max(1, epochs // 10))  # Ultralytics default warmup ~3 epochs
    # Minimum floor LR at end of cosine cycle (percentage of base_lr).
    min_lr_frac = 0.01

    training_curves = []
    val_curves = []
    best_loss = float('inf')
    best_state = None
    best_epoch = -1
    # If val_loader is available: select checkpoint by VAL loss (standard practice).
    # Fallback to train loss if val_loader is omitted.
    monitor_metric_name = "val_loss" if val_loader is not None else "train_loss"

    # Early Stopping: halts training early if monitor_loss does not improve
    # for `early_stopping_patience` consecutive epochs.
    # - <= 0: DISABLED (runs all epochs).
    # - Best checkpoint is restored regardless of early termination.
    early_stopping_enabled = early_stopping_patience > 0
    epochs_no_improve = 0
    stopped_early = False

    # 3. Custom Training Loop
    for epoch in range(1, epochs + 1):
        # LR Schedule: Linear Warmup -> Cosine Annealing.
        # - Warmup: linearly ramps from 10% to 100% base_lr over warmup_epochs.
        # - Cosine Annealing: smoothly decays LR from base_lr to min_lr_frac * base_lr.
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

        # Compute Val Loss (if val_loader available) -- NO weight update, used solely
        # to track best checkpoint.
        if val_loader is not None:
            val_loss = validate_one_epoch_c2(pipeline, val_loader, loss_fn, device)
            val_curves.append({'epoch': epoch, 'val_loss_detection': round(val_loss, 4)})
            monitor_loss = val_loss
        else:
            monitor_loss = avg_total

        # Save checkpoint if current epoch achieves lowest monitor_loss.
        # Uses .detach().clone() for every tensor to maintain an isolated snapshot.
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

        # Early stopping check after epoch logging.
        if early_stopping_enabled and epochs_no_improve >= early_stopping_patience:
            if not (epoch % 5 == 0 or epoch == epochs):
                # Log final stopped epoch if not already printed on a multiple of 5.
                photo_str = f", Loss_Photo: {avg_photo:.4f}" if is_learnable else ""
                val_str = f" | Val_Loss: {monitor_loss:.4f}" if val_loader is not None else ""
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss_Det: {avg_det:.4f}{photo_str} | Total: {avg_total:.4f}{val_str} | NoImprove: {epochs_no_improve}/{early_stopping_patience} | LR: {cur_lr:.2e} ({t_epoch:.1f}s)")
            print(f"  ⏹️  [EARLY STOPPING] Stopping training at epoch {epoch}/{epochs}: "
                  f"{monitor_metric_name} did not improve for {early_stopping_patience} consecutive epochs "
                  f"(best: epoch {best_epoch}, {monitor_metric_name}={best_loss:.4f}).")
            stopped_early = True
            break

    # Restore best checkpoint (lowest monitor_loss) before saving and evaluation.
    if best_state is not None:
        pipeline.load_state_dict(best_state)
        print(f"  ✅ [BEST CHECKPOINT] Restored epoch {best_epoch} ({monitor_metric_name}={best_loss:.4f}) for saving & evaluation.")

    # Save training curves (+ val curves if available)
    pd.DataFrame(training_curves).to_csv(output_dir / "training_curves.csv", index=False)
    if val_curves:
        pd.DataFrame(val_curves).to_csv(output_dir / "val_curves.csv", index=False)

    # Save Early Stopping metadata for diagnostic inspection.
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

    # Save weights
    if use_correction and pipeline.corrector is not None:
        torch.save(pipeline.corrector.state_dict(), output_dir / f"{method}_weights.pt")
    torch.save(pipeline.state_dict(), output_dir / "full_pipeline_weights.pt")

    # 4. Evaluate on each Condition
    test_conditions = ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']
    eval_records = []

    print(f"\n🔍 [EVALUATION] Evaluating Method '{METHOD_DISPLAY_NAMES.get(method, method)}' across {len(test_conditions)} test conditions:")
    for cond in test_conditions:
        sub_df = eval_df[eval_df['condition'].str.lower() == cond].reset_index(drop=True)
        if len(sub_df) == 0:
            continue
        m = evaluate_c2_pipeline(pipeline, sub_df, cond, device)
        eval_records.append(m)
        map_str = f"{m['mAP50_tear']:.1f}%" if m['mAP50_tear'] is not None else "N/A"
        print(f"  ▶ {cond:<14} ({len(sub_df):>3} imgs) | Acc Denom: {m['accuracy_denom']:>5.1f}% | mAP50 Tear: {map_str:>5} | Torn F1: {m['torn_f1']:>5.1f}% | False Alarm: {m['false_alarm_rate']:.1f}%")

    metrics_df = pd.DataFrame(eval_records)
    metrics_df.to_csv(output_dir / "metrics_per_condition.csv", index=False)

    return metrics_df, pipeline


# ─────────────────────────────────────────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Run experiment C2: benchmark MQTone against Gamma Correction, "
                                                   "CLAHE + Quality-Gate.")
    parser.add_argument("--data_dir", type=str, default=".", help="Root dataset directory")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--e1_config", type=str, default="results/yolov8n/config_used.yaml", help="E1 YAML configuration")
    parser.add_argument("--stage", type=str, choices=["ablation", "final"], default="ablation",
                        help="Execution stage: ablation (Fold 1, fast) or final (5-Fold, mean±std benchmark)")
    parser.add_argument("--config", type=str, default="all", choices=CORRECTION_METHODS + ["all"],
                        help="Method to run in ablation stage (Fold 1). 'all' = run all "
                             f"{CORRECTION_METHODS} (default: all)")
    parser.add_argument("--methods", type=str, default="all",
                        help="Comma-separated list of methods to run in final stage (5-Fold, mean±std). "
                             f"'all' = run all: {','.join(CORRECTION_METHODS)}. "
                             "Example: --methods none,mqtone,gamma,clahe")
    parser.add_argument("--lambda_photo", type=float, default=1.0, help="Weight lambda for Photometric Supervision Loss for learnable modules (default: 1.0)")
    parser.add_argument("--gate_threshold_sweep", action="store_true", default=True, help="Train and sweep threshold for Quality-Gate")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs (default: 100 with Cosine Annealing LR Schedule)")
    parser.add_argument("--early_stopping_patience", type=int, default=0,
                        help="Consecutive epochs without val_loss improvement before early stopping. <=0 disables it (default).")
    parser.add_argument("--early_stopping_min_delta", type=float, default=0.0,
                        help="Minimum loss decrease required to qualify as improvement when early stopping is active (default: 0.0)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--results_dir", type=str, default="results_c2", help="Directory to save results")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="YOLO model weights (e.g.: yolov8n.pt, yolov8s.pt, runs/detect/train/weights/best.pt)")
    parser.add_argument("--force_reaugment", action="store_true", default=False, help="Force re-running data augmentation even if augmented data exists")
    parser.add_argument("--force_retrain", action="store_true", default=False, help="Force retraining even if existing weights and metrics are found")
    parser.add_argument("--restrict_train_conditions", type=str, default="indoor,torn_clean",
                        help="Comma-separated conditions permitted for training pool, "
                             "e.g., 'indoor,torn_clean' to evaluate generalization "
                             "to adverse conditions (outdoor, backlight, overexposed, torn_bright). "
                             "Pass 'all' to use all images in train/.")
    args = parser.parse_args()
    set_seed(args.seed)

    data_path = resolve_data_path(args.data_dir)
    results_path = Path(args.results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    device = '0' if (args.device == 'auto' and torch.cuda.is_available()) else args.device
    device = normalize_device(device)
    print(f"🚀 [INIT] Starting C2 Pipeline on device: {device}")

    # Load metadata
    meta_file = Path(args.metadata).resolve()
    if not meta_file.exists() and (data_path / args.metadata).exists():
        meta_file = (data_path / args.metadata).resolve()
    df_meta = pd.read_csv(meta_file)

    # Map actual file paths for images (scanning train/valid/test root directories)
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

    # Early validation: fail fast if images in metadata.csv are missing on disk.
    n_missing = df_meta['img_path'].isna().sum()
    if n_missing > 0:
        missing_examples = df_meta.loc[df_meta['img_path'].isna(), 'filename'].head(5).tolist()
        raise FileNotFoundError(
            f"[DATA ERROR] {n_missing}/{len(df_meta)} images in metadata.csv not found on disk "
            f"under data_dir='{data_path}'. Sample missing files: {missing_examples}. "
            f"Please verify --data_dir points to the folder containing train/valid/test."
        )

    # 1. Prepare Protocol B splits & run automatic augmentation per fold
    restrict_conditions = None
    if args.restrict_train_conditions and args.restrict_train_conditions.lower() not in ['all', 'none', '']:
        restrict_conditions = [c.strip() for c in args.restrict_train_conditions.split(",") if c.strip()]

    fold_data, locked_test_df, gen_val_df = prepare_protocol_b_splits(
        df_meta, data_path, results_path, n_splits=5, seed=args.seed,
        force_reaugment=args.force_reaugment,
        restrict_train_conditions=restrict_conditions
    )

    # 2. Measure Computational Costs (Params + Latency) for ALL comparison methods
    cost_map = run_cost_analysis(device, results_path, yolo_weights=args.model)

    # 3. Quality-Gate: Training, Threshold Sweep & Session Simulation
    if args.gate_threshold_sweep:
        q_weights = results_path / "quality_gate_weights.pt"
        q_sweep_csv = results_path / "quality_gate_threshold_sweep.csv"
        if q_weights.exists() and q_sweep_csv.exists() and not args.force_retrain:
            print("\n⚡ [QUALITY-GATE FOUND] Pretrained weights and threshold sweep found for Quality Gate -> Skipping training.")
        else:
            print("\n" + "=" * 75)
            print("🛡️ QUALITY-GATE: TRAINING & SWEEPING TAU THRESHOLD")
            print("=" * 75)
            gate_train_df = fold_data[0]['fold_test_df']  # Use split from fold 1
            q_model = train_quality_gate(gate_train_df, fold_data[0]['fold_test_df'], device=device, epochs=20)
            torch.save(q_model.state_dict(), q_weights)

            sweep_quality_gate_threshold(q_model, fold_data[0]['fold_test_df'], device=device,
                                         output_csv=str(q_sweep_csv))
            simulate_session(q_model, locked_test_df, tau=0.6, device=device,
                             output_csv=str(results_path / "quality_gate_session_simulation.csv"))

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 1: Fast Ablation on Fold 1 (preliminary verification before 5-Fold)
    # ─────────────────────────────────────────────────────────────────────────
    if args.stage == "ablation":
        print("\n" + "=" * 85)
        print("📊 STAGE 1: CORRECTION METHOD ABLATION (Running on Fold 1)")
        print("=" * 85)

        methods_to_run = CORRECTION_METHODS if args.config == "all" else [('mqtone' if args.config == 'icnet' else args.config)]
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
        print("📋 METHOD ABLATION SUMMARY TABLE (STAGE 1 - FOLD 1)")
        print("=" * 130)
        print(f"{'Method':<26} | {'Condition':<14} | {'Acc Denom (%)':<15} | {'mAP50 Tear':<12} | {'Torn F1':<9} | {'Params':<10} | {'Latency'}")
        print("-" * 130)
        for _, r in summary_ablation_df.iterrows():
            print(f"{r['method']:<26} | {r['condition']:<14} | {r['accuracy_denom']:<15} | {str(r['mAP50_tear']):<12} | {r['torn_f1']:<9} | {str(r['added_params']):<10} | {r['added_latency_ms']} ms")
        print("=" * 130)
        print(f"📁 Ablation table saved to: {ablation_csv}\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STAGE 2: OFFICIAL 5-FOLD RESULTS — BENCHMARKING ALL METHODS (MEAN ± STD)
    # (Evaluated on Locked Test Set: MQTone vs Gamma, CLAHE, IAT, Zero-DCE, etc.)
    # ─────────────────────────────────────────────────────────────────────────
    elif args.stage == "final":
        raw_methods = CORRECTION_METHODS if args.methods.lower() == "all" \
            else [m.strip().lower() for m in args.methods.split(",") if m.strip()]
        methods_to_run = [('mqtone' if m == 'icnet' else m) for m in raw_methods]

        unknown_methods = [m for m in methods_to_run if m not in CORRECTION_METHODS]
        if unknown_methods:
            raise ValueError(f"[ERROR] --methods contains invalid methods: {unknown_methods}. "
                              f"Valid options: {CORRECTION_METHODS}")

        print("\n" + "=" * 85)
        print(f"🏆 STAGE 2: OFFICIAL 5-FOLD RESULTS — BENCHMARKING {len(methods_to_run)} METHODS")
        print(f"   {', '.join(METHOD_DISPLAY_NAMES.get(m, m) for m in methods_to_run)}")
        print(f"   (Evaluated on Locked Test Set: {len(locked_test_df)} images)")
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

        # ─── Aggregate Mean ± Std across 5 Folds, for EACH method x EACH condition ───
        # Columns include: accuracy_denom, banknote_avg_iou/mAP50/mAP50-95/miss_rate,
        # binary_tear_acc, false_alarm_rate, avg_IoU_tear, Params, Latency, Torn F1, Torn mAP.
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
        print("📊 OFFICIAL 5-FOLD RESULTS TABLE — BENCHMARKING MQTONE AGAINST BASELINES (MEAN ± STD)")
        print("=" * 150)
        print(f"{'Method':<24} | {'Condition':<12} | {'Acc Denom (%)':<15} | {'Torn F1 (%)':<15} | {'Torn mAP50 (%)':<17} | {'False Alarm (%)':<17} | {'Params':<10} | {'Latency (ms)'}")
        print("-" * 150)
        for _, r in headline_df.iterrows():
            print(f"{r['method']:<24} | {r['condition']:<12} | {r['accuracy_denom (%)']:<15} | {r['Torn_F1 (%)']:<15} | "
                  f"{r['Torn_mAP50 (%)']:<17} | {r['false_alarm_rate (%)']:<17} | {r['Params (added)']:<10} | {r['Latency_ms (added)']}")
        print("=" * 150)
        print(f"📁 Full table (all columns) saved to: {headline_csv}")
        print(f"📁 Raw per-fold/method metrics saved to: {results_path / 'final_all_methods_all_folds_raw.csv'}\n")


if __name__ == "__main__":
    main()