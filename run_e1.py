#!/usr/bin/env python3
"""
run_e1.py — Experiments E1 & E3: Problem Characterization & Task Degradation Analysis
Quantifying the impact of illumination conditions on:
  (a) Polymer banknote denomination classification
  (b) Tear detection and localization

Runs on GPU/CUDA with Ultralytics YOLO (yolov8n, yolo11n, ...)
Standard Protocol A (5-Fold Cross-Validation):
  - Standard indoor training pool (Normal/Indoor): 'indoor' (intact) + 'torn_clean' (clean torn) -> 5-Fold Stratified
  - Challenging illumination evaluation sets (Test):
      + 'outdoor' (outdoor illumination)
      + 'backlight' (strong backlight)
      + 'overexposed' (severe overexposure intact)
      + 'torn_bright' (torn + severe glare)
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from pathlib import Path

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
from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# Class configurations
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['10', '100', '20', '200', '50', '500', 'torn']
DENOM_CLASSES = {0: '10k', 1: '100k', 2: '20k', 3: '200k', 4: '50k', 5: '500k'}
DENOM_NAME_TO_ID = {
    '10k': 0, '10': 0, '10000': 0,
    '100k': 1, '100': 1, '100000': 1,
    '20k': 2, '20': 2, '20000': 2,
    '200k': 3, '200': 3, '200000': 3,
    '50k': 4, '50': 4, '50000': 4,
    '500k': 5, '500': 5, '500000': 5
}
TORN_CLASS_ID = 6

# ─────────────────────────────────────────────────────────────────────────────
# Set random seed
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
# IoU & Bounding Box utilities
# ─────────────────────────────────────────────────────────────────────────────
def xywh_to_xyxy(box):
    cx, cy, w, h = box
    return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]


def box_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    b1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    b2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = b1_area + b2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


# ─────────────────────────────────────────────────────────────────────────────
# Load and validate metadata
# ─────────────────────────────────────────────────────────────────────────────
def load_and_validate_metadata(metadata_path: Path, data_dir: Path):
    if not metadata_path.exists():
        raise FileNotFoundError(f"❌ Metadata file not found: '{metadata_path}'!")

    df = pd.read_csv(metadata_path)
    required_cols = {'filename', 'condition', 'is_torn'}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"❌ metadata.csv missing required columns. Required: {required_cols}")

    df['condition'] = df['condition'].astype(str).str.strip().str.lower()
    df['filename'] = df['filename'].astype(str).str.strip()
    df['is_torn'] = df['is_torn'].astype(str).str.strip().str.lower()

    # Locate actual files (only scanning train/valid/test root directories)
    img_map = {}
    lbl_map = {}
    valid_split_dirs = [data_dir / s for s in ['train', 'valid', 'test', 'val'] if (data_dir / s).exists()]
    if not valid_split_dirs:
        valid_split_dirs = [data_dir]

    for s_dir in valid_split_dirs:
        for root, _, files in os.walk(s_dir):
            for file in files:
                p = Path(root) / file
                if p.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    img_map[file] = p
                    img_map[p.name] = p
                elif p.suffix.lower() == '.txt' and file != 'classes.txt':
                    lbl_map[p.stem] = p

    df['img_path'] = df['filename'].map(img_map)
    df['lbl_path'] = df['filename'].apply(lambda x: lbl_map.get(Path(x).stem))

    missing_imgs = df[df['img_path'].isna()]
    if len(missing_imgs) > 0:
        raise FileNotFoundError(f"❌ Found {len(missing_imgs)} images in metadata not found in '{data_dir}'!")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Prepare K-Fold Splits for Indoor Pool (indoor + torn_clean)
# ─────────────────────────────────────────────────────────────────────────────
def prepare_kfold_dataset(df: pd.DataFrame, experiment_dir: Path, n_splits: int = 5, seed: int = 42):
    """
    Separate 'indoor' (intact) and 'torn_clean' (clean torn) as the standard training pool.
    Split into 5-Fold Stratified based on combinations of (denomination + torn status).
    """
    dataset_root = experiment_dir / "protocol_a_kfold"
    dataset_root.mkdir(parents=True, exist_ok=True)

    # 1. Indoor Train/Val pool
    normal_df = df[df['condition'].isin(['indoor', 'torn_clean'])].copy().reset_index(drop=True)
    if len(normal_df) == 0:
        raise ValueError("❌ No indoor or torn_clean images found for training!")

    # Stratified key
    normal_df['strat_key'] = normal_df['denomination_class'].astype(str) + "_" + normal_df['is_torn'].astype(str)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_configs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(normal_df, normal_df['strat_key']), start=1):
        fold_dir = dataset_root / f"fold_{fold_idx}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_df = normal_df.iloc[train_idx].reset_index(drop=True)
        val_df = normal_df.iloc[val_idx].reset_index(drop=True)

        # Save images & labels for Train & Val of this fold
        for split_name, split_data in [('train', train_df), ('val_indoor', val_df)]:
            img_out = fold_dir / split_name / "images"
            lbl_out = fold_dir / split_name / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for _, row in split_data.iterrows():
                dest_img = img_out / Path(row['filename']).name
                if not dest_img.exists():
                    shutil.copy2(row['img_path'], dest_img)
                dest_lbl = lbl_out / f"{Path(row['filename']).stem}.txt"
                if pd.notna(row['lbl_path']) and Path(row['lbl_path']).exists():
                    if not dest_lbl.exists():
                        shutil.copy2(row['lbl_path'], dest_lbl)
                else:
                    dest_lbl.touch()

        # Generate dataset.yaml file for this fold
        fold_yaml = fold_dir / "dataset.yaml"
        yaml_dict = {
            'path': str(fold_dir.resolve()).replace('\\', '/'),
            'train': 'train/images',
            'val': 'val_indoor/images',
            'nc': len(CLASS_NAMES),
            'names': CLASS_NAMES
        }
        with open(fold_yaml, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_dict, f, sort_keys=False)

        fold_configs.append({
            'fold': fold_idx,
            'yaml_path': fold_yaml,
            'train_df': train_df,
            'val_df': val_df
        })

    # 2. Prepare test sets under adverse illumination (shared across folds)
    test_splits = {}
    test_root = dataset_root / "test_sets"
    test_root.mkdir(parents=True, exist_ok=True)

    for cond in ['outdoor', 'backlight', 'overexposed', 'torn_bright']:
        cond_df = df[df['condition'] == cond].reset_index(drop=True)
        if len(cond_df) > 0:
            test_splits[cond] = cond_df
            img_out = test_root / cond / "images"
            lbl_out = test_root / cond / "labels"
            img_out.mkdir(parents=True, exist_ok=True)
            lbl_out.mkdir(parents=True, exist_ok=True)

            for _, row in cond_df.iterrows():
                dest_img = img_out / Path(row['filename']).name
                if not dest_img.exists():
                    shutil.copy2(row['img_path'], dest_img)
                dest_lbl = lbl_out / f"{Path(row['filename']).stem}.txt"
                if pd.notna(row['lbl_path']) and Path(row['lbl_path']).exists():
                    if not dest_lbl.exists():
                        shutil.copy2(row['lbl_path'], dest_lbl)
                else:
                    dest_lbl.touch()

            # Evaluation YAML
            test_yaml = test_root / cond / "dataset_eval.yaml"
            t_yaml_dict = {
                'path': str((test_root / cond).resolve()).replace('\\', '/'),
                'val': 'images',
                'nc': len(CLASS_NAMES),
                'names': CLASS_NAMES
            }
            with open(test_yaml, 'w', encoding='utf-8') as f:
                yaml.dump(t_yaml_dict, f, sort_keys=False)

    print(f"\n[DATASET] Successfully created {n_splits}-Fold Cross-Validation:")
    print(f"  - Standard training pool: {len(normal_df)} images (indoor: 390 + torn_clean: 210)")
    print(f"  - Per fold: Train ~ {len(fold_configs[0]['train_df'])}, Val Baseline ~ {len(fold_configs[0]['val_df'])}")
    for cond, c_df in test_splits.items():
        print(f"  - Test set '{cond}': {len(c_df)} images")

    return fold_configs, test_splits


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive evaluation by condition
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_condition(model: YOLO, split_df: pd.DataFrame, condition_name: str, device: str, conf_thresh: float = 0.25):
    predictions_raw = []
    
    # ── Task 1: Banknote (Denomination & Localization) ──────────────────
    y_true_denom = []
    y_pred_denom = []
    denom_ious = []
    total_gt_banknotes = 0
    matched_gt_banknotes = 0
    missed_banknotes = 0
    denom_conf_matrix = np.zeros((6, 6), dtype=int)

    # ── Task 2: Tear (Binary Classification & Localization) ──────────
    n_total_imgs = len(split_df)
    n_gt_torn_imgs = 0
    n_gt_intact_imgs = 0
    
    tp_tear_img = 0  # GT torn, Pred torn
    fp_tear_img = 0  # GT intact, Pred torn (False Alarm)
    fn_tear_img = 0  # GT torn, Pred intact (Missed tear image)
    tn_tear_img = 0  # GT intact, Pred intact

    total_gt_tear_boxes = 0
    matched_gt_tear_boxes = 0
    tear_box_ious = []

    for _, row in split_df.iterrows():
        img_path = str(row['img_path'])
        filename = row['filename']
        is_torn_gt = (str(row['is_torn']).lower() == 'true')

        if is_torn_gt:
            n_gt_torn_imgs += 1
        else:
            n_gt_intact_imgs += 1

        gt_denom_id = None
        gt_banknote_box = None
        gt_tear_boxes = []
        lbl_path = row['lbl_path']

        if pd.notna(lbl_path) and Path(lbl_path).exists():
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    cid = int(parts[0])
                    coords = [float(x) for x in parts[1:5]]
                    if cid == TORN_CLASS_ID:
                        gt_tear_boxes.append(xywh_to_xyxy(coords))
                    elif cid in DENOM_CLASSES:
                        gt_denom_id = cid
                        gt_banknote_box = xywh_to_xyxy(coords)

        if gt_denom_id is not None:
            y_true_denom.append(gt_denom_id)
            total_gt_banknotes += 1

        total_gt_tear_boxes += len(gt_tear_boxes)

        # ── Run Inference ─────────────────────────────────────────────────
        results = model.predict(img_path, conf=conf_thresh, device=device, verbose=False)[0]

        pred_boxes_data = []
        best_denom_id = None
        best_denom_conf = -1.0
        best_banknote_box = None
        pred_tear_boxes = []

        for b in results.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            xyxyn = b.xyxyn[0].tolist()

            pred_boxes_data.append({
                'class_id': cid,
                'class_name': CLASS_NAMES[cid] if cid < len(CLASS_NAMES) else str(cid),
                'confidence': round(conf, 4),
                'bbox_xyxyn': [round(x, 4) for x in xyxyn]
            })

            if cid in DENOM_CLASSES:
                if conf > best_denom_conf:
                    best_denom_conf = conf
                    best_denom_id = cid
                    best_banknote_box = xyxyn
            elif cid == TORN_CLASS_ID:
                pred_tear_boxes.append((xyxyn, conf))

        has_pred_tear = (len(pred_tear_boxes) > 0)

        # ── 1. Evaluate Banknote (Classification + Box IoU) ───────────────
        if gt_denom_id is not None:
            if best_denom_id is not None:
                y_pred_denom.append(best_denom_id)
                denom_conf_matrix[gt_denom_id, best_denom_id] += 1
                
                # Compute IoU of banknote bounding box
                if gt_banknote_box is not None and best_banknote_box is not None:
                    b_iou = box_iou(gt_banknote_box, best_banknote_box)
                    denom_ious.append(b_iou)
                    if b_iou >= 0.5:
                        matched_gt_banknotes += 1
            else:
                y_pred_denom.append(-1)
                missed_banknotes += 1

        # ── 2. Binary Tear Evaluation (Image-level Confusion Matrix) ──────
        if is_torn_gt and has_pred_tear:
            tp_tear_img += 1
        elif (not is_torn_gt) and has_pred_tear:
            fp_tear_img += 1  # False alarm
        elif is_torn_gt and (not has_pred_tear):
            fn_tear_img += 1  # Missed detection
        else:
            tn_tear_img += 1

        # ── 3. Evaluate Tear Box (Box-level IoU matching) ────────────────
        if len(gt_tear_boxes) > 0:
            for gt_box in gt_tear_boxes:
                best_iou = 0.0
                for pred_box, _ in pred_tear_boxes:
                    iou = box_iou(gt_box, pred_box)
                    if iou > best_iou:
                        best_iou = iou
                if best_iou >= 0.5:
                    matched_gt_tear_boxes += 1
                    tear_box_ious.append(best_iou)

        predictions_raw.append({
            'filename': filename,
            'condition': condition_name,
            'gt_denom': DENOM_CLASSES.get(gt_denom_id, None),
            'gt_is_torn': is_torn_gt,
            'gt_tear_count': len(gt_tear_boxes),
            'pred_denom': DENOM_CLASSES.get(best_denom_id, None),
            'pred_has_tear': has_pred_tear,
            'pred_denom_conf': round(best_denom_conf, 4) if best_denom_conf > 0 else 0.0,
            'predictions': pred_boxes_data
        })

    # ── COMPUTE METRICS FOR TASK 1: BANKNOTE ──────────────────────────
    if len(y_true_denom) > 0:
        correct_denom = sum(1 for yt, yp in zip(y_true_denom, y_pred_denom) if yt == yp)
        denom_accuracy = (correct_denom / len(y_true_denom)) * 100.0
    else:
        denom_accuracy = 0.0

    banknote_avg_iou = (np.mean(denom_ious) * 100.0) if len(denom_ious) > 0 else 0.0
    banknote_map50 = (matched_gt_banknotes / max(1, total_gt_banknotes)) * 100.0 if total_gt_banknotes > 0 else 0.0
    banknote_map50_95 = banknote_avg_iou * (banknote_map50 / 100.0)
    banknote_miss_rate = (missed_banknotes / max(1, total_gt_banknotes)) * 100.0 if total_gt_banknotes > 0 else 0.0

    # ── COMPUTE METRICS FOR TASK 2: TEAR ──────────────────────────────
    binary_tear_acc = ((tp_tear_img + tn_tear_img) / max(1, n_total_imgs)) * 100.0
    tear_precision = (tp_tear_img / max(1, (tp_tear_img + fp_tear_img))) * 100.0 if (tp_tear_img + fp_tear_img) > 0 else 0.0
    tear_recall = (tp_tear_img / max(1, (tp_tear_img + fn_tear_img))) * 100.0 if (tp_tear_img + fn_tear_img) > 0 else 0.0
    tear_f1 = (2 * tear_precision * tear_recall / (tear_precision + tear_recall)) if (tear_precision + tear_recall) > 0 else 0.0
    false_alarm_rate = (fp_tear_img / max(1, n_gt_intact_imgs)) * 100.0 if n_gt_intact_imgs > 0 else 0.0

    # Tear Bounding Box Metrics
    has_real_tears = (total_gt_tear_boxes > 0)
    tear_box_avg_iou = (np.mean(tear_box_ious) * 100.0) if len(tear_box_ious) > 0 else 0.0
    tear_map50 = (matched_gt_tear_boxes / total_gt_tear_boxes) * 100.0 if has_real_tears else 0.0
    tear_map50_95 = tear_box_avg_iou * (tear_map50 / 100.0) if has_real_tears else 0.0
    tear_box_miss_rate = ((total_gt_tear_boxes - matched_gt_tear_boxes) / total_gt_tear_boxes) * 100.0 if has_real_tears else 0.0

    metrics = {
        'condition': condition_name,
        'num_samples': n_total_imgs,
        'has_real_tears': has_real_tears,
        
        # Task 1: Banknote
        'accuracy_denom': round(denom_accuracy, 2),
        'banknote_avg_iou': round(banknote_avg_iou, 2),
        'banknote_map50': round(banknote_map50, 2),
        'banknote_map50_95': round(banknote_map50_95, 2),
        'banknote_miss_rate': round(banknote_miss_rate, 2),

        # Task 2: Tear
        'binary_tear_acc': round(binary_tear_acc, 2),
        'tear_precision': round(tear_precision, 2),
        'tear_recall': round(tear_recall, 2),
        'tear_f1': round(tear_f1, 2),
        'false_alarm_rate': round(false_alarm_rate, 2),
        'tear_map50': round(tear_map50, 2) if has_real_tears else 0.0,
        'tear_map50_95': round(tear_map50_95, 2) if has_real_tears else 0.0,
        'tear_box_avg_iou': round(tear_box_avg_iou, 2) if has_real_tears else 0.0,
        'tear_box_miss_rate': round(tear_box_miss_rate, 2) if has_real_tears else 0.0
    }

    return metrics, predictions_raw, denom_conf_matrix


# ─────────────────────────────────────────────────────────────────────────────
# Main Routine for Experiments E1 & E3 K-Fold
# ─────────────────────────────────────────────────────────────────────────────
def run_experiment_e1_kfold(data_dir: str, metadata: str, models: list, n_splits: int, seed: int, epochs: int, batch_size: int, device: str, results_dir: str):
    set_seed(seed)

    # Normalize paths
    if sys.platform != "win32" and ":" in str(data_dir):
        clean_drive = data_dir.replace("\\", "/")
        if Path(clean_drive).exists():
            data_path = Path(clean_drive).resolve()
        elif Path("train").exists():
            print(f"[INFO] Setting working directory to current path: '{Path('.').resolve()}'")
            data_path = Path(".").resolve()
        else:
            wsl_path = re.sub(r"^([a-zA-Z]):", r"/mnt/\1", clean_drive).lower()
            data_path = Path(wsl_path).resolve()
    else:
        data_path = Path(data_dir).resolve()

    metadata_path = Path(metadata).resolve()
    if not metadata_path.exists() and (data_path / metadata).exists():
        metadata_path = (data_path / metadata).resolve()

    results_path = Path(results_dir).resolve()
    results_path.mkdir(parents=True, exist_ok=True)

    if device == 'auto':
        device = '0' if torch.cuda.is_available() else 'cpu'

    print(f"🚀 [INIT] Launching Experiment E1 (5-Fold Cross-Validation) on device: {device}")

    # Prepare K-Fold splits
    df = load_and_validate_metadata(metadata_path, data_path)
    fold_configs, test_splits = prepare_kfold_dataset(df, results_path, n_splits=n_splits, seed=seed)

    denom_summary_rows = []
    tear_summary_rows = []
    e3_comparison_rows = []

    for model_name in models:
        print("\n" + "=" * 80)
        print(f"🔥 TRAINING AND EVALUATING MODEL: {model_name} (across {n_splits} Folds)")
        print("=" * 80)

        model_out_dir = results_path / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        # Store results per fold
        fold_condition_records = []

        for f_conf in fold_configs:
            f_idx = f_conf['fold']
            print(f"\n--- [Model: {model_name} | Fold {f_idx}/{n_splits}] ---")

            weights_name = f"{model_name}.pt" if not model_name.endswith('.pt') else model_name
            model = YOLO(weights_name)

            train_results = model.train(
                data=str(f_conf['yaml_path']),
                epochs=epochs,
                batch=batch_size,
                seed=seed + f_idx,
                device=device,
                project=str(model_out_dir),
                name=f"fold_{f_idx}",
                exist_ok=True,
                verbose=False
            )

            best_pt = model_out_dir / f"fold_{f_idx}" / "weights" / "best.pt"
            eval_model = YOLO(str(best_pt)) if best_pt.exists() else model

            # Evaluate this fold on Indoor Val (Baseline)
            m_indoor, preds_in, _ = evaluate_condition(eval_model, f_conf['val_df'], 'indoor_baseline', device)
            m_indoor['fold'] = f_idx
            m_indoor['delta_acc(%)'] = 0.0
            m_indoor['delta_mAP50(%)'] = 0.0
            fold_condition_records.append(m_indoor)

            baseline_acc = m_indoor['accuracy_denom']
            baseline_map = m_indoor.get('tear_map50', 0.0)

            # Evaluate across Test conditions
            for cond_name, cond_df in test_splits.items():
                m_cond, preds_cond, cm_cond = evaluate_condition(eval_model, cond_df, cond_name, device)
                m_cond['fold'] = f_idx
                m_cond['delta_acc(%)'] = round(m_cond['accuracy_denom'] - baseline_acc, 2)
                if m_cond['has_real_tears'] and baseline_map is not None:
                    m_cond['delta_mAP50(%)'] = round(m_cond['tear_map50'] - baseline_map, 2)
                else:
                    m_cond['delta_mAP50(%)'] = None
                fold_condition_records.append(m_cond)

        # Save raw metrics per fold
        f_df = pd.DataFrame(fold_condition_records)
        f_df.to_csv(model_out_dir / "all_folds_raw_metrics.csv", index=False)

        # ── 1. Aggregate TABLE: BANKNOTE DENOMINATION & LOCALIZATION (5-Fold) ─────────────
        for cond in ['indoor_baseline', 'outdoor', 'backlight', 'overexposed', 'torn_bright']:
            sub = f_df[f_df['condition'] == cond]
            if len(sub) == 0:
                continue

            acc_m, acc_s = sub['accuracy_denom'].mean(), sub['accuracy_denom'].std()
            d_acc_m, d_acc_s = sub['delta_acc(%)'].mean(), sub['delta_acc(%)'].std()
            map50_m, map50_s = sub['banknote_map50'].mean(), sub['banknote_map50'].std()
            map95_m, map95_s = sub['banknote_map50_95'].mean(), sub['banknote_map50_95'].std()
            iou_m, iou_s = sub['banknote_avg_iou'].mean(), sub['banknote_avg_iou'].std()
            miss_m, miss_s = sub['banknote_miss_rate'].mean(), sub['banknote_miss_rate'].std()

            denom_summary_rows.append({
                'model': model_name,
                'condition': cond,
                'accuracy_denom': f"{acc_m:.2f} ± {acc_s:.2f}",
                'delta_acc(%)': f"{d_acc_m:+.2f} ± {d_acc_s:.2f}" if cond != 'indoor_baseline' else "baseline",
                'banknote_mAP50': f"{map50_m:.2f} ± {map50_s:.2f}",
                'banknote_mAP50_95': f"{map95_m:.2f} ± {map95_s:.2f}",
                'banknote_avg_iou': f"{iou_m:.2f} ± {iou_s:.2f}",
                'banknote_miss_rate': f"{miss_m:.2f} ± {miss_s:.2f}",
                'raw_acc_mean': acc_m,
                'raw_delta_acc': d_acc_m
            })

        # ── 2. Aggregate TABLE: TEAR DETECTION & LOCALIZATION (5-Fold) ───────
        for cond in ['indoor_baseline', 'torn_bright', 'outdoor', 'backlight', 'overexposed']:
            sub = f_df[f_df['condition'] == cond]
            if len(sub) == 0:
                continue

            bin_acc_m, bin_acc_s = sub['binary_tear_acc'].mean(), sub['binary_tear_acc'].std()
            prec_m, prec_s = sub['tear_precision'].mean(), sub['tear_precision'].std()
            rec_m, rec_s = sub['tear_recall'].mean(), sub['tear_recall'].std()
            f1_m, f1_s = sub['tear_f1'].mean(), sub['tear_f1'].std()
            fa_m, fa_s = sub['false_alarm_rate'].mean(), sub['false_alarm_rate'].std()

            has_tears = sub['has_real_tears'].iloc[0]

            if has_tears:
                map_m, map_s = sub['tear_map50'].mean(), sub['tear_map50'].std()
                map95_m, map95_s = sub['tear_map50_95'].mean(), sub['tear_map50_95'].std()
                iou_m, iou_s = sub['tear_box_avg_iou'].mean(), sub['tear_box_avg_iou'].std()
                miss_m, miss_s = sub['tear_box_miss_rate'].mean(), sub['tear_box_miss_rate'].std()

                map_str = f"{map_m:.2f} ± {map_s:.2f}"
                map95_str = f"{map95_m:.2f} ± {map95_s:.2f}"
                iou_str = f"{iou_m:.2f} ± {iou_s:.2f}"
                miss_str = f"{miss_m:.2f} ± {miss_s:.2f}"
                prec_str = f"{prec_m:.2f} ± {prec_s:.2f}"
                rec_str = f"{rec_m:.2f} ± {rec_s:.2f}"
                f1_str = f"{f1_m:.2f} ± {f1_s:.2f}"
            else:
                map_str = "—"
                map95_str = "—"
                iou_str = "—"
                miss_str = "—"
                prec_str = "—"
                rec_str = "—"
                f1_str = "—"
                map_m = None

            tear_summary_rows.append({
                'model': model_name,
                'condition': cond,
                'data_type': "Torn" if has_tears else "Intact",
                'binary_tear_acc': f"{bin_acc_m:.2f} ± {bin_acc_s:.2f}",
                'tear_precision': prec_str,
                'tear_recall': rec_str,
                'tear_f1': f1_str,
                'false_alarm_rate': f"{fa_m:.2f} ± {fa_s:.2f}",
                'tear_mAP50': map_str,
                'tear_mAP50_95': map95_str,
                'tear_box_avg_iou': iou_str,
                'tear_box_miss_rate': miss_str,
                'raw_map_mean': map_m
            })

            # For Table E3
            if cond == 'torn_bright':
                denom_sub = f_df[f_df['condition'] == 'torn_bright']
                e3_comparison_rows.append({
                    'model': model_name,
                    'condition': 'torn_bright (Severe Glare)',
                    'task1_denom_acc_drop': f"{abs(denom_sub['delta_acc(%)'].mean()):.2f}%",
                    'task2_tear_mAP50_drop': f"{abs(map_m - f_df[f_df['condition']=='indoor_baseline']['tear_map50'].mean()):.2f}%" if map_m is not None else "N/A",
                    'more_vulnerable_task': "Classification" if abs(denom_sub['delta_acc(%)'].mean()) > abs(map_m - f_df[f_df['condition']=='indoor_baseline']['tear_map50'].mean()) else "Tear Localization"
                })

    # ── Export CSV ─────────────────────────────────────────────────────────────
    denom_df = pd.DataFrame(denom_summary_rows)
    tear_df = pd.DataFrame(tear_summary_rows)
    e3_df = pd.DataFrame(e3_comparison_rows)

    denom_csv = results_path / "summary_e1_denomination_localization_kfold.csv"
    tear_csv = results_path / "summary_e1_tear_detection_classification_kfold.csv"
    e3_csv = results_path / "summary_e3_degradation_comparison.csv"

    denom_df.to_csv(denom_csv, index=False)
    tear_df.to_csv(tear_csv, index=False)
    e3_df.to_csv(e3_csv, index=False)

    # ── PRINT DETAILED TERMINAL TABLES ───────────────────────────────────────
    print("\n" + "=" * 135)
    print("📋 TABLE 1 (E1): BANKNOTE DETECTION & CLASSIFICATION (5-FOLD MEAN ± STD)")
    print("=" * 135)
    print(f"{'Model':<9} | {'Condition':<16} | {'Top-1 Acc (%)':<16} | {'Δ vs Indoor (%)':<18} | {'Banknote mAP50':<16} | {'Banknote IoU(%)':<17} | {'Miss Rate(%)'}")
    print("-" * 135)
    for _, r in denom_df.iterrows():
        print(f"{r['model']:<9} | {r['condition']:<16} | {r['accuracy_denom']:<16} | {r['delta_acc(%)']:<18} | {r['banknote_mAP50']:<16} | {r['banknote_avg_iou']:<17} | {r['banknote_miss_rate']}")
    print("=" * 135)

    print("\n" + "=" * 155)
    print("🩹 TABLE 2 (E1): TEAR DETECTION & LOCALIZATION (5-FOLD MEAN ± STD)")
    print("=" * 155)
    print(f"{'Model':<9} | {'Condition':<16} | {'Data Type':<12} | {'Accuracy(%)':<16} | {'False Alarm(%)':<18} | {'Tear mAP50':<16} | {'Tear IoU(%)':<15} | {'Miss Rate(%)'}")
    print("-" * 155)
    for _, r in tear_df.iterrows():
        print(f"{r['model']:<9} | {r['condition']:<16} | {r['data_type']:<12} | {r['binary_tear_acc']:<16} | {r['false_alarm_rate']:<18} | {r['tear_mAP50']:<16} | {r['tear_box_avg_iou']:<15} | {r['tear_box_miss_rate']}")
    print("=" * 155)

    print("\n" + "=" * 105)
    print("📈 TABLE 3 (E3): TASK DEGRADATION COMPARISON UNDER SEVERE GLARE")
    print("=" * 105)
    print(f"{'Model':<9} | {'Test Condition':<24} | {'Denom Acc Degradation':<25} | {'Tear mAP50 Degradation':<25}")
    print("-" * 105)
    for _, r in e3_df.iterrows():
        print(f"{r['model']:<9} | {r['condition']:<24} | {r['task1_denom_acc_drop']:<25} | {r['task2_tear_mAP50_drop']:<25}")
    print("=" * 105)
    print(f"\n📁 Successfully saved 3 detailed CSV files:")
    print(f"  - Table 1: {denom_csv}")
    print(f"  - Table 2: {tear_csv}")
    print(f"  - Table 3: {e3_csv}\n")


def main():
    parser = argparse.ArgumentParser(description="Run experiments E1 & E3 with 5-Fold Cross-Validation.")
    parser.add_argument("--data_dir", type=str, default=".", help="Dataset directory")
    parser.add_argument("--metadata", type=str, default="metadata.csv", help="Path to metadata.csv")
    parser.add_argument("--models", nargs="+", default=["yolov8n", "yolo11n"], help="List of model architectures")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="Device (0, cpu, auto)")
    parser.add_argument("--results_dir", type=str, default="results", help="Directory to save results")

    args = parser.parse_args()
    run_experiment_e1_kfold(
        data_dir=args.data_dir,
        metadata=args.metadata,
        models=args.models,
        n_splits=args.n_splits,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch,
        device=args.device,
        results_dir=args.results_dir
    )


if __name__ == "__main__":
    main()
