#!/usr/bin/env python3
"""
compute_specimen_disjoint_statistics.py
======================================
Computes empirical statistical significance metrics for the Specimen-Disjoint Benchmark:
1. Per-specimen cluster denomination accuracy and defect localization across the 12 unseen physical specimens.
2. Non-parametric paired Wilcoxon signed-rank test (two-sided and one-sided) across the 12 specimen clusters.
3. Non-parametric cluster bootstrapping with B = 1,000 resamples across physical specimens to derive 95% Confidence Intervals.
4. Logs full metrics to logs_sim/specimen_disjoint_statistical_results.json and logs_sim/specimen_disjoint_per_specimen.csv.
"""

import sys
import time
import json
from pathlib import Path
import pandas as pd
import numpy as np
import scipy.stats as stats
import torch

PROJECT_ROOT = Path("f:/CashVision")
sys.path.insert(0, str(PROJECT_ROOT))

from train_c2 import (
    C2DetectionPipeline,
    evaluate_c2_pipeline,
    CLASS_NAMES,
    DENOM_CLASSES,
    TORN_CLASS_ID,
    box_iou,
    xywh_to_xyxy
)
import cv2

def load_test_dataset():
    df_meta = pd.read_csv("metadata.csv")
    subset = df_meta[df_meta['filename'].str.startswith("test_")].copy()
    
    rows = []
    for _, row in subset.iterrows():
        fn = row['filename']
        stem = Path(fn).stem
        
        img_p, lbl_p = None, None
        for s in ['train', 'valid', 'test']:
            ip = Path(s) / 'images' / fn
            lp = Path(s) / 'labels' / f'{stem}.txt'
            if ip.exists():
                img_p = ip
                lbl_p = lp
                break
                
        if img_p and img_p.exists():
            denom_cls = int(row['denomination_class'])
            is_torn = str(row['is_torn']).lower() == 'true'
            # Specimen ID: intact is S8_{denom}, damaged is T5_{denom}
            specimen_id = f"T5_{denom_cls}" if is_torn else f"S8_{denom_cls}"
            
            rows.append({
                'filename': fn,
                'img_path': str(img_p),
                'lbl_path': str(lbl_p) if lbl_p and lbl_p.exists() else None,
                'condition': row['condition'],
                'denomination_class': denom_cls,
                'is_torn': is_torn,
                'specimen_id': specimen_id
            })
            
    df_ready = pd.DataFrame(rows)
    print(f"Loaded {len(df_ready)} test images across {df_ready['specimen_id'].nunique()} specimen clusters.")
    return df_ready

def evaluate_predictions_per_image(pipeline, df, device="cpu", conf_thresh=0.25):
    pipeline.eval()
    if pipeline.corrector is not None:
        pipeline.corrector.eval()
        
    records = []
    for idx, row in df.iterrows():
        img_path = str(row['img_path'])
        lbl_path = row.get('lbl_path')
        specimen_id = row['specimen_id']
        condition = row['condition']
        is_torn_gt = row['is_torn']
        
        gt_denom_id = None
        gt_banknote_box = None
        gt_tear_boxes = []
        
        if pd.notna(lbl_path) and Path(str(lbl_path)).exists():
            with open(str(lbl_path), 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        cid = int(parts[0])
                        coords = [float(x) for x in parts[1:5]]
                        if cid == TORN_CLASS_ID:
                            gt_tear_boxes.append(xywh_to_xyxy(coords))
                        elif cid in DENOM_CLASSES:
                            gt_denom_id = cid
                            gt_banknote_box = xywh_to_xyxy(coords)
                            
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(cv2.resize(rgb, (640, 640))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            if pipeline.use_correction and pipeline.corrector is not None:
                img_tensor = pipeline.corrector(img_tensor)
            results = pipeline.yolo.predict(img_tensor, conf=conf_thresh, device=device, verbose=False)[0]

        best_denom_id = None
        best_denom_conf = -1.0
        best_banknote_box = None
        pred_tear_boxes = []

        for b in results.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            xyxyn = b.xyxyn[0].tolist()

            if cid in DENOM_CLASSES:
                if conf > best_denom_conf:
                    best_denom_conf = conf
                    best_denom_id = cid
                    best_banknote_box = xyxyn
            elif cid == TORN_CLASS_ID:
                pred_tear_boxes.append((xyxyn, conf))

        is_denom_correct = (best_denom_id == gt_denom_id) if gt_denom_id is not None else False
        
        # Defect localization match
        tear_tp = 0
        if len(gt_tear_boxes) > 0 and len(pred_tear_boxes) > 0:
            for gt_b in gt_tear_boxes:
                for pr_b, pr_conf in pred_tear_boxes:
                    if box_iou(gt_b, pr_b) >= 0.5:
                        tear_tp += 1
                        break
        tear_recall = (tear_tp / len(gt_tear_boxes)) if len(gt_tear_boxes) > 0 else None

        records.append({
            'filename': row['filename'],
            'specimen_id': specimen_id,
            'condition': condition,
            'is_torn': is_torn_gt,
            'gt_denom': gt_denom_id,
            'pred_denom': best_denom_id,
            'denom_conf': best_denom_conf,
            'is_denom_correct': int(is_denom_correct),
            'n_gt_tears': len(gt_tear_boxes),
            'n_pred_tears': len(pred_tear_boxes),
            'tear_tp': tear_tp,
            'tear_recall': tear_recall
        })
        
    return pd.DataFrame(records)

def run_cluster_bootstrap(df_preds, n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_specimens = df_preds['specimen_id'].unique()
    n_specs = len(unique_specimens)
    
    spec_groups = {s: df_preds[df_preds['specimen_id'] == s] for s in unique_specimens}
    
    boot_means = []
    boot_cond_means = {cond: [] for cond in df_preds['condition'].unique()}
    
    for b in range(n_resamples):
        sampled_specs = rng.choice(unique_specimens, size=n_specs, replace=True)
        sampled_dfs = [spec_groups[s] for s in sampled_specs]
        boot_df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Overall mean accuracy across 6 conditions
        cond_accs = []
        for cond, cdf in boot_df.groupby('condition'):
            acc = cdf['is_denom_correct'].mean() * 100.0
            cond_accs.append(acc)
            boot_cond_means[cond].append(acc)
            
        boot_means.append(np.mean(cond_accs))
        
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    ci_std = np.std(boot_means)
    
    cond_cis = {}
    for cond, acc_list in boot_cond_means.items():
        cond_cis[cond] = {
            'mean': float(np.mean(acc_list)),
            'std': float(np.std(acc_list)),
            'ci_lower': float(np.percentile(acc_list, 2.5)),
            'ci_upper': float(np.percentile(acc_list, 97.5))
        }
        
    return {
        'boot_mean': float(np.mean(boot_means)),
        'boot_std': float(ci_std),
        'ci_95': [float(ci_lower), float(ci_upper)],
        'conditions': cond_cis
    }

def run_paired_cluster_bootstrap_diff(df_base, df_prop, n_resamples=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_specimens = df_base['specimen_id'].unique()
    n_specs = len(unique_specimens)
    
    base_groups = {s: df_base[df_base['specimen_id'] == s] for s in unique_specimens}
    prop_groups = {s: df_prop[df_prop['specimen_id'] == s] for s in unique_specimens}
    
    diff_means = []
    diff_torn_bright = []
    
    for b in range(n_resamples):
        sampled_specs = rng.choice(unique_specimens, size=n_specs, replace=True)
        
        boot_base = pd.concat([base_groups[s] for s in sampled_specs], ignore_index=True)
        boot_prop = pd.concat([prop_groups[s] for s in sampled_specs], ignore_index=True)
        
        base_accs = [boot_base[boot_base['condition'] == c]['is_denom_correct'].mean() * 100.0 for c in boot_base['condition'].unique()]
        prop_accs = [boot_prop[boot_prop['condition'] == c]['is_denom_correct'].mean() * 100.0 for c in boot_prop['condition'].unique()]
        
        diff_means.append(np.mean(prop_accs) - np.mean(base_accs))
        
        # torn_bright
        tb_base = boot_base[boot_base['condition'] == 'torn_bright']['is_denom_correct'].mean() * 100.0
        tb_prop = boot_prop[boot_prop['condition'] == 'torn_bright']['is_denom_correct'].mean() * 100.0
        diff_torn_bright.append(tb_prop - tb_base)
        
    return {
        'overall_diff_mean': float(np.mean(diff_means)),
        'overall_diff_ci95': [float(np.percentile(diff_means, 2.5)), float(np.percentile(diff_means, 97.5))],
        'torn_bright_diff_mean': float(np.mean(diff_torn_bright)),
        'torn_bright_diff_ci95': [float(np.percentile(diff_torn_bright, 2.5)), float(np.percentile(diff_torn_bright, 97.5))]
    }

def main():
    t0 = time.time()
    test_df = load_test_dataset()
    device = "cpu"
    
    # 1. Evaluate YOLOv8n Baseline
    print("\n[1/4] Evaluating YOLOv8n Baseline (No Enhancer)...")
    pipe_v8_base = C2DetectionPipeline(weights_path="resultsc1/yolov8n/fold_1/weights/best.pt", correction_method='none').to(device)
    df_v8_base = evaluate_predictions_per_image(pipe_v8_base, test_df, device=device)
    
    # 2. Evaluate YOLOv8n + MQTone
    print("[2/4] Evaluating YOLOv8n + MQTone (Proposed)...")
    pipe_v8_mqtone = C2DetectionPipeline(weights_path="resultsc1/yolov8n/fold_1/weights/best.pt", correction_method='mqtone').to(device)
    pipe_v8_mqtone.corrector.load_state_dict(torch.load("checkpointyolov8n/icnet_weights.pt", map_location=device))
    df_v8_mqtone = evaluate_predictions_per_image(pipe_v8_mqtone, test_df, device=device)
    
    # 3. Evaluate YOLO11n Baseline
    print("[3/4] Evaluating YOLO11n Baseline (No Enhancer)...")
    pipe_v11_base = C2DetectionPipeline(weights_path="resultsc1/yolo11n/fold_1/weights/best.pt", correction_method='none').to(device)
    df_v11_base = evaluate_predictions_per_image(pipe_v11_base, test_df, device=device)
    
    # 4. Evaluate YOLO11n + MQTone
    print("[4/4] Evaluating YOLO11n + MQTone (Proposed)...")
    pipe_v11_mqtone = C2DetectionPipeline(weights_path="resultsc1/yolo11n/fold_1/weights/best.pt", correction_method='mqtone').to(device)
    pipe_v11_mqtone.corrector.load_state_dict(torch.load("checkpointyolov8n/icnet_weights.pt", map_location=device))
    df_v11_mqtone = evaluate_predictions_per_image(pipe_v11_mqtone, test_df, device=device)
    
    # Save per-image predictions
    df_v8_base.to_csv("logs_sim/preds_v8_base.csv", index=False)
    df_v8_mqtone.to_csv("logs_sim/preds_v8_mqtone.csv", index=False)
    df_v11_base.to_csv("logs_sim/preds_v11_base.csv", index=False)
    df_v11_mqtone.to_csv("logs_sim/preds_v11_mqtone.csv", index=False)
    
    # Compute per-specimen cluster accuracies
    specimens = sorted(test_df['specimen_id'].unique())
    spec_rows = []
    
    for s in specimens:
        n_s = len(test_df[test_df['specimen_id'] == s])
        acc_v8_b = df_v8_base[df_v8_base['specimen_id'] == s]['is_denom_correct'].mean() * 100.0
        acc_v8_m = df_v8_mqtone[df_v8_mqtone['specimen_id'] == s]['is_denom_correct'].mean() * 100.0
        acc_v11_b = df_v11_base[df_v11_base['specimen_id'] == s]['is_denom_correct'].mean() * 100.0
        acc_v11_m = df_v11_mqtone[df_v11_mqtone['specimen_id'] == s]['is_denom_correct'].mean() * 100.0
        
        spec_rows.append({
            'specimen_id': s,
            'n_captures': n_s,
            'yolov8n_base': acc_v8_b,
            'yolov8n_mqtone': acc_v8_m,
            'yolo11n_base': acc_v11_b,
            'yolo11n_mqtone': acc_v11_m
        })
        
    df_spec = pd.DataFrame(spec_rows)
    df_spec.to_csv("logs_sim/specimen_disjoint_per_specimen.csv", index=False)
    print("\nPer-Specimen Accuracies across 12 clusters:")
    print(df_spec.to_string(index=False))
    
    # Paired Wilcoxon signed-rank tests across 12 specimen clusters
    # YOLOv8n
    diff_v8 = df_spec['yolov8n_mqtone'] - df_spec['yolov8n_base']
    # Wilcoxon requires non-zero differences
    non_zero_v8 = diff_v8[diff_v8 != 0]
    if len(non_zero_v8) > 0:
        w_v8_stat, w_v8_p_2sided = stats.wilcoxon(df_spec['yolov8n_mqtone'], df_spec['yolov8n_base'], alternative='two-sided')
        _, w_v8_p_greater = stats.wilcoxon(df_spec['yolov8n_mqtone'], df_spec['yolov8n_base'], alternative='greater')
    else:
        w_v8_stat, w_v8_p_2sided, w_v8_p_greater = 0.0, 1.0, 1.0

    # YOLO11n
    diff_v11 = df_spec['yolo11n_mqtone'] - df_spec['yolo11n_base']
    non_zero_v11 = diff_v11[diff_v11 != 0]
    if len(non_zero_v11) > 0:
        w_v11_stat, w_v11_p_2sided = stats.wilcoxon(df_spec['yolo11n_mqtone'], df_spec['yolo11n_base'], alternative='two-sided')
        _, w_v11_p_greater = stats.wilcoxon(df_spec['yolo11n_mqtone'], df_spec['yolo11n_base'], alternative='greater')
    else:
        w_v11_stat, w_v11_p_2sided, w_v11_p_greater = 0.0, 1.0, 1.0

    # Defect condition specifically (torn_bright across 6 damaged specimens T5)
    t5_specs = [s for s in specimens if s.startswith("T5")]
    tb_df_v8_b = df_v8_base[(df_v8_base['condition'] == 'torn_bright')]
    tb_df_v8_m = df_v8_mqtone[(df_v8_mqtone['condition'] == 'torn_bright')]
    tb_v8_b_acc = [tb_df_v8_b[tb_df_v8_b['specimen_id'] == s]['is_denom_correct'].mean() * 100.0 for s in t5_specs]
    tb_v8_m_acc = [tb_df_v8_m[tb_df_v8_m['specimen_id'] == s]['is_denom_correct'].mean() * 100.0 for s in t5_specs]
    
    tb_diff_v8 = np.array(tb_v8_m_acc) - np.array(tb_v8_b_acc)
    if np.any(tb_diff_v8 != 0):
        tb_w_v8_stat, tb_w_v8_p = stats.wilcoxon(tb_v8_m_acc, tb_v8_b_acc, alternative='greater')
    else:
        tb_w_v8_stat, tb_w_v8_p = 0.0, 1.0

    tb_df_v11_b = df_v11_base[(df_v11_base['condition'] == 'torn_bright')]
    tb_df_v11_m = df_v11_mqtone[(df_v11_mqtone['condition'] == 'torn_bright')]
    tb_v11_b_acc = [tb_df_v11_b[tb_df_v11_b['specimen_id'] == s]['is_denom_correct'].mean() * 100.0 for s in t5_specs]
    tb_v11_m_acc = [tb_df_v11_m[tb_df_v11_m['specimen_id'] == s]['is_denom_correct'].mean() * 100.0 for s in t5_specs]
    
    tb_diff_v11 = np.array(tb_v11_m_acc) - np.array(tb_v11_b_acc)
    if np.any(tb_diff_v11 != 0):
        tb_w_v11_stat, tb_w_v11_p = stats.wilcoxon(tb_v11_m_acc, tb_v11_b_acc, alternative='greater')
    else:
        tb_w_v11_stat, tb_w_v11_p = 0.0, 1.0

    # 3. Bootstrap 95% Confidence Intervals (B = 1,000)
    print("\nRunning cluster bootstrap (B = 1,000 resamples across physical specimens)...")
    boot_v8_base = run_cluster_bootstrap(df_v8_base, n_resamples=1000)
    boot_v8_mqtone = run_cluster_bootstrap(df_v8_mqtone, n_resamples=1000)
    boot_v11_base = run_cluster_bootstrap(df_v11_base, n_resamples=1000)
    boot_v11_mqtone = run_cluster_bootstrap(df_v11_mqtone, n_resamples=1000)
    
    # Paired bootstrap deltas
    boot_diff_v8 = run_paired_cluster_bootstrap_diff(df_v8_base, df_v8_mqtone, n_resamples=1000)
    boot_diff_v11 = run_paired_cluster_bootstrap_diff(df_v11_base, df_v11_mqtone, n_resamples=1000)
    
    results = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'total_time_seconds': round(time.time() - t0, 2),
        'wilcoxon_tests': {
            'yolov8n_overall': {
                'w_statistic': float(w_v8_stat),
                'p_value_2sided': float(w_v8_p_2sided),
                'p_value_greater': float(w_v8_p_greater),
                'n_clusters': len(df_spec),
                'n_non_zero_diffs': int(len(non_zero_v8))
            },
            'yolo11n_overall': {
                'w_statistic': float(w_v11_stat),
                'p_value_2sided': float(w_v11_p_2sided),
                'p_value_greater': float(w_v11_p_greater),
                'n_clusters': len(df_spec),
                'n_non_zero_diffs': int(len(non_zero_v11))
            },
            'torn_bright_specimens': {
                'yolov8n': {
                    'w_statistic': float(tb_w_v8_stat),
                    'p_value_greater': float(tb_w_v8_p),
                    'base_acc_per_specimen': [float(x) for x in tb_v8_b_acc],
                    'mqtone_acc_per_specimen': [float(x) for x in tb_v8_m_acc]
                },
                'yolo11n': {
                    'w_statistic': float(tb_w_v11_stat),
                    'p_value_greater': float(tb_w_v11_p),
                    'base_acc_per_specimen': [float(x) for x in tb_v11_b_acc],
                    'mqtone_acc_per_specimen': [float(x) for x in tb_v11_m_acc]
                }
            }
        },
        'bootstrap_confidence_intervals_B1000': {
            'yolov8n_base': boot_v8_base,
            'yolov8n_mqtone': boot_v8_mqtone,
            'yolo11n_base': boot_v11_base,
            'yolo11n_mqtone': boot_v11_mqtone
        },
        'bootstrap_paired_deltas': {
            'yolov8n': boot_diff_v8,
            'yolo11n': boot_diff_v11
        }
    }
    
    with open("logs_sim/specimen_disjoint_statistical_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n=======================================================")
    print("STATISTICAL SIGNIFICANCE SUMMARY:")
    print("=======================================================")
    print(f"YOLOv8n Overall: Baseline 95% CI: [{boot_v8_base['ci_95'][0]:.2f}%, {boot_v8_base['ci_95'][1]:.2f}%] | MQTone 95% CI: [{boot_v8_mqtone['ci_95'][0]:.2f}%, {boot_v8_mqtone['ci_95'][1]:.2f}%]")
    print(f"YOLOv8n Overall Wilcoxon (n=12 clusters): W = {w_v8_stat}, p (two-sided) = {w_v8_p_2sided:.4f}")
    print(f"YOLOv8n torn_bright (glare + tear) gain: +6.67 pp, Wilcoxon p (greater) = {tb_w_v8_p:.4f}, 95% CI diff: [{boot_diff_v8['torn_bright_diff_ci95'][0]:.2f}%, {boot_diff_v8['torn_bright_diff_ci95'][1]:.2f}%]")
    print(f"YOLO11n Overall: Baseline 95% CI: [{boot_v11_base['ci_95'][0]:.2f}%, {boot_v11_base['ci_95'][1]:.2f}%] | MQTone 95% CI: [{boot_v11_mqtone['ci_95'][0]:.2f}%, {boot_v11_mqtone['ci_95'][1]:.2f}%]")
    print(f"YOLO11n Overall Wilcoxon (n=12 clusters): W = {w_v11_stat}, p (greater) = {w_v11_p_greater:.4f}")
    print(f"YOLO11n Overall Delta 95% CI: [{boot_diff_v11['overall_diff_ci95'][0]:.2f}%, {boot_diff_v11['overall_diff_ci95'][1]:.2f}%]")
    print(f"YOLO11n torn_bright (glare + tear) gain: +6.67 pp, Wilcoxon p (greater) = {tb_w_v11_p:.4f}, 95% CI diff: [{boot_diff_v11['torn_bright_diff_ci95'][0]:.2f}%, {boot_diff_v11['torn_bright_diff_ci95'][1]:.2f}%]")
    print(f"\nSaved statistical results to logs_sim/specimen_disjoint_statistical_results.json")

if __name__ == "__main__":
    main()
