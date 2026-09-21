#!/usr/bin/env python3
"""
run_specimen_disjoint_benchmark.py
==================================
Official execution of the Specimen-Disjoint Benchmark across:
- YOLOv8n Baseline (No Enhancer)
- YOLOv8n + MQTone (Proposed)
- YOLO11n Baseline (No Enhancer)
- YOLO11n + MQTone (Proposed)

Evaluated on:
- Seen Banknotes: In-Distribution Validation Pool (val_ prefix, K_val = 12 physical notes, n = 276 captures)
- Unseen Banknote Specimens: Locked Holdout Test Pool (test_ prefix, K_test = 12 physical notes, n = 276 captures:
  60 indoor, 60 outdoor, 48 backlight, 48 overexposed, 30 torn clean, 30 torn bright).
"""

import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np
import torch

PROJECT_ROOT = Path("f:/CashVision")
sys.path.insert(0, str(PROJECT_ROOT))

from train_c2 import (
    C2DetectionPipeline,
    evaluate_c2_pipeline,
    CLASS_NAMES,
    DENOM_CLASSES,
    TORN_CLASS_ID
)
from mqtone import MQTone

def prepare_dataset_df(prefix="test_"):
    df_meta = pd.read_csv("metadata.csv")
    subset = df_meta[df_meta['filename'].str.startswith(prefix)].copy()
    
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
            rows.append({
                'filename': fn,
                'img_path': str(img_p),
                'lbl_path': str(lbl_p) if lbl_p and lbl_p.exists() else None,
                'condition': row['condition'],
                'denomination_class': row['denomination_class'],
                'is_torn': str(row['is_torn']).lower() == 'true'
            })
            
    df_ready = pd.DataFrame(rows)
    print(f"Prepared {len(df_ready)} images for prefix='{prefix}'")
    return df_ready

def evaluate_configuration(config_name, pipeline, val_df, test_df, device="cpu"):
    print(f"\n=======================================================")
    print(f"Running evaluation for: {config_name}")
    print(f"=======================================================")
    
    # 1. In-Distribution Validation Accuracy (Seen Notes, Indoor)
    val_indoor_df = val_df[val_df['condition'] == 'indoor']
    val_res = evaluate_c2_pipeline(pipeline, val_indoor_df, "val_indoor", device=device, conf_thresh=0.25)
    in_dist_val = val_res['accuracy_denom']
    print(f"  -> In-Dist. Val (Seen notes, Indoor): {in_dist_val:.2f}%")
    
    # 2. Unseen Test Evaluation across 6 conditions
    conditions = ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']
    cond_results = {}
    
    for cond in conditions:
        sub_df = test_df[test_df['condition'] == cond]
        res = evaluate_c2_pipeline(pipeline, sub_df, cond, device=device, conf_thresh=0.25)
        
        acc = res['accuracy_denom']
        tear_map = res.get('mAP50_tear', 0.0)
        if tear_map is None:
            tear_map = 0.0
            
        cond_results[cond] = {
            'denom_acc': acc,
            'tear_map50': tear_map,
            'n': len(sub_df)
        }
        if 'torn' in cond:
            print(f"  -> Test [{cond}] (n={len(sub_df)}): denom_acc={acc:.2f}%, tear_map50={tear_map:.2f}%")
        else:
            print(f"  -> Test [{cond}] (n={len(sub_df)}): denom_acc={acc:.2f}%")
        
    overall_mean = np.mean([cond_results[c]['denom_acc'] for c in conditions])
    print(f"  -> Overall Mean on Unseen Notes: {overall_mean:.2f}%")
    
    return {
        'config': config_name,
        'in_dist_val': in_dist_val,
        'results': cond_results,
        'overall_mean': round(overall_mean, 2)
    }

if __name__ == "__main__":
    t0 = time.time()
    val_df = prepare_dataset_df("val_")
    test_df = prepare_dataset_df("test_")
    device = "cpu"
    
    output_rows = []
    
    # Model 1: YOLOv8n Baseline (No Enhancer)
    pipe_v8_base = C2DetectionPipeline(weights_path="resultsc1/yolov8n/fold_1/weights/best.pt", correction_method='none').to(device)
    res_v8_base = evaluate_configuration("YOLOv8n Baseline (No Enhancer)", pipe_v8_base, val_df, test_df, device=device)
    output_rows.append(res_v8_base)
    
    # Model 2: YOLOv8n + MQTone (Proposed)
    pipe_v8_mqtone = C2DetectionPipeline(weights_path="resultsc1/yolov8n/fold_1/weights/best.pt", correction_method='mqtone').to(device)
    pipe_v8_mqtone.corrector.load_state_dict(torch.load("checkpointyolov8n/icnet_weights.pt", map_location=device))
    res_v8_mqtone = evaluate_configuration("YOLOv8n + MQTone (Proposed)", pipe_v8_mqtone, val_df, test_df, device=device)
    output_rows.append(res_v8_mqtone)
    
    # Model 3: YOLO11n Baseline (No Enhancer)
    pipe_v11_base = C2DetectionPipeline(weights_path="resultsc1/yolo11n/fold_1/weights/best.pt", correction_method='none').to(device)
    res_v11_base = evaluate_configuration("YOLO11n Baseline (No Enhancer)", pipe_v11_base, val_df, test_df, device=device)
    output_rows.append(res_v11_base)
    
    # Model 4: YOLO11n + MQTone (Proposed)
    pipe_v11_mqtone = C2DetectionPipeline(weights_path="resultsc1/yolo11n/fold_1/weights/best.pt", correction_method='mqtone').to(device)
    pipe_v11_mqtone.corrector.load_state_dict(torch.load("checkpointyolov8n/icnet_weights.pt", map_location=device))
    res_v11_mqtone = evaluate_configuration("YOLO11n + MQTone (Proposed)", pipe_v11_mqtone, val_df, test_df, device=device)
    output_rows.append(res_v11_mqtone)
    
    print("\n\n=======================================================")
    print("FINAL SUMMARY TABLE (Specimen-Disjoint Benchmark):")
    print("=======================================================")
    records = []
    for row in output_rows:
        cfg = row['config']
        v_in = row['in_dist_val']
        r = row['results']
        mean_acc = row['overall_mean']
        print(f"{cfg}: InDist={v_in:.2f}% | Indoor={r['indoor']['denom_acc']:.2f}% | Outdoor={r['outdoor']['denom_acc']:.2f}% | Backlight={r['backlight']['denom_acc']:.2f}% | Overexp={r['overexposed']['denom_acc']:.2f}% | TornClean={r['torn_clean']['denom_acc']:.2f} [{r['torn_clean']['tear_map50']:.2f}] | TornBright={r['torn_bright']['denom_acc']:.2f} [{r['torn_bright']['tear_map50']:.2f}] | Mean={mean_acc:.2f}%")
        records.append({
            'config': cfg,
            'in_dist_val': v_in,
            'indoor': r['indoor']['denom_acc'],
            'outdoor': r['outdoor']['denom_acc'],
            'backlight': r['backlight']['denom_acc'],
            'overexposed': r['overexposed']['denom_acc'],
            'torn_clean': r['torn_clean']['denom_acc'],
            'torn_clean_tear_map50': r['torn_clean']['tear_map50'],
            'torn_bright': r['torn_bright']['denom_acc'],
            'torn_bright_tear_map50': r['torn_bright']['tear_map50'],
            'overall_mean': mean_acc
        })
        
    out_df = pd.DataFrame(records)
    out_csv = Path("logs_sim/specimen_disjoint_benchmark_results.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved benchmark results to {out_csv}")
    print(f"Total time elapsed: {time.time()-t0:.1f}s")
