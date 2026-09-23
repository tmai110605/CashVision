import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path("f:/CashVision")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from mqtone import MQTone
from quality_gate import QualityGate

CLASS_NAMES = ['10', '100', '20', '200', '50', '500', 'torn']
DENOM_NAMES = ['10', '100', '20', '200', '50', '500']
DENOM_VALUE_MAP = {
    10000: '10', 100000: '100', 20000: '20', 200000: '200', 50000: '50', 500000: '500'
}

def box_iou(box1, box2):
    # box: [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    a2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def load_ground_truth(lbl_path, img_w=640, img_h=640):
    denom_cls = None
    denom_box = None
    tear_boxes = []
    if lbl_path.exists():
        with open(lbl_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    cid = int(parts[0])
                    cx, cy, w, h = [float(x) for x in parts[1:5]]
                    x1 = (cx - w / 2.0) * img_w
                    y1 = (cy - h / 2.0) * img_h
                    x2 = (cx + w / 2.0) * img_w
                    y2 = (cy + h / 2.0) * img_h
                    box = [x1, y1, x2, y2]
                    if cid in range(6):
                        denom_cls = CLASS_NAMES[cid]
                        denom_box = box
                    elif cid == 6:
                        tear_boxes.append(box)
    return denom_cls, denom_box, tear_boxes

def evaluate_model(model_name, yolo_model, mqtone_model=None, split_prefix='test_'):
    df = pd.read_csv('metadata.csv')
    subset = df[df['filename'].str.startswith(split_prefix)].copy()
    
    results_by_cond = {}
    conditions = ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']
    
    for cond in conditions:
        cond_df = subset[subset['condition'] == cond]
        total = len(cond_df)
        correct_denom = 0
        tear_tp = 0
        tear_gt_count = 0
        tear_pred_count = 0
        
        for _, row in cond_df.iterrows():
            fn = row['filename']
            stem = Path(fn).stem
            
            # Find img and label
            img_path = None
            lbl_path = None
            for s in ['train', 'valid', 'test']:
                ip = Path(s) / 'images' / fn
                lp = Path(s) / 'labels' / f'{stem}.txt'
                if ip.exists():
                    img_path = ip
                    lbl_path = lp
                    break
            
            if img_path is None or not img_path.exists():
                continue
                
            gt_denom, gt_denom_box, gt_tear_boxes = load_ground_truth(lbl_path)
            if gt_denom is None:
                # fallback from metadata denomination_class
                gt_denom = DENOM_VALUE_MAP.get(row['denomination_class'], None)
                
            # Read img
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            
            # Preprocessing with MQTone if enabled
            if mqtone_model is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                rgb_resized = cv2.resize(rgb, (640, 640))
                inp_t = torch.from_numpy(rgb_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
                with torch.no_grad():
                    corr_t = mqtone_model(inp_t)
                corr_rgb = (corr_t.squeeze(0).permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                feed_img = cv2.cvtColor(corr_rgb, cv2.COLOR_RGB2BGR)
            else:
                feed_img = bgr
                
            # YOLO inference
            preds = yolo_model(feed_img, conf=0.25, verbose=False)[0]
            
            pred_denom = None
            best_denom_conf = -1.0
            pred_tear_boxes = []
            
            for b in preds.boxes:
                cls_id = int(b.cls[0].item())
                conf = float(b.conf[0].item())
                xyxy = b.xyxy[0].tolist()
                
                if cls_id in range(6):
                    if conf > best_denom_conf:
                        best_denom_conf = conf
                        pred_denom = CLASS_NAMES[cls_id]
                elif cls_id == 6:
                    pred_tear_boxes.append((xyxy, conf))
                    
            if pred_denom == gt_denom:
                correct_denom += 1
                
            if len(gt_tear_boxes) > 0:
                tear_gt_count += len(gt_tear_boxes)
                tear_pred_count += len(pred_tear_boxes)
                for gt_b in gt_tear_boxes:
                    for pr_b, _ in pred_tear_boxes:
                        if box_iou(gt_b, pr_b) >= 0.5:
                            tear_tp += 1
                            break
                            
        denom_acc = (correct_denom / total * 100.0) if total > 0 else 0.0
        tear_map50 = (tear_tp / max(1, tear_gt_count) * 100.0) if tear_gt_count > 0 else 0.0
        results_by_cond[cond] = {
            "n": total,
            "denom_acc": denom_acc,
            "tear_map50": tear_map50
        }
        
    overall_denom = np.mean([results_by_cond[c]["denom_acc"] for c in conditions])
    return {
        "model": model_name,
        "split": split_prefix,
        "results": results_by_cond,
        "overall_mean": overall_denom
    }

if __name__ == "__main__":
    print("Testing evaluation on test_ subset (276 images)...")
    v8_model = YOLO("resultsc1/yolov8n/fold_1/weights/best.pt")
    res_v8 = evaluate_model("YOLOv8n Baseline", v8_model, split_prefix="test_")
    print("YOLOv8n Baseline Test Results:")
    for cond, data in res_v8["results"].items():
        print(f"  {cond}: acc={data['denom_acc']:.2f}%, tear_map50={data['tear_map50']:.2f}%")
    print(f"  Overall Mean: {res_v8['overall_mean']:.2f}%")
