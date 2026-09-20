#!/usr/bin/env python3
"""
train_c2.py — Custom PyTorch Training & Evaluation Engine for C2
Integrated with:
  - Low-level DetectionModel (YOLOv8n / YOLOv11n)
  - Registry Correction Modules: MQTone (proposed C2 module), IAT (Transformer),
    Zero-DCE, Zero-DCE++, and non-learnable traditional baselines
    (Gamma Correction, CLAHE) — see enhancement_methods.py.
  - Photometric Supervision Loss for LEARNABLE modules (MQTone, IAT, Zero-DCE, Zero-DCE++)
    against clean reference images; traditional baselines (Gamma/CLAHE) omit this loss.
  - Custom training loop & evaluation engine (including Torn F1 / Torn mAP)

Note: Consistency Loss (on {light, dark} pairs) and PairBatchSampler have
been completely removed from this pipeline.
"""

import os
import sys
import time
from pathlib import Path

# Fix UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.utils.loss import v8DetectionLoss

from enhancement_methods import build_correction_module, NON_LEARNABLE_METHODS, CORRECTION_METHODS

CLASS_NAMES = ['10', '100', '20', '200', '50', '500', 'torn']
DENOM_CLASSES = {0: '10k', 1: '100k', 2: '20k', 3: '200k', 4: '50k', 5: '500k'}
TORN_CLASS_ID = 6


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
# Dataset & Custom Pair Sampler
# ─────────────────────────────────────────────────────────────────────────────
class CashVisionDataset(Dataset):
    """
    Dataset loading images, YOLO format bounding boxes, and pair_id/aug_type metadata
    (descriptive metadata, no longer used for Consistency Loss -- removed from pipeline).

    load_clean_ref: if True, concurrently loads the corresponding CLEAN REFERENCE image
    (undegraded) via 'orig_img_path' in metadata (recorded by augment.py).
    Used as target for MQTone photometric supervision loss: since all transformations
    in augment.py (global brightness/contrast/gamma + local glare/shadow) preserve spatial
    geometry (no crop/rotate/flip), degraded and clean images are strictly pixel-aligned,
    allowing direct use as (input, target) pairs for photometric reconstruction/identity
    loss separate from detection loss. Defaults to False to avoid I/O overhead.
    """
    def __init__(self, df: pd.DataFrame, img_size: int = 640, load_clean_ref: bool = False):
        self.img_size = img_size
        self.load_clean_ref = load_clean_ref
        self.samples = []

        for idx, row in df.iterrows():
            img_p = Path(str(row['img_path']))
            lbl_p = Path(str(row['lbl_path'])) if pd.notna(row.get('lbl_path')) else None
            pair_id = str(row.get('pair_id', '')) if pd.notna(row.get('pair_id')) else ''
            aug_type = str(row.get('augmentation_type', 'original'))
            condition = str(row.get('condition', 'unknown'))
            is_torn = (str(row.get('is_torn', 'false')).lower() == 'true')

            orig_img_path = None
            if load_clean_ref:
                raw_orig = row.get('orig_img_path', None)
                if pd.notna(raw_orig) and str(raw_orig).strip():
                    orig_img_path = str(raw_orig)
                else:
                    # Legacy metadata might not have this column -- safe fallback:
                    # treat the current image as its own clean reference (identity)
                    # instead of crashing. 'original' images remain valid either way.
                    orig_img_path = str(img_p)

            # Read YOLO annotations
            boxes = []  # list of [class_id, cx, cy, w, h]
            if lbl_p and lbl_p.exists():
                with open(lbl_p, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cid = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            boxes.append([cid] + coords)

            self.samples.append({
                'img_path': str(img_p),
                'orig_img_path': orig_img_path,
                'filename': img_p.name,
                'boxes': boxes,
                'pair_id': pair_id,
                'aug_type': aug_type,
                'condition': condition,
                'is_torn': is_torn
            })

    def __len__(self):
        return len(self.samples)

    def _load_img_tensor(self, path: str):
        bgr = cv2.imread(path)
        if bgr is None:
            return torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[0] != self.img_size or rgb.shape[1] != self.img_size:
            rgb = cv2.resize(rgb, (self.img_size, self.img_size))
        return torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0

    def __getitem__(self, idx):
        item = self.samples[idx]
        img_tensor = self._load_img_tensor(item['img_path'])

        out = {
            'img': img_tensor,
            'boxes': item['boxes'],
            'pair_id': item['pair_id'],
            'aug_type': item['aug_type'],
            'filename': item['filename'],
            'condition': item['condition'],
            'is_torn': item['is_torn'],
            'index': idx
        }

        if self.load_clean_ref:
            # Original images frequently repeat in a batch (each original generates
            # ~5 augmented versions sharing the same orig_img_path) -- re-reading is safer
            # and simpler than manual in-memory caching across DataLoader worker forks.
            out['clean_img'] = self._load_img_tensor(item['orig_img_path'])

        return out


def cashvision_collate_fn(batch):
    """
    Collate function batching samples for DetectionModel:
    Produces tensors: 'img' (B, 3, H, W), 'cls' (N, 1), 'bboxes' (N, 4), 'batch_idx' (N,)
    If load_clean_ref is enabled, also collates 'clean_img' (B, 3, H, W) -- clean target
    used for photometric supervision loss (MQTone, learnable modules).
    """
    imgs = torch.stack([item['img'] for item in batch], dim=0)
    clean_imgs = None
    if 'clean_img' in batch[0]:
        clean_imgs = torch.stack([item['clean_img'] for item in batch], dim=0)

    cls_list = []
    bbox_list = []
    batch_idx_list = []

    pair_ids = [item['pair_id'] for item in batch]
    aug_types = [item['aug_type'] for item in batch]
    filenames = [item['filename'] for item in batch]
    conditions = [item['condition'] for item in batch]
    is_torns = [item['is_torn'] for item in batch]

    for b_idx, item in enumerate(batch):
        for box in item['boxes']:
            cls_list.append([box[0]])
            bbox_list.append(box[1:5])
            batch_idx_list.append(b_idx)

    if len(cls_list) > 0:
        cls_tensor = torch.tensor(cls_list, dtype=torch.float32)
        bbox_tensor = torch.tensor(bbox_list, dtype=torch.float32)
        batch_idx_tensor = torch.tensor(batch_idx_list, dtype=torch.int64)
    else:
        cls_tensor = torch.zeros((0, 1), dtype=torch.float32)
        bbox_tensor = torch.zeros((0, 4), dtype=torch.float32)
        batch_idx_tensor = torch.zeros((0,), dtype=torch.int64)

    return {
        'img': imgs,
        'clean_img': clean_imgs,
        'cls': cls_tensor,
        'bboxes': bbox_tensor,
        'batch_idx': batch_idx_tensor,
        'pair_ids': pair_ids,
        'aug_types': aug_types,
        'filenames': filenames,
        'conditions': conditions,
        'is_torns': is_torns,
        'batch_size': len(batch)
    }


# ─────────────────────────────────────────────────────────────────────────────
# Model Wrapper: Correction Module (MQTone / Gamma / CLAHE / None) + YOLO
# DetectionModel
# ─────────────────────────────────────────────────────────────────────────────
class C2DetectionPipeline(nn.Module):
    """
    Combined wrapper:
    - Optional Correction Module (correction_method), see enhancement_methods.py:
      'none' | 'mqtone' (proposed C2) | 'gamma' | 'clahe'
    - DetectionModel YOLOv8n / YOLOv11n
    """
    def __init__(self, weights_path: str = "yolov8n.pt", correction_method: str = "mqtone"):
        super().__init__()
        correction_method = correction_method.lower()
        if correction_method == 'icnet':
            correction_method = 'mqtone'
        if correction_method not in CORRECTION_METHODS:
            raise ValueError(f"Invalid correction_method: '{correction_method}'. "
                              f"Valid options: {CORRECTION_METHODS}")
        self.correction_method = correction_method
        self.use_correction = (correction_method != 'none')
        # Non-learnable modules (Gamma/CLAHE): applied as fixed transformations,
        # not trained, and omit Photometric Loss (see train_one_epoch_c2).
        self.is_learnable_correction = self.use_correction and (correction_method not in NON_LEARNABLE_METHODS)

        # 1. Correction Module
        self.corrector = build_correction_module(correction_method)

        # 2. YOLOv8 DetectionModel
        yolo = YOLO(weights_path)
        # IMPORTANT: object.__setattr__ instead of self.yolo = yolo.
        # ultralytics.YOLO inherits nn.Module, so standard attribute assignment
        # causes PyTorch to register it as a submodule. Then pipeline.train()/eval()
        # would recursively call yolo.train(mode) -- which Ultralytics overrides to execute
        # full training pipeline rather than toggling the .training flag, raising
        # "TypeError: 'bool' object is not callable". Using object.__setattr__ keeps
        # self.yolo as a regular attribute rather than a tracked submodule.
        object.__setattr__(self, 'yolo', yolo)
        self.detector = yolo.model
        self.detector.args = get_cfg(DEFAULT_CFG)

        # Release .pt weights from Ultralytics undergo strip_optimizer(): parameters
        # have requires_grad=False and may be converted to half-precision. Because our
        # pipeline uses a custom training loop (without the default Trainer),
        # we explicitly restore requires_grad=True so loss has grad_fn during .backward().
        self.detector = self.detector.float()
        for p in self.detector.parameters():
            p.requires_grad = True

    def forward(self, x: torch.Tensor, return_corrected: bool = False):
        corrected = None
        if self.use_correction and self.corrector is not None:
            x = self.corrector(x)
            corrected = x
        preds = self.detector(x)
        if return_corrected:
            return preds, corrected
        return preds


# ─────────────────────────────────────────────────────────────────────────────
# Train 1 Epoch
# ─────────────────────────────────────────────────────────────────────────────
def train_one_epoch_c2(
    pipeline: C2DetectionPipeline,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: v8DetectionLoss,
    device: str,
    epoch: int,
    lambda_photo: float = 1.0
):
    pipeline.train()
    if pipeline.corrector is not None:
        pipeline.corrector.train()

    total_loss_sum = 0.0
    det_loss_sum = 0.0
    photo_loss_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        imgs = batch['img'].to(device)
        batch_yolo = {
            'img': imgs,
            'cls': batch['cls'].to(device),
            'bboxes': batch['bboxes'].to(device),
            'batch_idx': batch['batch_idx'].to(device)
        }

        optimizer.zero_grad()

        # 1. Forward Pass (return_corrected=True to reuse correction module output
        # for photometric supervision loss below without redundant forward pass)
        preds, corrected_imgs = pipeline(imgs, return_corrected=True)

        # 2. Detection Loss
        loss_components, loss_items = loss_fn(preds, batch_yolo)
        L_detection = loss_components.sum()

        # 2b. Photometric Supervision Loss (applied to LEARNABLE modules: MQTone,
        # IAT, Zero-DCE, Zero-DCE++ -- NOT applied to non-learnable classical methods
        # such as Gamma/CLAHE) -- L1 between corrected image and CLEAN REFERENCE
        # image (clean_img, provided via orig_img_path from augment.py -- strictly pixel-aligned).
        # For 'original' images, clean_img is identical -> enforces near-identity behavior.
        # For degraded images -> actively guides the module back to clean reference.
        L_photo = torch.tensor(0.0, device=device)
        if pipeline.is_learnable_correction and corrected_imgs is not None and batch.get('clean_img') is not None:
            clean_imgs = batch['clean_img'].to(device)
            L_photo = F.l1_loss(corrected_imgs, clean_imgs)

        # 3. Total Loss (Detection Loss + Photometric Supervision Loss for learnable
        # modules; Consistency Loss has been eliminated from the pipeline)
        L_total = L_detection \
            + (lambda_photo * L_photo if pipeline.is_learnable_correction else 0.0)

        L_total.backward()

        # Gradient clipping: clips gradients to max_norm=10.0 for training stability,
        # preventing sudden gradient spikes from destabilizing correction parameters.
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']],
            max_norm=10.0
        )

        optimizer.step()

        total_loss_sum += L_total.item()
        det_loss_sum += L_detection.item()
        photo_loss_sum += L_photo.item() if pipeline.is_learnable_correction else 0.0
        n_batches += 1

    avg_total = total_loss_sum / max(1, n_batches)
    avg_det = det_loss_sum / max(1, n_batches)
    avg_photo = photo_loss_sum / max(1, n_batches)

    return avg_total, avg_det, avg_photo


def _set_bn_eval(module: nn.Module):
    """
    Sets BatchNorm layers specifically to eval() mode (using accumulated running_mean/var
    without updating stats on the current batch), while keeping the rest of the model
    in train() mode -- necessary for validate_one_epoch_c2 below.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate 1 Epoch on Validation Set (NO weight updates)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch_c2(
    pipeline: C2DetectionPipeline,
    dataloader: DataLoader,
    loss_fn: v8DetectionLoss,
    device: str
) -> float:
    """
    Computes average Detection Loss on the validation split (held-out 20%,
    augmented independently), used as the BEST CHECKPOINT SELECTION criterion
    per epoch (standard early stopping / model selection). Does NOT update weights;
    does NOT serve as final test report (final metrics come from
    evaluate_c2_pipeline() on Locked Test Set).

    Key technical notes:
    - Multi-scale raw outputs from Ultralytics Detect head are only returned
      when `self.training=True`; in .eval() mode it returns decoded boxes -- incorrect
      format for v8DetectionLoss. Hence this function keeps pipeline in .train() mode.
    - To prevent validation stats leakage into BatchNorm running stats,
      _set_bn_eval() freezes BN layers into eval() while Detect head outputs training format.
    - torch.no_grad() prevents computation graph construction and saves memory.
    - Selection criterion relies on pure Detection Loss for fair comparison across methods.
    """
    pipeline.train()
    _set_bn_eval(pipeline)

    total_loss_sum = 0.0
    n_batches = 0

    for batch in dataloader:
        imgs = batch['img'].to(device)
        batch_yolo = {
            'img': imgs,
            'cls': batch['cls'].to(device),
            'bboxes': batch['bboxes'].to(device),
            'batch_idx': batch['batch_idx'].to(device)
        }

        preds = pipeline(imgs)
        loss_components, _ = loss_fn(preds, batch_yolo)
        L_detection = loss_components.sum()

        total_loss_sum += L_detection.item()
        n_batches += 1

    return total_loss_sum / max(1, n_batches)


# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Evaluation for C2
# ─────────────────────────────────────────────────────────────────────────────
def evaluate_c2_pipeline(pipeline: C2DetectionPipeline, df: pd.DataFrame, condition_name: str, device: str, conf_thresh: float = 0.25):
    pipeline.eval()
    if pipeline.corrector is not None:
        pipeline.corrector.eval()

    y_true_denom = []
    y_pred_denom = []
    denom_ious = []
    total_gt_banknotes = 0
    matched_gt_banknotes = 0
    missed_banknotes = 0

    n_total_imgs = len(df)
    n_gt_intact_imgs = 0
    tp_tear_img = 0
    fp_tear_img = 0
    fn_tear_img = 0
    tn_tear_img = 0

    total_gt_tear_boxes = 0
    matched_gt_tear_boxes = 0
    tear_box_ious = []

    for _, row in df.iterrows():
        img_path = str(row['img_path'])
        is_torn_gt = (str(row['is_torn']).lower() == 'true')
        if not is_torn_gt:
            n_gt_intact_imgs += 1

        gt_denom_id = None
        gt_banknote_box = None
        gt_tear_boxes = []
        lbl_path = row.get('lbl_path')

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

        if gt_denom_id is not None:
            y_true_denom.append(gt_denom_id)
            total_gt_banknotes += 1

        total_gt_tear_boxes += len(gt_tear_boxes)

        # Inference
        bgr = cv2.imread(img_path)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(cv2.resize(rgb, (640, 640))).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

        with torch.no_grad():
            if pipeline.use_correction and pipeline.corrector is not None:
                img_tensor = pipeline.corrector(img_tensor)
            # Use predict method of high-level YOLO wrapper (self.yolo), not self.detector
            # (self.detector is a raw DetectionModel without standard Results bounding boxes)
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

        has_pred_tear = (len(pred_tear_boxes) > 0)

        # 1. Evaluate Denomination & Banknote Localization
        if gt_denom_id is not None:
            if best_denom_id is not None:
                y_pred_denom.append(best_denom_id)
                if gt_banknote_box is not None and best_banknote_box is not None:
                    b_iou = box_iou(gt_banknote_box, best_banknote_box)
                    denom_ious.append(b_iou)
                    if b_iou >= 0.5:
                        matched_gt_banknotes += 1
            else:
                y_pred_denom.append(-1)
                missed_banknotes += 1

        # 2. Binary Tear Evaluation
        if is_torn_gt and has_pred_tear:
            tp_tear_img += 1
        elif (not is_torn_gt) and has_pred_tear:
            fp_tear_img += 1
        elif is_torn_gt and (not has_pred_tear):
            fn_tear_img += 1
        else:
            tn_tear_img += 1

        # 3. Evaluate Tear Bounding Boxes
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

    # Compute evaluation metrics
    if len(y_true_denom) > 0:
        correct_denom = sum(1 for yt, yp in zip(y_true_denom, y_pred_denom) if yt == yp)
        denom_acc = (correct_denom / len(y_true_denom)) * 100.0
    else:
        denom_acc = 0.0

    banknote_avg_iou = (np.mean(denom_ious) * 100.0) if len(denom_ious) > 0 else 0.0
    banknote_map50 = (matched_gt_banknotes / max(1, total_gt_banknotes)) * 100.0 if total_gt_banknotes > 0 else 0.0
    banknote_map50_95 = banknote_avg_iou * (banknote_map50 / 100.0)
    banknote_miss_rate = (missed_banknotes / max(1, total_gt_banknotes)) * 100.0 if total_gt_banknotes > 0 else 0.0

    binary_tear_acc = ((tp_tear_img + tn_tear_img) / max(1, n_total_imgs)) * 100.0
    false_alarm_rate = (fp_tear_img / max(1, n_gt_intact_imgs)) * 100.0 if n_gt_intact_imgs > 0 else 0.0

    # Torn F1 (image-level binary classification: "torn or intact") -- using
    # tp/fp/fn counts from step 2 above. Balances Precision (correct tear alarms)
    # and Recall (true torn banknotes detected).
    torn_precision = (tp_tear_img / (tp_tear_img + fp_tear_img) * 100.0) if (tp_tear_img + fp_tear_img) > 0 else 0.0
    torn_recall = (tp_tear_img / (tp_tear_img + fn_tear_img) * 100.0) if (tp_tear_img + fn_tear_img) > 0 else 0.0
    torn_f1 = (2 * torn_precision * torn_recall / (torn_precision + torn_recall)) if (torn_precision + torn_recall) > 0 else 0.0

    has_real_tears = (total_gt_tear_boxes > 0)
    tear_box_avg_iou = (np.mean(tear_box_ious) * 100.0) if len(tear_box_ious) > 0 else 0.0
    tear_map50 = (matched_gt_tear_boxes / total_gt_tear_boxes) * 100.0 if has_real_tears else 0.0
    tear_map50_95 = tear_box_avg_iou * (tear_map50 / 100.0) if has_real_tears else 0.0
    tear_miss_rate = ((total_gt_tear_boxes - matched_gt_tear_boxes) / total_gt_tear_boxes) * 100.0 if has_real_tears else 0.0

    return {
        'condition': condition_name,
        'num_samples': n_total_imgs,
        'has_real_tears': has_real_tears,
        'accuracy_denom': round(denom_acc, 2),
        'banknote_avg_iou': round(banknote_avg_iou, 2),
        'banknote_map50': round(banknote_map50, 2),
        'banknote_map50_95': round(banknote_map50_95, 2),
        'banknote_miss_rate': round(banknote_miss_rate, 2),
        'binary_tear_acc': round(binary_tear_acc, 2),
        'false_alarm_rate': round(false_alarm_rate, 2),
        'torn_precision': round(torn_precision, 2),
        'torn_recall': round(torn_recall, 2),
        'torn_f1': round(torn_f1, 2),
        'mAP50_tear': round(tear_map50, 2) if has_real_tears else None,
        'mAP50_95_tear': round(tear_map50_95, 2) if has_real_tears else None,
        'avg_IoU_tear': round(tear_box_avg_iou, 2) if has_real_tears else None,
        'miss_rate_tear': round(tear_miss_rate, 2) if has_real_tears else None
    }