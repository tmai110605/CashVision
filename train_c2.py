#!/usr/bin/env python3
"""
train_c2.py — Custom PyTorch Training & Evaluation Engine cho C2
Tích hợp:
  - Low-level DetectionModel (YOLOv8n)
  - Registry Correction Module: IC-Net (đề xuất của C2) + các phương pháp so
    sánh cổ điển KHÔNG học được (Gamma Correction, CLAHE) — xem
    enhancement_methods.py. (Zero-DCE++/SCI/Retinexformer đã bị loại bỏ hoàn
    toàn khỏi registry, xem ghi chú lý do trong enhancement_methods.py)
  - Photometric Supervision Loss cho các module HỌC ĐƯỢC (so với ảnh tham
    chiếu sạch); các module cổ điển (Gamma/CLAHE) không có loss này
  - Custom training loop & Evaluation engine (bao gồm Torn F1 / Torn mAP)

Ghi chú: Consistency Loss (trên cặp ảnh {light, dark}) và PairBatchSampler đã
được loại bỏ hoàn toàn khỏi pipeline này.
"""

import os
import sys
import time
from pathlib import Path

# Fix UTF-8 encoding trên Windows
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
    Dataset nạp ảnh, bounding boxes định dạng YOLO, và thông tin pair_id/aug_type
    (metadata mô tả, không còn dùng cho Consistency Loss -- đã bị loại bỏ khỏi pipeline).

    load_clean_ref: nếu True, đồng thời nạp ảnh THAM CHIẾU SẠCH (chưa degrade) tương
    ứng qua cột 'orig_img_path' trong metadata (do augment.py ghi -- xem task 1/2).
    Dùng làm target cho photometric supervision loss của IC-Net: vì mọi phép biến đổi
    trong augment.py (brightness/contrast/gamma toàn cục + glare/shadow cục bộ) đều
    KHÔNG làm lệch không gian ảnh (không crop/rotate/flip), ảnh degrade và ảnh gốc
    luôn align pixel-to-pixel, nên có thể dùng trực tiếp làm cặp (input, target) cho
    một loss tái tạo/identity, tách biệt khỏi detection loss. Mặc định False để không
    tốn thêm I/O ở các config không dùng IC-Net (A/C).
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
                    # Metadata cũ (trước task 1/2) có thể chưa có cột này -- fallback
                    # an toàn: coi ảnh hiện tại là ảnh sạch của chính nó (identity),
                    # thay vì crash. Ảnh 'original' luôn đúng nghĩa dù có fallback hay không.
                    orig_img_path = str(img_p)

            # Đọc nhãn YOLO
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
            # Ảnh gốc thường trùng lặp nhiều lần trong batch (mỗi ảnh gốc sinh ra
            # ~5 bản augment cùng orig_img_path) -- chấp nhận đọc lại nhiều lần vì
            # đơn giản & an toàn hơn cache thủ công trong Dataset (Dataset có thể
            # được fork sang nhiều worker process của DataLoader).
            out['clean_img'] = self._load_img_tensor(item['orig_img_path'])

        return out


def cashvision_collate_fn(batch):
    """
    Collate function gộp batch cho DetectionModel:
    Tạo tensor 'img' (B, 3, H, W), 'cls' (N, 1), 'bboxes' (N, 4), 'batch_idx' (N,)
    Nếu dataset bật load_clean_ref, gộp thêm 'clean_img' (B, 3, H, W) -- target sạch
    dùng cho photometric supervision loss của IC-Net (task 1).
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
# Model Wrapper: Correction Module (IC-Net / Gamma / CLAHE / None) + YOLOv8
# DetectionModel
# ─────────────────────────────────────────────────────────────────────────────
class C2DetectionPipeline(nn.Module):
    """
    Wrapper kết hợp:
    - Correction Module tùy chọn (correction_method), xem enhancement_methods.py:
      'none' | 'icnet' (đề xuất C2) | 'gamma' | 'clahe'
    - DetectionModel YOLOv8n
    """
    def __init__(self, weights_path: str = "yolov8n.pt", correction_method: str = "icnet"):
        super().__init__()
        correction_method = correction_method.lower()
        if correction_method not in CORRECTION_METHODS:
            raise ValueError(f"correction_method không hợp lệ: '{correction_method}'. "
                              f"Các lựa chọn hợp lệ: {CORRECTION_METHODS}")
        self.correction_method = correction_method
        self.use_correction = (correction_method != 'none')
        # Module KHÔNG có tham số học được (Gamma/CLAHE): áp dụng như 1 phép biến đổi
        # cố định, không train riêng, không có Photometric Loss (xem train_one_epoch_c2).
        self.is_learnable_correction = self.use_correction and (correction_method not in NON_LEARNABLE_METHODS)

        # 1. Correction Module
        self.corrector = build_correction_module(correction_method)

        # 2. YOLOv8 DetectionModel
        yolo = YOLO(weights_path)
        # QUAN TRỌNG: object.__setattr__ thay vì self.yolo = yolo.
        # ultralytics.YOLO (class Model) cũng kế thừa nn.Module, nên gán bình thường sẽ
        # khiến PyTorch tự đăng ký nó làm submodule con. Khi đó pipeline.train()/eval()
        # sẽ đệ quy gọi yolo.train(mode)/yolo.eval() -- nhưng Model.train() bị Ultralytics
        # override để nghĩa là "chạy training pipeline đầy đủ" (nhận trainer=None, **kwargs),
        # không phải bật cờ .training như nn.Module chuẩn -> gây lỗi
        # "TypeError: 'bool' object is not callable". Dùng object.__setattr__ để lưu
        # self.yolo như attribute thường, không bị PyTorch quản lý như submodule.
        object.__setattr__(self, 'yolo', yolo)
        self.detector = yolo.model
        self.detector.args = get_cfg(DEFAULT_CFG)

        # File .pt release của Ultralytics đã qua strip_optimizer(): toàn bộ tham số
        # bị set requires_grad=False và ép về half-precision (FP16), vì file này chỉ
        # dùng làm checkpoint khởi đầu cho Trainer chính thức của Ultralytics (Trainer
        # tự bật lại requires_grad trước khi train). Do pipeline này tự viết training
        # loop riêng (không qua Trainer), cần bật lại thủ công ở đây, nếu không
        # loss sẽ không có grad_fn khi gọi .backward().
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
# Huấn luyện 1 Epoch
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

        # 1. Forward Pass (return_corrected=True để tái sử dụng output correction
        # module cho photometric loss bên dưới, tránh phải forward lần thứ 2)
        preds, corrected_imgs = pipeline(imgs, return_corrected=True)

        # 2. Detection Loss
        loss_components, loss_items = loss_fn(preds, batch_yolo)
        L_detection = loss_components.sum()

        # 2b. Photometric Supervision Loss (chỉ áp dụng cho module HỌC ĐƯỢC: IC-Net --
        # method HỌC ĐƯỢC duy nhất còn lại sau khi loại Zero-DCE++/SCI/Retinexformer
        # khỏi registry -- KHÔNG áp dụng cho module cổ điển không có tham số như
        # Gamma/CLAHE) -- L1 giữa ảnh đã sửa và ảnh THAM CHIẾU SẠCH
        # (clean_img, do augment.py cung cấp qua orig_img_path -- luôn align pixel vì
        # augment chỉ đổi photometric, không đổi hình học). Với ảnh 'original'
        # clean_img == chính nó -> ép module gần identity trên ảnh đã đẹp sẵn. Với
        # ảnh degrade (photo/light/dark/torn/local_only) -> ép module thực sự sửa về
        # đúng ảnh gốc, thay vì chỉ được tối ưu gián tiếp qua detection loss.
        L_photo = torch.tensor(0.0, device=device)
        if pipeline.is_learnable_correction and corrected_imgs is not None and batch.get('clean_img') is not None:
            clean_imgs = batch['clean_img'].to(device)
            L_photo = F.l1_loss(corrected_imgs, clean_imgs)

        # 3. Tổng Loss (chỉ Detection Loss + Photometric Supervision Loss cho module
        # HỌC ĐƯỢC; Consistency Loss đã bị loại bỏ hoàn toàn khỏi pipeline)
        L_total = L_detection \
            + (lambda_photo * L_photo if pipeline.is_learnable_correction else 0.0)

        L_total.backward()

        # Gradient clipping: Ultralytics Trainer gốc luôn áp dụng bước này tự động
        # (max_norm=10.0), nhưng custom loop này không có -- thiếu nó có thể khiến
        # 1 batch có gradient bất thường (đặc biệt đầu training, khi chỉ 1 trong 2
        # loss signal hoạt động) đẩy tham số (đặc biệt correction module) lệch khỏi
        # vùng ổn định.
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
    Đặt riêng các lớp BatchNorm về chế độ eval() (dùng running_mean/running_var
    đã tích luỹ, KHÔNG cập nhật running stats theo batch hiện tại), trong khi phần
    còn lại của model vẫn ở train() -- cần thiết cho validate_one_epoch_c2 bên dưới.
    """
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval()


# ─────────────────────────────────────────────────────────────────────────────
# Đánh giá 1 Epoch trên tập Validation (KHÔNG cập nhật trọng số)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate_one_epoch_c2(
    pipeline: C2DetectionPipeline,
    dataloader: DataLoader,
    loss_fn: v8DetectionLoss,
    device: str
) -> float:
    """
    Tính Detection Loss trung bình trên tập validation của fold (20% còn lại,
    đã augment riêng), dùng làm tiêu chí CHỌN CHECKPOINT tốt nhất mỗi epoch
    (giống early-stopping/model-selection chuẩn). KHÔNG cập nhật trọng số,
    KHÔNG dùng để báo cáo kết quả cuối cùng (kết quả cuối luôn đến từ
    evaluate_c2_pipeline() trên Locked Test Set).

    Lưu ý kỹ thuật quan trọng:
    - Đầu ra thô (raw, đa tỉ lệ) của Detect head trong Ultralytics chỉ được trả về
      khi `self.training=True` (tức model đang ở .train()); ở .eval() model trả về
      bounding-box đã decode -- SAI FORMAT mà v8DetectionLoss cần. Do đó hàm này
      GIỮ NGUYÊN pipeline.train() (không gọi pipeline.eval()) để lấy đúng format.
    - Nhưng nếu để nguyên train() thông thường, mỗi forward pass trên batch validation
      sẽ tiếp tục cập nhật running_mean/running_var của BatchNorm theo chính batch
      validation đó -> rò rỉ thống kê tập validation vào các layer BatchNorm, ảnh
      hưởng ngược lại hành vi model lúc inference thật (evaluate_c2_pipeline dùng
      .eval() với running stats này). Để tránh rò rỉ, hàm _set_bn_eval() ở trên
      đóng băng riêng các lớp BatchNorm về eval() (dùng running stats có sẵn, không
      cập nhật), trong khi Detect head vẫn "nghĩ" mình đang training để trả đúng format.
    - torch.no_grad() đảm bảo không build computation graph -> không tốn bộ nhớ,
      không có gradient nào được tính hay áp dụng.

    Việc chọn checkpoint dựa trên avg Detection Loss thuần (không cộng Photometric
    Loss) để tiêu chí so sánh nhất quán giữa 2 config A (baseline) / B (IC-Net).
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
# Đánh giá Chi tiết Toàn diện C2
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
            # Dùng predict method của wrapper YOLO cấp cao (self.yolo), không phải self.detector
            # (self.detector chỉ là DetectionModel thô, .predict() của nó khác chữ ký và
            # không trả về Results object có .boxes như code bên dưới cần)
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

        # 1. Đánh giá Mệnh giá & Định vị tờ tiền
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

        # 2. Đánh giá Nhị phân Rách
        if is_torn_gt and has_pred_tear:
            tp_tear_img += 1
        elif (not is_torn_gt) and has_pred_tear:
            fp_tear_img += 1
        elif is_torn_gt and (not has_pred_tear):
            fn_tear_img += 1
        else:
            tn_tear_img += 1

        # 3. Đánh giá Box Vết rách
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

    # Tính toán chỉ số
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

    # Torn F1 (ở cấp độ ẢNH, phân loại nhị phân "có rách hay không") -- dùng chung
    # tp/fp/fn đã đếm ở bước 2 phía trên. Đây là chỉ số MỚI, bổ sung theo yêu cầu,
    # đo cân bằng giữa Precision (bao nhiêu cảnh báo rách là đúng) và Recall (bắt
    # được bao nhiêu ảnh rách thật sự) -- false_alarm_rate ở trên chỉ phản ánh 1 vế.
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