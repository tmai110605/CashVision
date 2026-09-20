import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from quality_gate import QualityGate
from mqtone import MQTone
from train_c2 import C2DetectionPipeline, CLASS_NAMES, DENOM_CLASSES, TORN_CLASS_ID


def preprocess_thumbnail(frame_bgr: np.ndarray, thumbnail_size: int = 64) -> torch.Tensor:
    """Preprocess frame to 64x64 RGB float tensor [0, 1] for QualityGate / MQTone."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (thumbnail_size, thumbnail_size))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return tensor


def detect_note_presence_fast(frame_bgr: np.ndarray) -> bool:
    """
    Fast heuristic to check if a banknote or object is present in frame:
    - Checks standard deviation of luminance & color variance.
    - If camera is covered or pointed at completely blank wall with no texture, returns False.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))
    # Threshold for presence of visual texture / object
    return std_dev > 15.0


class LightPackage:
    """
    Lightweight package running Quality-Gate on 64x64 thumbnail (~0.2 - 0.5 ms).
    Estimates illumination quality (Good, Underexposed, Overexposed) and banknote presence.
    """
    def __init__(
        self,
        weights_path: str = "checkpointyolov8n/quality_gate_weights.pt",
        device: str = "cpu"
    ):
        self.device = device
        self.thumbnail_size = 64
        self.model = QualityGate(thumbnail_size=self.thumbnail_size).to(device)
        self.model.eval()

        p = Path(weights_path)
        if not p.exists():
            # Fallback path check
            alt_p = Path("resultsv8n/quality_gate_weights.pt")
            if alt_p.exists():
                p = alt_p

        if p.exists():
            state_dict = torch.load(str(p), map_location=device)
            self.model.load_state_dict(state_dict)
            self.loaded = True
        else:
            self.loaded = False
            print(f"⚠️ [LightPackage] Warning: QualityGate weights not found at {weights_path}")

        self.labels_map = {0: 'good', 1: 'underexposed', 2: 'overexposed'}

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, tau: float = 0.6) -> Dict[str, Any]:
        """
        Runs Quality-Gate inference.
        Returns:
            quality: float (probability of good quality or composite score in [0, 1])
            has_note: bool
            pred_class: 'good' | 'underexposed' | 'overexposed'
            confidence: float
            probs: [p_good, p_under, p_over]
            is_passed: bool (True if frame is acceptable for detection)
        """
        t0 = time.perf_counter()
        has_note = detect_note_presence_fast(frame_bgr)
        tensor = preprocess_thumbnail(frame_bgr, self.thumbnail_size).to(self.device)

        logits = self.model(tensor)
        probs = F.softmax(logits, dim=-1)[0]
        conf, pred_cls = torch.max(probs, dim=-1)

        pred_cls_idx = int(pred_cls.item())
        conf_val = float(conf.item())
        prob_good = float(probs[0].item())
        prob_under = float(probs[1].item())
        prob_over = float(probs[2].item())

        # Frame passes if predicted good or degradation is below confidence threshold tau
        is_blocked = (pred_cls_idx != 0) and (conf_val > tau)
        is_passed = not is_blocked

        # Composite quality score: P(good) adjusted by pass status
        quality_score = prob_good if is_passed else (1.0 - conf_val)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return {
            "quality": float(quality_score),
            "prob_good": prob_good,
            "prob_under": prob_under,
            "prob_over": prob_over,
            "pred_class": self.labels_map.get(pred_cls_idx, 'unknown'),
            "confidence": conf_val,
            "is_passed": bool(is_passed),
            "has_note": bool(has_note),
            "latency_ms": float(latency_ms)
        }


class FullPackage:
    """
    Full Package: SOTA Illumination Correction (MQTone) + YOLOv8 Detection Pipeline.
    Loads checkpointyolov8n/full_pipeline_weights.pt.
    """
    def __init__(
        self,
        weights_path: str = "checkpointyolov8n/full_pipeline_weights.pt",
        yolo_base: str = "yolov8n.pt",
        device: str = "cpu",
        img_size: int = 640
    ):
        self.device = device
        self.img_size = img_size
        self.pipeline = C2DetectionPipeline(weights_path=yolo_base, correction_method='mqtone').to(device)
        self.pipeline.eval()

        p = Path(weights_path)
        if p.exists():
            state_dict = torch.load(str(p), map_location=device)
            self.pipeline.load_state_dict(state_dict)
            self.loaded = True
        else:
            self.loaded = False
            print(f"⚠️ [FullPackage] Warning: weights not found at {weights_path}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, conf_thresh: float = 0.25) -> Dict[str, Any]:
        """
        Runs MQTone enhancement followed by YOLO detection.
        Returns detailed predictions and corrected image.
        """
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        # 1. MQTone correction
        t_ic_start = time.perf_counter()
        if self.pipeline.use_correction and self.pipeline.corrector is not None:
            corrected_tensor = self.pipeline.corrector(img_tensor)
        else:
            corrected_tensor = img_tensor
        t_ic_end = time.perf_counter()
        ic_latency_ms = (t_ic_end - t_ic_start) * 1000.0

        # Convert corrected image back to RGB numpy array
        corrected_np = (corrected_tensor[0].permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        # 2. YOLO Detection
        t_yolo_start = time.perf_counter()
        results = self.pipeline.yolo.predict(
            corrected_tensor,
            conf=conf_thresh,
            device=self.device,
            verbose=False
        )[0]
        t_yolo_end = time.perf_counter()
        yolo_latency_ms = (t_yolo_end - t_yolo_start) * 1000.0

        # Parse detections
        best_denom_id = None
        best_denom_name = None
        best_denom_conf = -1.0
        best_denom_box = None
        tear_detections = []

        for b in results.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()  # [x1, y1, x2, y2] relative to 640x640

            # Rescale boxes to original frame dimensions
            scale_x = w_orig / float(self.img_size)
            scale_y = h_orig / float(self.img_size)
            orig_box = [
                xyxy[0] * scale_x,
                xyxy[1] * scale_y,
                xyxy[2] * scale_x,
                xyxy[3] * scale_y
            ]

            if cid in DENOM_CLASSES:
                if conf > best_denom_conf:
                    best_denom_conf = conf
                    best_denom_id = cid
                    best_denom_name = CLASS_NAMES[cid]
                    best_denom_box = orig_box
            elif cid == TORN_CLASS_ID:
                tear_detections.append({
                    "box": orig_box,
                    "conf": conf,
                    "class_name": "torn"
                })

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        is_torn_pred = len(tear_detections) > 0

        # Human-readable speech summary for VI users
        if best_denom_name is not None:
            denom_str = f"{int(best_denom_name) * 1000:,} VND"
            if is_torn_pred:
                speech_text = f"{denom_str} note, tear defect detected"
            else:
                speech_text = f"{denom_str} note, intact"
        else:
            speech_text = "Unclear denomination"

        return {
            "denomination_id": best_denom_id,
            "denomination_name": best_denom_name,
            "denomination_conf": float(best_denom_conf) if best_denom_conf > 0 else 0.0,
            "denomination_box": best_denom_box,
            "is_torn": is_torn_pred,
            "tear_detections": tear_detections,
            "num_tears": len(tear_detections),
            "speech_text": speech_text,
            "corrected_image": corrected_np,
            "mqtone_latency_ms": float(ic_latency_ms),
            "ic_latency_ms": float(ic_latency_ms),
            "yolo_latency_ms": float(yolo_latency_ms),
            "total_latency_ms": float(total_latency_ms)
        }


class DetectorOnlyPackage:
    """
    Detector-Only Baseline (B1): YOLOv8n without MQTone and without Quality-Gate.
    Loads resultsv8n/final_config_none/fold_1/full_pipeline_weights.pt.
    """
    def __init__(
        self,
        weights_path: str = "resultsv8n/final_config_none/fold_1/full_pipeline_weights.pt",
        yolo_base: str = "yolov8n.pt",
        device: str = "cpu",
        img_size: int = 640
    ):
        self.device = device
        self.img_size = img_size
        self.pipeline = C2DetectionPipeline(weights_path=yolo_base, correction_method='none').to(device)
        self.pipeline.eval()

        p = Path(weights_path)
        if not p.exists():
            alt = Path("checkpointyolov8n/full_pipeline_weights.pt")
            if alt.exists():
                p = alt

        if p.exists():
            state_dict = torch.load(str(p), map_location=device)
            det_state_dict = {k: v for k, v in state_dict.items() if not k.startswith("corrector.")}
            self.pipeline.load_state_dict(det_state_dict, strict=False)
            self.loaded = True
        else:
            self.loaded = False
            print(f"⚠️ [DetectorOnlyPackage] Warning: weights not found at {weights_path}")

    @torch.no_grad()
    def infer(self, frame_bgr: np.ndarray, conf_thresh: float = 0.25) -> Dict[str, Any]:
        """
        Runs pure YOLO detection directly on raw image without photometric enhancement.
        """
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = rgb.shape[:2]
        resized = cv2.resize(rgb, (self.img_size, self.img_size))
        img_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float().to(self.device) / 255.0

        results = self.pipeline.yolo.predict(
            img_tensor,
            conf=conf_thresh,
            device=self.device,
            verbose=False
        )[0]

        best_denom_id = None
        best_denom_name = None
        best_denom_conf = -1.0
        best_denom_box = None
        tear_detections = []

        for b in results.boxes:
            cid = int(b.cls.item())
            conf = float(b.conf.item())
            xyxy = b.xyxy[0].tolist()

            scale_x = w_orig / float(self.img_size)
            scale_y = h_orig / float(self.img_size)
            orig_box = [
                xyxy[0] * scale_x,
                xyxy[1] * scale_y,
                xyxy[2] * scale_x,
                xyxy[3] * scale_y
            ]

            if cid in DENOM_CLASSES:
                if conf > best_denom_conf:
                    best_denom_conf = conf
                    best_denom_id = cid
                    best_denom_name = CLASS_NAMES[cid]
                    best_denom_box = orig_box
            elif cid == TORN_CLASS_ID:
                tear_detections.append({
                    "box": orig_box,
                    "conf": conf,
                    "class_name": "torn"
                })

        total_latency_ms = (time.perf_counter() - t0) * 1000.0
        is_torn_pred = len(tear_detections) > 0

        if best_denom_name is not None:
            denom_str = f"{int(best_denom_name) * 1000:,} VND"
            if is_torn_pred:
                speech_text = f"{denom_str} note, tear defect detected"
            else:
                speech_text = f"{denom_str} note, intact"
        else:
            speech_text = "Unclear denomination"

        return {
            "denomination_id": best_denom_id,
            "denomination_name": best_denom_name,
            "denomination_conf": float(best_denom_conf) if best_denom_conf > 0 else 0.0,
            "denomination_box": best_denom_box,
            "is_torn": is_torn_pred,
            "tear_detections": tear_detections,
            "num_tears": len(tear_detections),
            "speech_text": speech_text,
            "corrected_image": None,
            "ic_latency_ms": 0.0,
            "yolo_latency_ms": float(total_latency_ms),
            "total_latency_ms": float(total_latency_ms)
        }
