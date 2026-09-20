#!/usr/bin/env python3
"""
quality_gate.py — Quality-Gate Module
Ultra-lightweight sensory classifier running per frame to assess optical quality:
  Class 0: good (indoor, outdoor)
  Class 1: underexposed (backlight)
  Class 2: overexposed (overexposed)
Modules: Standalone training, tau threshold sweep, and streaming session simulation.
"""

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2


class QualityGate(nn.Module):
    """
    Quality-Gate Sub-network:
    Input: 64x64x3 thumbnail
    Output: Logits for 3 photometric classes: [good, underexposed, overexposed]
    """
    def __init__(self, thumbnail_size: int = 64):
        super().__init__()
        self.thumbnail_size = thumbnail_size

        self.net = nn.Sequential(
            # 64x64x3 -> 32x32x12
            nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(12),
            nn.ReLU(inplace=True),

            # 32x32x12 -> 16x16x24
            nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            # 16x16x24 -> 8x8x24
            nn.Conv2d(24, 24, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(24, 3)  # 3 classes: good, underexposed, overexposed
        )

    def forward(self, x: torch.Tensor):
        if x.shape[-2:] != (self.thumbnail_size, self.thumbnail_size):
            x = F.interpolate(x, size=(self.thumbnail_size, self.thumbnail_size),
                              mode='bilinear', align_corners=False)
        return self.net(x)

    def predict_quality(self, x: torch.Tensor, tau: float = 0.6):
        """
        Decision rule:
        - If argmax != 0 ('good') AND confidence > tau -> Defer frame (is_passed = False)
        - Otherwise -> Pass frame to deep detector (is_passed = True)
        """
        self.eval()
        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=-1)
            conf, pred_cls = torch.max(probs, dim=-1)

            is_blocked = (pred_cls != 0) & (conf > tau)
            is_passed = ~is_blocked

            labels_map = {0: 'good', 1: 'underexposed', 2: 'overexposed'}
            return is_passed, pred_cls, conf, probs


class QualityGateDataset(Dataset):
    """Dataset reader assigning 3-class photometric quality labels from environmental metadata."""
    def __init__(self, df: pd.DataFrame, thumbnail_size: int = 64):
        self.samples = []
        self.thumbnail_size = thumbnail_size

        for _, row in df.iterrows():
            cond = str(row['condition']).lower()
            if cond in ['indoor', 'outdoor', 'torn_clean']:
                label = 0  # good
            elif cond in ['backlight']:
                label = 1  # underexposed
            elif cond in ['overexposed', 'torn_bright']:
                label = 2  # overexposed
            else:
                label = 0

            self.samples.append((str(row['img_path']), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        bgr = cv2.imread(img_path)
        if bgr is None:
            # Fallback empty tensor
            tensor = torch.zeros((3, self.thumbnail_size, self.thumbnail_size), dtype=torch.float32)
        else:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (self.thumbnail_size, self.thumbnail_size))
            tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0

        return tensor, label


def train_quality_gate(train_df: pd.DataFrame, val_df: pd.DataFrame, device: str = 'cuda', epochs: int = 25, batch_size: int = 32, lr: float = 1e-3):
    """Train Quality-Gate independently using Cross-Entropy Loss."""
    train_dataset = QualityGateDataset(train_df)
    val_dataset = QualityGateDataset(val_df)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = QualityGate().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    best_weights = None

    print(f"\n[QUALITY-GATE] Starting training ({len(train_dataset)} train samples, {len(val_dataset)} val samples)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = (correct / total) * 100.0

        # Validation evaluation
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_acc = (val_correct / max(1, val_total)) * 100.0

        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_weights = model.state_dict().copy()

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs:02d} | Train Loss: {total_loss/total:.4f}, Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

    model.load_state_dict(best_weights)
    print(f"[QUALITY-GATE] Training complete. Best Val Acc: {best_val_acc:.2f}%")
    return model


def sweep_quality_gate_threshold(model: QualityGate, val_df: pd.DataFrame, device: str = 'cuda', output_csv: str = None):
    """
    Sweep tau acceptance thresholds from 0.30 to 0.95 to report Precision, Recall, F1
    for detecting adverse frames (underexposed/overexposed).
    """
    val_dataset = QualityGateDataset(val_df)
    loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    model.eval()
    model.to(device)

    all_probs = []
    all_targets = []  # 0: good, 1: bad (underexposed or overexposed)

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=-1)
            all_probs.append(probs.cpu())
            all_targets.append(labels.numpy())

    all_probs = torch.cat(all_probs, dim=0).numpy()
    all_targets = np.concatenate(all_targets, axis=0)
    is_bad_gt = (all_targets != 0)  # Ground truth adverse frames

    tau_candidates = np.arange(0.30, 0.96, 0.05)
    records = []

    for tau in tau_candidates:
        pred_cls = np.argmax(all_probs, axis=1)
        conf = np.max(all_probs, axis=1)

        # Defer frame if pred != good and conf > tau
        pred_blocked = (pred_cls != 0) & (conf > tau)

        tp = np.sum(pred_blocked & is_bad_gt)
        fp = np.sum(pred_blocked & (~is_bad_gt))
        fn = np.sum((~pred_blocked) & is_bad_gt)
        tn = np.sum((~pred_blocked) & (~is_bad_gt))

        precision = (tp / max(1, (tp + fp))) * 100.0
        recall = (tp / max(1, (tp + fn))) * 100.0
        f1 = (2 * precision * recall / max(1e-6, (precision + recall)))
        blocked_rate = (np.sum(pred_blocked) / len(all_targets)) * 100.0

        records.append({
            'tau': round(float(tau), 2),
            'precision_bad(%)': round(precision, 2),
            'recall_bad(%)': round(recall, 2),
            'f1_score(%)': round(f1, 2),
            'false_alarm_good(%)': round((fp / max(1, np.sum(~is_bad_gt))) * 100.0, 2),
            'blocked_rate(%)': round(blocked_rate, 2)
        })

    df_sweep = pd.DataFrame(records)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df_sweep.to_csv(output_csv, index=False)
        print(f"[QUALITY-GATE] Exported tau threshold sweep -> {output_csv}")

    return df_sweep


def simulate_session(quality_gate: QualityGate, test_df: pd.DataFrame, tau: float = 0.6, device: str = 'cuda', output_csv: str = None):
    """
    Simulate streaming session with / without Quality-Gate:
    Measures:
      (a) Percentage of frames forwarded to downstream detector
      (b) Number of heavy neural inference invocations
      (c) Estimated compute/energy savings (% frames shed)
    """
    dataset = QualityGateDataset(test_df)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    quality_gate.eval()
    quality_gate.to(device)

    total_frames = len(test_df)
    passed_frames = 0
    blocked_frames = 0

    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            is_passed, _, _, _ = quality_gate.predict_quality(imgs, tau=tau)
            passed_frames += is_passed.sum().item()
            blocked_frames += (~is_passed).sum().item()

    energy_saved_ratio = (blocked_frames / max(1, total_frames)) * 100.0

    simulation_results = [
        {
            'pipeline': 'Without Quality-Gate (Baseline)',
            'total_input_frames': total_frames,
            'main_model_inferences': total_frames,
            'filtered_bad_frames': 0,
            'energy_savings_proxy(%)': '0.0%'
        },
        {
            'pipeline': f'With Quality-Gate (tau={tau})',
            'total_input_frames': total_frames,
            'main_model_inferences': passed_frames,
            'filtered_bad_frames': blocked_frames,
            'energy_savings_proxy(%)': f"{energy_saved_ratio:.2f}%"
        }
    ]

    sim_df = pd.DataFrame(simulation_results)
    if output_csv:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        sim_df.to_csv(output_csv, index=False)
        print(f"[QUALITY-GATE] Exported session simulation -> {output_csv}")

    return sim_df
