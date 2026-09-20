# Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance

<iframe src="graphic.pdf" width="100%" height="600px"></iframe>

A real-time, illumination-robust, and energy-efficient mobile computer vision system for joint denomination recognition and tear defect localization of Vietnamese polymer banknotes on handheld devices to assist visually impaired individuals.

---

## 📌 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Hardware Benchmark Setup](#2-hardware-benchmark-setup)
3. [Environment Setup](#3-environment-setup)
4. [Running Experiment C1 (Problem Characterization)](#4-running-experiment-c1-problem-characterization)
5. [Running Experiment C2 (MQTone & Enhancement Comparison)](#5-running-experiment-c2-mqtone--enhancement-comparison)
6. [Running Experiment Video (Continuous Streaming Benchmark)](#6-running-experiment-video-continuous-streaming-benchmark)
7. [Exporting ONNX Models & Android Application](#7-exporting-onnx-models--android-application)

---

## 1. Project Overview

Polymer banknotes (BOPP substrate) feature smooth, non-porous surfaces and transparent diffractive optical windows that reflect intense specular flash glare, blinding camera sensors and obliterating denomination numerals and tactile security features. In everyday circulation, banknotes also suffer mechanical tear fractures along fold lines.

**CashVision** resolves these challenges through an integrated, energy-aware mobile architecture:
***MQTone (Micro-scale Quality Tone-mapping Network):** An ultra-lightweight dual-branch tone-mapping neural module ($19{,}686$ parameters, $<0.1$\,MB footprint) that selectively suppresses localized specular glare and recovers obscured contrast in $1.25$\,ms (GPU) / $23.2$\,ms (mobile CPU).
* **Quality-Gate:** A 2-tier optical monitoring module (Tier-1 Spatial Texture Filter $\sigma_{\text{gray}} > 15.0$ in $<0.05$\,ms + Tier-2 photometric CNN classifier in $1.70$\,ms) that filters incoming video frames before triggering heavy inference.
* **Dual-Task Detector (YOLOv8n):** Simultaneously predicts 6 banknote denominations ($10\text{k}, 20\text{k}, 50\text{k}, 100\text{k}, 200\text{k}, 500\text{k}$ VND) and localizes physical tear defect bounding boxes.
* **Temporal Consensus FSM:** A 3-state finite state machine that accumulates candidate detections across consecutive frames ($K_{\text{con}} = 3$), eliminating single-frame transient errors and saving battery power.

```
Camera Stream (30 FPS)
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier-1 Spatial Texture Filter (σ_gray > 15.0)  [< 0.05 ms]   │──► Blank / Hand Occluded (Drop)
└─────────────────────────────────────────────────────────────┘
         │ (Pass)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Tier-2 Quality-Gate Light Classifier           [1.70 ms]    │──► Poor Lighting / Motion Blur (Defer)
└─────────────────────────────────────────────────────────────┘
         │ (Optical readiness q ≥ 0.60)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Temporal Consensus FSM                                      │
│ (SEARCHING ──► READY_TO_VERIFY ──► CONFIRMED)               │
└─────────────────────────────────────────────────────────────┘
         │ (Trigger conditionally)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ MQTone Photometric Tone-Mapping Network        [23.2 ms CPU]│
│ Local Glare Suppression & Contrast Restoration              │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Dual-Task YOLOv8n Detector                     [~200 ms CPU]│
│ Denomination Verification + Tear Localization               │
└─────────────────────────────────────────────────────────────┘
         │ (Reach consensus K_con = 3 consecutive frames)
         ▼
┌─────────────────────────────────────────────────────────────┐
│ Audio Speech (TTS) + Haptic Tactile Feedback (Vibration)    │
│ Suppress deep inference to preserve battery autonomy        │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Hardware Benchmark Setup

All experiments in this project were conducted and benchmarked on the following hardware environments:
* **Mobile Handheld Device (Smartphone):** **Samsung Galaxy A54** (Octa-core Exynos 1380 SoC: 4$\times$ Cortex-A78 @ 2.4\,GHz + 4$\times$ Cortex-A55 @ 2.0\,GHz, 8\,GB RAM, Android 14). Used for dataset capture, 36 continuous video recordings, on-device Android execution, and live telemetry profiling (FPS, battery power, current, and temperature).
* **Video Benchmark Computer (Host PC):** **Intel Core i7-1265U CPU @ 1.80\,GHz** (10 cores, 12 threads, 16\,GB RAM, Windows 11). Used to execute the continuous 36-video streaming benchmark and measure simulated latency/energy metrics.
* **Training & Static Evaluation GPU:** **NVIDIA Tesla T4** (16\,GB VRAM, CUDA 12.2). Used for deterministic 5-fold model training and GPU inference latency benchmarking.

---

## 3. Environment Setup

### System Requirements
* Python 3.10 or higher
* CUDA 12.1 / 12.2 (for GPU-accelerated model training)

### Dependencies Installation
```bash
# Create and activate conda environment
conda create -n cashvision python=3.10 -y
conda activate cashvision

# Install PyTorch with CUDA 12.1 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install required packages
pip install ultralytics==8.3.0 opencv-python onnx onnxruntime-gpu pandas numpy scipy matplotlib seaborn tqdm
```

---

## 4. Running Experiment C1 (Problem Characterization)

### Purpose
Quantify the baseline performance degradation of standard detectors (YOLOv8n, YOLO11n) trained under nominal indoor conditions (`indoor`, `torn_clean`) when evaluated across adverse real-world lighting conditions (`outdoor`, `backlight`, `overexposed`, `torn_bright`).

### Execution Commands
```bash
# Run 5-fold cross-validation on YOLOv8n and YOLO11n
python run_e1.py --models yolov8n yolo11n --n_splits 5 --epochs 100 --batch 16 --device 0

# Run on CPU
python run_e1.py --models yolov8n --n_splits 5 --epochs 100 --batch 16 --device cpu
```

Summary metrics (denomination recognition accuracy, tear mAP@50, and miss rates) are automatically recorded in `resultsc1/`.

---

## 5. Running Experiment C2 (MQTone & Enhancement Comparison)

### Purpose
Train and evaluate **MQTone** against 9 enhancement/restoration methods (Uncorrected Baseline, Gamma Correction, CLAHE, RetinexNet, EnlightenGAN, Afifi et al. Laplacian, IAT, Zero-DCE, Zero-DCE++) under a rigorous 5-fold cross-validation protocol.

### Execution Commands

#### 1. Evaluate all methods across 5 folds:
```bash
# Train and evaluate all 10 enhancement methods across 5 folds
python run_c2.py --model yolov8n.pt --methods all --epochs 100 --device auto
```

#### 2. Evaluate specific methods individually:
```bash
# Run MQTone only
python run_c2.py --model yolov8n.pt --methods mqtone --epochs 100 --device auto

# Run Zero-DCE++ only
python run_c2.py --model yolov8n.pt --methods zerodce_pp --epochs 100 --device auto

# Run RetinexNet only
python run_c2.py --model yolov8n.pt --methods retinexnet --epochs 100 --device auto
```

#### 3. Export C2 summary tables:
```bash
# Export formatted summary CSV and LaTeX comparison table
python export_tables.py
```
Detailed fold-by-fold results are saved under `resultv8n/` and `resultv11n/`.

---

## 6. Running Experiment Video (Continuous Streaming Benchmark)

### Purpose
Benchmark **36 continuous handheld video streams** ($9{,}060$ frames @ 30 FPS, 7.0–9.5 seconds per session) located in `video_test/` covering all 6 denominations $\times$ 6 real-world conditions, comparing three operational paradigms:
* **$B_0$ (Uniform-Rate):** Evaluates the full pipeline on 100% of incoming frames (30 FPS).
* **$B_1$ (Single-Shot Blind Delay):** Waits a 1.0-second timer delay before capturing and evaluating a single static frame.
* **Cascade:** Quality-Gate monitors frames, selectively invokes full inference only upon confirmed optical stability, and aggregates predictions via the FSM.

### Execution Commands

#### 1. Run all 36 video streams (108 independent runs):
```bash
python run_video_benchmark.py --fps 30.0 --cooldown 0.5
```

#### 2. Quick test run (2 video streams):
```bash
python run_video_benchmark.py --limit 2
```

#### 3. Run component ablation analysis:
```bash
# Evaluate 6 ablated cascade pipeline variants
python recompute_video_benchmark.py
```

#### 4. Export video metrics and plots:
```bash
# Export summary metrics (FPS, Energy Joules, Latency ms, Time-to-Confirmation s)
python export_video_tables.py

# Generate Pareto curve between energy consumption and latency
python plot_benchmark.py
```
Frame-by-frame JSONL telemetry logs are saved in `logs_sim/`.

---

## 7. Exporting ONNX Models & Android Application

### 1. Export PyTorch Models to Mobile ONNX
Convert trained PyTorch weights to optimized mobile ONNX format:
```bash
python export_mobile_onnx.py
```
Optimized ONNX files are generated in `models_mobile/`:
* `quality_gate.onnx` ($46$\,KB): Input shape `[1, 3, 64, 64]`.
* `mqtone.onnx` ($125$\,KB): Input shape `[1, 3, 64, 64]`.
* `yolov8n_cashvision.onnx` ($12.8$\,MB): Input shape `[1, 3, 640, 640]`.

### 2. Android Application (`app_cashvision`)
The `app_cashvision/` directory contains the complete Android Studio project:
* Runs **100% offline** on mobile CPU via **native ONNX Runtime C++** (`setInterOpNumThreads(2)`).
* Integrates Android CameraX for real-time video frame acquisition.
* Automatic Text-to-Speech (TTS) denomination announcement and haptic vibration upon tear detection.
* Integrated `BatteryMeter.kt` for sampling real-time current, voltage, and hardware power draw.

**Build and Deployment Steps:**
1. Open the `app_cashvision` folder in **Android Studio**.
2. Connect a **Samsung Galaxy A54** smartphone via USB with **USB Debugging** enabled.
3. Ensure the three ONNX model files are located in `app/src/main/assets/`.
4. Click **Run** (`Shift + F10`) or run from terminal:
   ```bash
   cd app_cashvision
   ./gradlew assembleDebug
   ```
5. Parse and audit live telemetry logs collected from the device:
   ```bash
   python audit_phone_logs.py
   python parse_phone_logs.py
   ```
