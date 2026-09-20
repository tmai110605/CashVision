# Protocol: Quantitative Benchmarking against State-of-the-Art Assistive Banknote Systems

**Document Identifier:** `protocols/sota_comparison_protocol.md`  
**Associated Manuscript:** *Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance* (System: CashVision)  
**Target Venue:** *Expert Systems with Applications* (ESWA), Elsevier  
**Scope:** Addressing Reviewer Defect 5.3 (Missing Task-Level Comparison with Published Currency Systems)

---

## 1. Context & Reviewer Requirement

In the current manuscript, prior assistive currency recognition works are cited in Section 1 and Section 2:
- **\citet{AlZuBi2023}:** Intelligent real-time banknote recognition system for visually impaired people using deep convolutional neural networks (*Multimedia Tools and Applications*).
- **\citet{Dhar2024}:** Deep learning based currency recognition system for visually impaired persons: A comprehensive review (*Expert Systems with Applications*).
- **\citet{Ghanem2025}:** Recognizing Egyptian currency for people with visual impairment using deep learning models (*Scientific Reports*).

However, while Table 3 benchmarks generic image enhancement modules (Gamma, CLAHE, Zero-DCE, IAT, EnlightenGAN, Afifi et al.), **no quantitative comparison against prior published assistive banknote recognition systems is presented**.

In *Expert Systems with Applications*, reviewers require comparison against the state of the art for the *specific problem domain* (assistive banknote verification), not merely against auxiliary image preprocessors. Without this, reviewers will argue that the authors have not proven CashVision surpasses established currency recognition architectures.

---

## 2. Selected Prior SOTA Banknote Systems for Direct Comparison

To establish a comprehensive and rigorous task-level comparison, three representative assistive banknote recognition pipelines must be benchmarked:

### 2.1 Al-Zu'bi et al. (2023) — Lightweight Custom DCNN
- **Architecture:** Customized compact deep convolutional neural network optimized for real-time banknote classification.
- **Input / Target:** Static RGB banknote captures; multi-class classification head.
- **Original Domain:** Jordanian Dinar banknotes under clean indoor lighting.
- **Edge Characteristics:** Designed for mobile execution, but lacks specular glare handling, defect detection capability, and temporal sensory gating.

### 2.2 Dhar & Uddin (2024) — Standard Mobile Backbone Baseline (MobileNetV3 / ResNet-50)
- **Architecture:** Canonical transfer learning baseline identified in Dhar & Uddin's comprehensive review: MobileNetV3-Small / Large and ResNet-50 backbones for assistive currency classification.
- **Input / Target:** Static RGB captures, whole-image denomination classification.
- **Edge Characteristics:** Highly efficient feature extractors, but lack bounding-box localization, defect detection, and optical quality gating.

### 2.3 Ghanem et al. (2025) — Modern YOLO-based Assistive Recognition
- **Architecture:** Contemporary YOLO detector (YOLOv8 / YOLOv9 / YOLOv10) adapted for real-time assistive currency identification.
- **Input / Target:** Bounding-box detection of banknote denominations.
- **Original Domain:** Egyptian paper currency under controlled and handheld static setups.
- **Edge Characteristics:** Fast single-frame bounding-box inference, but lacks polymer specular glare adaptation, structural tear inspection, and energy-aware cascade execution.

---

## 3. Minimum Viable Benchmark Specification

To guarantee a fair, repeatable, and falsifiable comparison, all systems must be evaluated under harmonized conditions:

### 3.1 Evaluation Platform & Hardware
- **Target Device:** Samsung Galaxy A54 smartphone (Exynos 1380 SoC, 8 GB RAM, Android 14) via native ONNX Runtime C++ (2 threads) on mobile CPU.
- **Host PC Reference:** Intel Core i7-1265U CPU @ 1.80 GHz, 10 cores, 16 GB RAM.

### 3.2 Evaluation Datasets
1. **CashVision Multi-Condition Polymer Benchmark (1,812 static images):** Evaluates cross-condition robustness across normal indoor, outdoor sunlight, strong backlighting, specular overexposure, and torn banknotes.
2. **CashVision Continuous Video Benchmark (36 streams, 9,060 frames):** Evaluates dynamic streaming accuracy, Time-to-Confirmation (TTC), and financial valuation hazards.

### 3.3 Comparative Performance Dimensions

The evaluation must report eight standardized metrics:
1. **Denomination Identification Accuracy ($\text{Acc}_{\text{denom}}$, %):** Evaluated under both benign indoor lighting and adverse optical stress (specular overexposure and backlighting).
2. **Defect Inspection Support:** Binary capability to detect and localize physical structural tears (Yes / No; Tear mAP@50).
3. **Model Footprint:** Parameter count ($\text{M}$) and ONNX file size ($\text{MB}$).
4. **On-Device Frame Latency:** Empirical execution latency ($T_{\text{infer}}$, $\text{ms}$) on the Samsung Galaxy A54 mobile CPU.
5. **Streaming Throughput:** Achieved video preview framerate ($\text{FPS}$).
6. **Energy / Power Profile:** Active SoC power draw ($\text{W}$) and session energy ($\text{J}$) on mobile hardware.
7. **Adverse Optical Handling:** Algorithmic mechanism for handling non-Lambertian specular glare and localized blowout.
8. **Temporal Arbitration Layer:** Use of rule-governed sensory gating / FSM consensus to eliminate acoustic fluttering (Yes / No).

---

## 4. Manuscript Integration

The comparative framework and empirical skeleton must be integrated into Section 3 of the manuscript as `Table~\ref{tab:sota_banknote_comparison}` with empirical accuracy cells marked `---` and `\AUTHORACTION{Execute SOTA banknote benchmark protocol}`.
