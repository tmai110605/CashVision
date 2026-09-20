# Graphical Abstract Specification (Elsevier ESWA Compliant)

**System Name:** CashVision  
**Manuscript Title:** Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance  
**Target Journal:** Expert Systems with Applications (Elsevier)  
**Document Status:** Build Specification for Replacement Artwork (Zero Generative-AI Policy)

---

## 1. Compliance Rationale & Elsevier Policy

Elsevier's policy strictly prohibits the use of generative AI or AI-assisted image generation/alteration tools in producing artwork, figures, or graphical abstracts in submitted manuscripts. Image forensics may be applied, and authors may be required to provide authentic pre-AI raw source files. The original graphical abstract (`graphic.pdf` / `graphic.png`) features a 3D photorealistic synthetic rendering (hand, smartphone, specular light rays) that violates this policy and poses an immediate desk-rejection hazard.

This specification prescribes a replacement graphical abstract constructed **exclusively** from:
1. Authentic real-world photographic captures already present in the benchmark dataset (`train_200k_overexposed_0019.png`, `train_100k_torn_overexposed_0013.png`, `train_10k_indoor_0008.png`, etc.).
2. Vector-drawn process blocks, arrows, and state machines (SVG / TikZ / Inkscape / Adobe Illustrator).
3. Authentic screen captures of the compiled Android application (`app_cashvision`) running on the physical Samsung Galaxy A54 test device.

---

## 2. Dimensional & Format Specifications

- **Dimensions:** Exactly 531 $\times$ 1328 pixels at 300 DPI (minimum width 531 px; aspect ratio approximately $1:2.5$ or $1:3$, horizontal orientation) or vector PDF ($180 \text{ mm} \times 70 \text{ mm}$).
- **Target File Formats:** High-resolution PDF (`figures/graphical_abstract.pdf`) and 300 DPI TIFF/PNG (`figures/graphical_abstract.png`).
- **Typography:** Open-source, clean sans-serif typography (Helvetica, Arial, or DejaVu Sans).
  - Main Panel Titles: 11 pt Bold, Uppercase.
  - Sub-captions & Flow Labels: 9 pt Regular / Semibold.
  - Quantitative Highlights & Badges: 8.5 pt Bold.
- **Color Palette (Elsevier Applied-AI Palette):**
  - Primary Accent (System/Cascade): `#1A56DB` (Deep Royal Blue)
  - Photometric MQTone / Glare Recovery: `#D97706` (Amber / Warm Ochre)
  - Downstream YOLOv8n / Defect Detection: `#059669` (Emerald Green)
  - Warning / Financial Hazard Alert: `#DC2626` (Crimson Red)
  - Neutral Backgrounds: `#F8FAFC` (Off-white canvas), `#E2E8F0` (Card borders), `#1E293B` (Dark slate text).

---

## 3. Three-Panel Layout Architecture

The graphical abstract follows a clean left-to-right assistive execution workflow divided into three interconnected panels:

```
+---------------------------------------------------------------------------------------------------------+
|                                    CASHVISION GRAPHICAL ABSTRACT                                         |
+-----------------------------+------------------------------------+--------------------------------------+
| PANEL 1: REAL-WORLD INPUTS  | PANEL 2: ADAPTIVE ON-DEVICE STACK  | PANEL 3: MULTIMODAL ASSISTIVE OUTPUT |
| & ADVERSE OPTICAL STRESSORS | (LIGHTWEIGHT REASONING CASCADE)    | & EMPIRICAL HIGHLIGHTS               |
+-----------------------------+------------------------------------+--------------------------------------+
| [Real Photo: BOPP Glare]    | [Tier-1 Spatial Variance: 0.05 ms] | [Real App Screenshot: Samsung A54]   |
| 200,000 VND saturated      |                  |                 | Screen showing Bounding Boxes        |
|                             |                  v                 | Denom: "200k VND" + Defect: "Torn"   |
| [Real Photo: Substrate Tear]| [Tier-2 Quality-Gate: 1.70 ms]     |                                      |
| 100,000 VND jagged fold     | (Rejects motion blur & saturation) | [Acoustic & Haptic Indicators]       |
|                             |                  |                 | Audio: "200,000 Dong - Torn Note"    |
| [Problem Callout Box]:      |                  v                 | Haptic: [Dual Vibration Pulse]       |
| - BOPP Specular Reflection  | [MQTone: 19.6k params, 23.2 ms]    |                                      |
| - Macro-Tear Propagation    | (8x8 Local Residual Glare Damp)    | [Key Validated Metrics Badges]:      |
| - High Hazard in Single-Shot|                  |                 | - 0/36 Monetary Valuation Errors     |
|   (14.6% financial risk)    |                  v                 | - 61.3% Compute Energy Savings       |
|                             | [YOLOv8n Dual-Task Detection]      | - 20.5 FPS Camera Preview Fluidity   |
|                             | (Denomination + Tear Localization) | - SUS: 78.2 (Grade B) / TLX: -42.3%  |
|                             |                  |                 |                                      |
|                             |                  v                 |                                      |
|                             | [Temporal Consensus FSM]           |                                      |
|                             | (3-State Arbitration & Latching)   |                                      |
+-----------------------------+------------------------------------+--------------------------------------+
| Caption: Real-time, energy-efficient assistive banknote verification under non-Lambertian polymer glare.|
+---------------------------------------------------------------------------------------------------------+
```

---

## 4. Detailed Panel Contents and Asset Mapping

### Panel 1: Adverse Optical Challenges (Left Panel, 28% Width)
- **Title:** `1. ADVERSE OPTICAL INPUTS`
- **Asset 1.1:** Real photograph from dataset: `train_200k_overexposed_0019.png` (or `train_500k_torn_overexposed_0015.png`).
  - Overlay: A dashed red inset box zooming onto the blown-out specular highlight covering the denomination numeral.
  - Sub-label: `Severe Specular Glare (BOPP Polymer Substrate)`.
- **Asset 1.2:** Real photograph from dataset: `train_100k_torn_clean_0021.png`.
  - Overlay: A dashed amber inset box zooming onto the jagged tear along the fold line.
  - Sub-label: `Macroscopic Substrate Tear Defect`.
- **Callout Card:**
  - Red background `#FEF2F2`, border `#FCA5A5`, text `#991B1B`.
  - Content: `Single-Shot Hazard: 14.6% valuation errors (mistaking 200k for 10k VND). Exhaustive inference causes palm-discomfort thermal throttling (43.5°C).`

### Panel 2: On-Device Energy-Aware Cascade (Middle Panel, 42% Width)
- **Title:** `2. CASHVISION ON-DEVICE EXPERT CASCADE`
- **Component 2.1 (Quality-Gate Pre-Filter):**
  - Block A: `Tier-1 Spatial Variance Filter` (`\sigma_{\text{gray}} > 15.0`, $<0.05$\,ms CPU). Instantly drops blank/pocket frames.
  - Block B: `Tier-2 Photometric CNN` ($64 \times 64$, $<50$\,KB, $1.70$\,ms CPU). Computes optical readiness $q \ge 0.60$.
  - Flow arrow with label: `Triggers inference only upon optical stability (16.4% active triggers)`.
- **Component 2.2 (MQTone Photometric Restoration):**
  - Block C: `MQTone Sub-Network` ($19{,}686$ params, $<0.1$\,MB ONNX, $23.2$\,ms CPU).
  - Micro-diagram: Global curve parameters $(\gamma_g, \beta_g, \alpha_g)$ + Local $8 \times 8$ grid residual maps $(\Delta\gamma, \Delta\beta, \Delta\alpha)$.
  - Sub-label: `Physics-guided glare dampening (+7.73 pp overexposure gain; zero chromatic drift)`.
- **Component 2.3 (Joint Dual-Task YOLOv8n):**
  - Block D: `Joint YOLOv8n` ($3.2$\,M params, $640 \times 640$).
  - Outputs: Multi-class Denomination Bounding Box + Defect Fracture Bounding Box (\texttt{torn}).
- **Component 2.4 (Temporal Consensus FSM):**
  - Block E: `3-State Consensus FSM` ($\texttt{SEARCHING} \to \texttt{READY\_TO\_VERIFY} \to \texttt{CONFIRMED}$).
  - Rule Badges: `K_{\text{opt}}=3 buffer stability`, `Confidence-weighted voting`, `2-frame defect confirmation`.

### Panel 3: Multimodal Assistive Delivery & Validated Impact (Right Panel, 30% Width)
- **Title:** `3. REAL-WORLD ASSISTIVE IMPACT`
- **Asset 3.1 (Actual Smartphone Screenshot):**
  - Authentic capture from Samsung Galaxy A54 running `app_cashvision`: Camera preview showing green bounding box around 200,000 VND banknote and red bounding box on tear defect, with UI text display "200,000 VND | TORN".
- **Asset 3.2 (Assistive Output Icons):**
  - Speaker wave icon: `Text-to-Speech: "200,000 Dong. Caution: Torn Banknote"`
  - Vibration icon: `Dual Haptic Pulse (Tear Alert)`
- **Summary Metrics Badges (Container Cards):**
  - Card 1: `0/36 Valuation Errors` (95% CI: $[0.0\%, 9.6\%]$ in live smartphone field trials)
  - Card 2: `-61.3% Compute Energy Proxy` (Host PC) / `-36.5% Mobile Active Power` ($2.79$\,W vs.\ $4.39$\,W)
  - Card 3: `20.5 FPS Camera Preview Fluidity` ($97.9\%$ lower amortized frame latency vs.\ $B_0$)
  - Card 4: `SUS: 78.2 (Grade B)` | `NASA-TLX: -42.3% Workload` | `Hazard Rate: 1.0%` ($N=16$ blindfolded cohort)

---

## 5. Caption String (For Submission Metadata)

> **Graphical Abstract Caption:**  
> Overview of the CashVision mobile assistive expert system for polymer currency inspection. (1) Real-world polymer banknotes (BOPP) under unconstrained handheld presentation exhibit severe non-Lambertian specular glare and physical substrate tears, creating dangerous financial valuation hazards for blind users. (2) CashVision deploys an on-device energy-aware cascade running 100% offline via native ONNX Runtime C++ on a Samsung Galaxy A54 smartphone. An ultra-fast dual-tier Quality-Gate ($1.70$\,ms) and a 3-state Temporal Consensus FSM screen video frames, invoking the ultra-compact MQTone restoration network ($19{,}686$ parameters, suppressing specular blowout by $+7.73$ percentage points) and dual-task YOLOv8n detector only when optical stability is confirmed. (3) Verified classifications are delivered via speech and haptic pulses, achieving zero monetary valuation errors ($0/36$), unlocking full 20.5\,FPS preview cadence, reducing active power draw by $36.5\%$, and elevating System Usability Scale (SUS) to 78.2 with a 42.3\% cognitive workload reduction in a 16-participant blindfolded user study.

---

## 6. Build Instructions

1. Assemble vector components in Inkscape or Adobe Illustrator at 300 DPI canvas ($3540 \times 1416$ px).
2. Insert only raw photographic TIFF/PNG files from the repository (`train_200k_overexposed_0019.png`, etc.) without applying AI filters, upscalers, or synthetic generative brushes.
3. Capture a direct Android screenshot of `app_cashvision` via `adb exec-out screencap -p > figures/screenshot_a54.png` on the Samsung A54 device.
4. Export as CMYK/RGB vector PDF (`figures/graphical_abstract.pdf`) and 300 DPI flattened PNG (`figures/graphical_abstract.png`).
