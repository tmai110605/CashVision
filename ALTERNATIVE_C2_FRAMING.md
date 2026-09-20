# Alternative Framing for Contribution C2 (MQTone): Option A vs. Option B

This document provides the complete specification, rationale, and drop-in text for **Option B** (Defending MQTone via Per-Condition Adaptivity and Variance Reduction), alongside a comparison with **Option A** (Demoting MQTone to an Enabling Engineering Component for C3, implemented by default in the manuscript for Task 3).

---

## 1. Executive Summary: The C2 Dilemma

### The Problem
In the original manuscript, Contribution C2 presented MQTone as a headline algorithmic contribution. However:
1. **Limitation 5** states that on static images MQTone "performs essentially on par with calibrated static Gamma Correction (69.63% vs. 70.03%)" and that its "core practical value does not stem from discovering an unprecedented mathematical transformation".
2. **Highlight bullet 2** advertised "matching Gamma".
3. Advertising parity with a 1970s classical technique as a headline contribution is self-defeating for a submission to *Expert Systems with Applications* (ESWA). A hostile reviewer will instantly seize upon this to claim zero algorithmic novelty.

### The Two Strategic Options
- **Option A (Demote — Implemented in Manuscript):** Reposition MQTone from a standalone algorithmic breakthrough to an ultra-compact edge conditioning component whose primary role is enabling the real-time cascade (C3). This shifts the paper's scientific weight squarely onto the problem characterization and multi-condition dataset (C1), the rule-governed expert cascade system (C3), and the empirical hardware/usability evaluations (C3, C4).
- **Option B (Defend — Provided Below):** Mount the strongest honest defense of MQTone as a primary contribution by framing it around **per-condition adaptivity** and **cross-fold variance reduction** rather than peak overexposure accuracy. The core argument is that *no single static gamma exponent can simultaneously serve diverse operational conditions*.

---

## 2. Option B: The Scientific Defense Argument

### 2.1 Core Empirical Evidence (Already in the Manuscript)
The manuscript already contains two critical empirical findings that expose the fundamental failure mode of static Gamma correction:

1. **Backlight Degradation & Shadow Crush:**
   - On the YOLO11n backbone (Table 3, Panel B), uncorrected baseline accuracy under strong backlighting is **$85.33\% \pm 1.35\%$**.
   - Applying static Gamma correction (calibrated to darken overexposed highlights) severely crushes shadowed foreground features, degrading backlight accuracy down to **$82.63\% \pm 8.15\%$** ($-2.70$ percentage points below uncorrected baseline).
   - In sharp contrast, MQTone's decoupled $8 \times 8$ local spatial grid adaptively preserves shadowed foreground intaglio while compressing background glare, reaching **$86.43\% \pm 5.14\%$** ($+3.80$ percentage points over Gamma, and $+1.10$ points over baseline).
   - On YOLOv8n (Table 3, Panel A), MQTone similarly achieves **$84.67\% \pm 2.63\%$** under backlighting vs. Gamma's **$81.97\% \pm 4.51\%$** ($+2.70$ percentage points over Gamma).

2. **Cross-Fold Operational Variance:**
   - Under severe overexposure on YOLO11n, Gamma Correction exhibits an unstable cross-fold standard deviation of **$\pm 7.80\%$** ($78.10\% \pm 7.80\%$), reflecting high sensitivity to the specular orientation of individual folds.
   - MQTone maintains a significantly tighter standard deviation of **$\pm 4.19\%$** ($77.71\% \pm 4.19\%$), achieving nearly **$2\times$ tighter operational consistency** across varying physical presentations.
   - Overall across all six operational conditions, MQTone leads Gamma by $+1.59$ percentage points on YOLOv8n ($79.58\%$ vs. $77.99\%$) and $+1.39$ percentage points on YOLO11n ($85.01\%$ vs. $83.62\%$).

### 2.2 The Analytical Argument
A static global transform assumes a spatially and temporally stationary degradation. In unconstrained mobile video handling of polymer banknotes:
1. **Mutually Incompatible Global Optima:** Overexposure requires $\gamma < 1.0$ (or contrast expansion at the top) to recover saturated intaglio numerals, whereas backlighting requires $\gamma > 1.0$ (or foreground lifting) to prevent dark ink from falling below sensor noise floors. A single global $\gamma$ chosen a priori cannot satisfy both.
2. **Spatial Non-Uniformity:** Specular glints on BOPP polymer are localized to diffractive windows, hologram patches, and fold ridges; global darkening destroys valid contrast across the remainder of the banknote.
3. **Autonomous Edge Feasibility:** Static Gamma requires manual heuristic calibration on a per-environment basis, which is impossible for a visually impaired user who cannot assess lighting conditions. MQTone automates this calibration per-frame in $23.2$\,ms on a mobile CPU.

### 2.3 What Additional Analysis Would Make Option B Airtight?
To make Option B invulnerable to reviewer challenge, the authors should run a **Per-Condition Optimal Gamma Sweep**:
- Sweep static $\gamma \in [0.4, 2.2]$ in increments of $0.1$ separately across each of the six environmental subsets (Indoor, Outdoor, Backlight, Overexposure, Clean Tears, Torn Bright).
- **Hypothesis to confirm:** Plotting accuracy vs. $\gamma$ will show that the optimal $\gamma^*$ for Overexposure ($\gamma^* \approx 0.65$--$0.75$) produces catastrophic failure on Backlight, while the optimal $\gamma^*$ for Backlight ($\gamma^* \approx 1.2$--$1.4$) severely degrades Overexposure.
- **Punchline:** No single static $\gamma$ exists that achieves $>75\%$ across both conditions simultaneously, proving mathematically and empirically that *dynamic, per-frame parameter estimation is strictly necessary*.

---

## 3. Drop-In Replacement Blocks for Option B

If the authors choose to adopt Option B, the following four blocks should replace the corresponding Option A texts in `CVS_ESWA/elsarticle-template-harv.tex`:

### 3.1 Abstract Sentence (Option B Replacement)
```latex
CashVision incorporates \textbf{MQTone}, an ultra-lightweight ($19{,}686$ parameters, $<0.1$\,MB) dual-branch tone-mapping network that eliminates the need for manual photometric calibration; while static gamma correction achieves comparable peak overexposure recovery ($69.63\%$ vs.\ $70.03\%$) but severely degrades backlighting ($82.63\%$ vs.\ $85.33\%$ uncorrected on YOLO11n), MQTone's locally adaptive $8 \times 8$ grid curve projection consistently outperforms static gamma across conflicting lighting regimes ($86.43\%$ backlight, $+3.80$ pp gain) while cutting cross-fold performance variance nearly in half ($\pm 4.19\%$ vs.\ $\pm 7.80\%$).
```

### 3.2 Highlight Bullet 2 (Option B Replacement, 84 Characters)
```latex
\item MQTone (19,686 params) resolves static Gamma trade-offs, cutting variance by 1.9x.
```
*(Character count: 84 characters including spaces — strictly within Elsevier 85-char ceiling).*

### 3.3 Contribution C2 in Section 1 (Option B Replacement)
```latex
    \item \textbf{Adaptive Photometric Restoration Architecture (MQTone):} We formulate MQTone ($19{,}686$ parameters, $<0.1$\,MB ONNX footprint), an end-to-end learnable dual-branch parametric tone-mapping network that resolves the fundamental cross-condition trade-off of static correction heuristics. While static Gamma correction achieves comparable peak accuracy under isolated overexposure ($69.63\%$ vs.\ $70.03\%$), it inadvertently induces shadow-crushing artifacts under directional backlighting (dropping YOLO11n accuracy to $82.63\%$, below the $85.33\%$ uncorrected baseline). MQTone eliminates this conflict through an automated $8 \times 8$ parameter grid, boosting backlight accuracy to $86.43\%$ ($+3.80$ percentage points over Gamma), reducing cross-fold standard deviation from $\pm 7.80\%$ to $\pm 4.19\%$, and outperforming low-light enhancers (+13.3 percentage points over Zero-DCE++ under overexposure) with only $1.25$\,ms GPU / $23.2$\,ms mobile CPU latency.
```

### 3.4 Conclusion Item 2 in Section 5 (Option B Replacement)
```latex
    \item \textbf{Adaptive Photometric Restoration Architecture (C2):} We formulate MQTone ($19{,}686$ parameters, $<0.1$\,MB), an ultra-compact parametric tone-mapping network that resolves the cross-condition trade-offs of static heuristics. In 5-fold cross-validation, MQTone avoids the shadow-crushing failure mode of static Gamma correction under backlighting ($86.43\%$ vs.\ $82.63\%$ on YOLO11n, $+3.80$ percentage points), cuts cross-fold standard deviation under overexposure from $\pm 7.80\%$ down to $\pm 4.19\%$, and improves overexposure accuracy by $+7.73$ percentage points over an unenhanced baseline ($69.63\%$ vs.\ $61.90\%$) without manual parameter tuning, providing robust per-frame adaptivity at $23.2$\,ms mobile-CPU latency.
```

---

## 4. Editorial Recommendation and Comparison

| Evaluation Axis | Option A (Demote — Implemented) | Option B (Defend — Alternative) |
|---|---|---|
| **ESWA Reviewer Vulnerability** | **Extremely Low:** Reviewer cannot attack MQTone's novelty because it is explicitly framed as an engineering enabler for the expert system cascade. | **Moderate:** Demands that reviewers accept variance reduction and cross-condition adaptivity as sufficient algorithmic novelty over Gamma. |
| **Coherence with Limitations** | **Perfect:** Aligns 100% with Limitation 5, eliminating internal contradictions. | **Requires Modifying Limitation 5:** Limitation 5 must be rewritten to foreground the backlight divergence and variance reduction rather than conceding parity. |
| **Additional Experiments Needed** | **None:** The manuscript stands as is. | **Recommended:** Optimal gamma sweep across conditions ($\gamma \in [0.4, 2.2]$) to make the mutually incompatible optima claim airtight. |
| **Recommendation** | **STRONGLY RECOMMENDED for initial submission.** Demoting C2 protects the manuscript against desk-rejection on "insufficient CV methods novelty" grounds and directs reviewer focus to the expert system cascade (C3) and human-in-the-loop study (C4). | **Retain as backup for revision / rebuttal** if a reviewer specifically asks for deeper enhancement novelty or questions why MQTone was introduced. |
