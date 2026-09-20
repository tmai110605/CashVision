# Protocol: On-Device Smartphone Benchmark under Adverse Illumination Conditions

**Document Identifier:** `protocols/adverse_lighting_ondevice_protocol.md`  
**Associated Manuscript:** *Seeing Through the Glare: A Real-Time, Energy-Efficient Mobile Banknote Inspector for Visually Impaired Assistance* (System: CashVision)  
**Target Venue:** *Expert Systems with Applications* (ESWA), Elsevier  
**Scope:** Addressing Reviewer Defect 5.4 (Bridging the Offline vs. On-Device Adverse Illumination Gap)

---

## 1. Problem Statement: The Adverse Lighting On-Device Gap

In the current manuscript, the empirical evaluation is bifurcated:
1. **Adverse Optical Conditions (Offline):** The multi-condition static benchmark (1,812 images, Section 3.2 & 3.3) and the 36-stream continuous video benchmark (9,060 frames, Section 4.1) systematically evaluate severe specular glare, directional backlighting, outdoor shadows, and torn bright notes. However, this video benchmark was evaluated **offline on a personal computer** (Intel Core i7-1265U CPU) using a software CPU-load proxy model.
2. **Live Smartphone Field Trials (On-Device):** The physical smartphone trials on the Samsung Galaxy A54 (Section 4.2, 108 runs) and the blindfolded usability study (Section 4.3, 288 trials) were conducted **exclusively under everyday ambient indoor lighting**.

Consequently, the claim that CashVision achieves $83.3\%$ live on-device accuracy with zero valuation errors and $36.5\%$ active power reduction is currently substantiated on hardware **only under benign ambient lighting**. The performance, latency, and thermal behavior of the on-device system under physical adverse optical conditions (live specular glare, live window backlighting, live direct sunlight) remain an unmeasured gap.

To resolve this defect, this protocol specifies the minimum experimental campaign required to measure on-device performance and telemetry under live adverse operational lighting.

---

## 2. Target Adverse Environmental Conditions for Live On-Device Testing

The physical smartphone test campaign must evaluate three distinct adverse lighting regimes in real-world retail and domestic environments:

### Condition 1: Live Severe Specular Glare (Polymer Overexposure)
- **Illumination Setup:** Banknote illuminated by direct directional artificial light sources (e.g., concentrated LED desk lamp or high-intensity mobile flash luminaire at $30^\circ$--$60^\circ$ incidence, producing $>800$\,lux on the note surface).
- **Physical Optomechanics:** Induces localized camera sensor saturation ($I_{\text{gray}} > 220$) across reflective polymer substrate windows, intaglio ink portraits, and metallic security strips.

### Condition 2: Live Directional Window Backlighting
- **Illumination Setup:** Banknote held directly in front of a bright ambient background (e.g., an unshaded south-facing window during daylight hours or high-luminance backlit display panel, background illuminance $>1{,}500$\,lux, foreground banknote illuminance $<150$\,lux).
- **Physical Optomechanics:** Creates severe contrast inversion, shadow-crushing of foreground denominations, and wide dynamic range exceeding standard mobile camera ISP sensor capabilities.

### Condition 3: Live Direct Outdoor Sunlight with Cast Shadows
- **Illumination Setup:** Unconstrained outdoor handheld presentation under direct midday sunlight ($>15{,}000$\,lux) with moving partial hand/finger shadows cast across the note face.
- **Physical Optomechanics:** Induces harsh directional gradients, non-uniform chromatic shifts, and rapid auto-exposure / white-balance hunting in the Android CameraX pipeline.

---

## 3. Experimental Matrix & Execution Parameters

To ensure statistical equivalence with the ambient indoor field trials reported in Table 8 (`tab:smartphone_telemetry`):

### 3.1 Session Allocation
- **Denominations:** All 6 circulating Vietnamese polymer denominations ($10\text{k}, 20\text{k}, 50\text{k}, 100\text{k}, 200\text{k}, 500\text{k}$ VND).
- **Substrate States:** 3 pristine notes and 3 physically worn / creased notes per denomination.
- **Sessions per Condition:** $6\text{ denominations} \times 6\text{ sessions} = \mathbf{36\text{ physical sessions}}$ per condition.
- **Paradigms Evaluated:** All 3 core paradigms:
  1. $B_0$ (Uniform-rate full MQTone + YOLOv8n inference on every frame)
  2. $B_1$ (Timer-triggered single-shot capture at $t = 1.0$\,s)
  3. **Cascade** (Proposed adaptive sensory-gated pipeline)
- **Total Physical Runs:** $3\text{ conditions} \times 36\text{ sessions} \times 3\text{ paradigms} = \mathbf{324\text{ complete physical runs}}$ (spanning $\approx 59{,}000$ camera frames).

### 3.2 Standardized Measurement Window
- Each session executed for a standardized duration of $30.0$\,s with continuous hardware logging.
- A mandatory 5-minute cooldown period between runs to verify device chassis returns to baseline ($25.0^\circ\text{C}$).

---

## 4. Hardware Telemetry & Empirical Metrics to Log

The Android application (`app_cashvision`) must log synchronized sensor and hardware counters at $\ge 10$\,Hz:

| Telemetry Dimension | Source API / Hardware Counter | Logged Metric | Analytical Objective |
|---|---|---|---|
| **Power & Battery** | Android `BatteryManager` (`BATTERY_PROPERTY_CURRENT_NOW`, `BATTERY_PROPERTY_CAPACITY`) | Instantaneous Power $P(t)$ (W), Cumulative Session Energy $\int P dt$ (J) | Verify whether the $36.5\%$ power savings holds under dynamic adverse triggering. |
| **Throughput & Latency** | High-resolution timestamping (`System.nanoTime()`) | Frame processing latency $T_{\text{frame}}$ (ms), CameraX preview framerate (FPS), Time-to-Confirmation $\text{TTC}$ (s) | Quantify whether adverse glare induces gating stalls or preview cadence drops. |
| **Thermal Behavior** | Linux kernel `/sys/class/thermal/` and Android `PowerManager` thermal status | CPU core temperatures ($^\circ\text{C}$), chassis exterior temperature ($^\circ\text{C}$ via infrared radiometer), thermal throttling events | Confirm passive thermal sustainability under outdoor ambient temperatures. |
| **Gating Dynamics** | CashVision FSM engine logs | Pre-inference gate pass rate ($P_{\text{active}}$, \%), stability counter dwell time ($K_{\text{opt}}$), FSM state transition traces | Determine whether adverse lighting increases verification attempts or timeout deferrals. |
| **Assistive Reliability** | Experimenter ground-truth logging | Verification Accuracy (\%), Safe Deferral Rate (\%), Monetary Valuation Hazard Rate (\%) | Confirm whether the zero-valuation-error guarantee ($0/36$) holds under direct physical glare and backlighting. |

---

## 5. Required Manuscript Scoping & Action

Until these 324 on-device adverse runs are physically executed:
1. **All on-device claims** in the Abstract, Section 4.2, Section 4.4, and Section 5 (Conclusion) must explicitly be scoped as: *"under everyday ambient indoor lighting"*.
2. **The adverse-condition claims** must explicitly be framed as: *"established offline on a multi-condition benchmark of 1,812 static captures and 36 continuous video streams"*.
3. The open experimental gap must be explicitly documented in Limitations (Section 5) and referenced to this protocol.
