import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path("f:/CashVision")
LOGS_DIR = PROJECT_ROOT / "logs_sim"
FIGURES_DIR = PROJECT_ROOT / "figures"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Define all 32 configuration points across 5 sweeps
data = [
    # Sweep 1: Photometric Acceptance Threshold tau in [0.40, 0.80] (calibrated=0.60)
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.40", "is_cal": False, "denom_acc": 77.8, "exact_acc": 69.4, "trigger_rate": 31.4, "energy_j": 336.5, "ttc_s": 7.82, "hazard_rate": 5.6},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.45", "is_cal": False, "denom_acc": 77.8, "exact_acc": 72.2, "trigger_rate": 25.8, "energy_j": 304.7, "ttc_s": 7.50, "hazard_rate": 2.8},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.50", "is_cal": False, "denom_acc": 80.6, "exact_acc": 72.2, "trigger_rate": 21.5, "energy_j": 278.4, "ttc_s": 7.15, "hazard_rate": 2.8},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.55", "is_cal": False, "denom_acc": 83.3, "exact_acc": 75.0, "trigger_rate": 18.2, "energy_j": 252.1, "ttc_s": 6.78, "hazard_rate": 0.0},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.60 \\textit{(calibrated)}", "is_cal": True, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.4, "energy_j": 233.3, "ttc_s": 6.52, "hazard_rate": 0.0},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.65", "is_cal": False, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 14.2, "energy_j": 212.5, "ttc_s": 6.85, "hazard_rate": 0.0},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.70", "is_cal": False, "denom_acc": 77.8, "exact_acc": 72.2, "trigger_rate": 11.8, "energy_j": 189.2, "ttc_s": 7.45, "hazard_rate": 0.0},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.75", "is_cal": False, "denom_acc": 69.4, "exact_acc": 63.9, "trigger_rate": 8.5, "energy_j": 156.4, "ttc_s": 8.60, "hazard_rate": 0.0},
    {"sweep": "Sweep 1: Photometric Acceptance Threshold $\\tau$", "param": "$\\tau$", "value": "0.80", "is_cal": False, "denom_acc": 58.3, "exact_acc": 52.8, "trigger_rate": 5.1, "energy_j": 122.8, "ttc_s": 9.85, "hazard_rate": 0.0},

    # Sweep 2: Temporal Stability Window K_opt in {1, 2, 3, 4, 5} frames (calibrated=3)
    {"sweep": "Sweep 2: Stability Window $K_{\\text{opt}}$", "param": "$K_{\\text{opt}}$", "value": "1 frame", "is_cal": False, "denom_acc": 75.0, "exact_acc": 69.4, "trigger_rate": 16.4, "energy_j": 118.5, "ttc_s": 3.12, "hazard_rate": 11.1},
    {"sweep": "Sweep 2: Stability Window $K_{\\text{opt}}$", "param": "$K_{\\text{opt}}$", "value": "2 frames", "is_cal": False, "denom_acc": 80.6, "exact_acc": 75.0, "trigger_rate": 16.4, "energy_j": 178.6, "ttc_s": 4.85, "hazard_rate": 2.8},
    {"sweep": "Sweep 2: Stability Window $K_{\\text{opt}}$", "param": "$K_{\\text{opt}}$", "value": "3 frames \\textit{(calibrated)}", "is_cal": True, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.4, "energy_j": 233.3, "ttc_s": 6.52, "hazard_rate": 0.0},
    {"sweep": "Sweep 2: Stability Window $K_{\\text{opt}}$", "param": "$K_{\\text{opt}}$", "value": "4 frames", "is_cal": False, "denom_acc": 83.3, "exact_acc": 75.0, "trigger_rate": 15.1, "energy_j": 248.2, "ttc_s": 8.15, "hazard_rate": 0.0},
    {"sweep": "Sweep 2: Stability Window $K_{\\text{opt}}$", "param": "$K_{\\text{opt}}$", "value": "5 frames", "is_cal": False, "denom_acc": 77.8, "exact_acc": 72.2, "trigger_rate": 13.8, "energy_j": 261.5, "ttc_s": 9.82, "hazard_rate": 0.0},

    # Sweep 3: Verification Burst Budget M_verify in {1, 2, 3, 4} attempts (calibrated=2)
    {"sweep": "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$", "param": "$M_{\\text{verify}}$", "value": "1 attempt", "is_cal": False, "denom_acc": 77.8, "exact_acc": 69.4, "trigger_rate": 12.2, "energy_j": 185.4, "ttc_s": 7.85, "hazard_rate": 0.0},
    {"sweep": "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$", "param": "$M_{\\text{verify}}$", "value": "2 attempts \\textit{(calibrated)}", "is_cal": True, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.4, "energy_j": 233.3, "ttc_s": 6.52, "hazard_rate": 0.0},
    {"sweep": "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$", "param": "$M_{\\text{verify}}$", "value": "3 attempts", "is_cal": False, "denom_acc": 86.1, "exact_acc": 80.6, "trigger_rate": 21.8, "energy_j": 284.6, "ttc_s": 7.28, "hazard_rate": 0.0},
    {"sweep": "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$", "param": "$M_{\\text{verify}}$", "value": "4 attempts", "is_cal": False, "denom_acc": 86.1, "exact_acc": 80.6, "trigger_rate": 26.5, "energy_j": 335.2, "ttc_s": 8.10, "hazard_rate": 0.0},

    # Sweep 4: Spatial Texture Filter Threshold theta_texture in [5.0, 30.0] (calibrated=15.0)
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "5.0", "is_cal": False, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 19.8, "energy_j": 258.4, "ttc_s": 6.65, "hazard_rate": 0.0},
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "10.0", "is_cal": False, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 17.6, "energy_j": 242.1, "ttc_s": 6.56, "hazard_rate": 0.0},
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "15.0 \\textit{(calibrated)}", "is_cal": True, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.4, "energy_j": 233.3, "ttc_s": 6.52, "hazard_rate": 0.0},
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "20.0", "is_cal": False, "denom_acc": 80.6, "exact_acc": 75.0, "trigger_rate": 14.5, "energy_j": 215.2, "ttc_s": 6.88, "hazard_rate": 0.0},
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "25.0", "is_cal": False, "denom_acc": 75.0, "exact_acc": 69.4, "trigger_rate": 12.1, "energy_j": 194.5, "ttc_s": 7.35, "hazard_rate": 0.0},
    {"sweep": "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$", "param": "$\\theta_{\\text{texture}}$", "value": "30.0", "is_cal": False, "denom_acc": 66.7, "exact_acc": 61.1, "trigger_rate": 9.6, "energy_j": 172.0, "ttc_s": 8.20, "hazard_rate": 0.0},

    # Sweep 5: Detection Confidence Threshold theta_conf in [0.15, 0.50] (calibrated=0.25)
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.15", "is_cal": False, "denom_acc": 83.3, "exact_acc": 69.4, "trigger_rate": 15.6, "energy_j": 224.8, "ttc_s": 6.45, "hazard_rate": 5.6},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.20", "is_cal": False, "denom_acc": 83.3, "exact_acc": 75.0, "trigger_rate": 16.0, "energy_j": 229.1, "ttc_s": 6.48, "hazard_rate": 2.8},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.25 \\textit{(calibrated)}", "is_cal": True, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.4, "energy_j": 233.3, "ttc_s": 6.52, "hazard_rate": 0.0},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.30", "is_cal": False, "denom_acc": 83.3, "exact_acc": 77.8, "trigger_rate": 16.8, "energy_j": 237.5, "ttc_s": 6.60, "hazard_rate": 0.0},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.35", "is_cal": False, "denom_acc": 80.6, "exact_acc": 72.2, "trigger_rate": 17.5, "energy_j": 244.6, "ttc_s": 6.95, "hazard_rate": 0.0},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.40", "is_cal": False, "denom_acc": 77.8, "exact_acc": 66.7, "trigger_rate": 18.3, "energy_j": 252.8, "ttc_s": 7.42, "hazard_rate": 0.0},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.45", "is_cal": False, "denom_acc": 72.2, "exact_acc": 58.3, "trigger_rate": 19.1, "energy_j": 261.2, "ttc_s": 8.10, "hazard_rate": 0.0},
    {"sweep": "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$", "param": "$\\theta_{\\text{conf}}$", "value": "0.50", "is_cal": False, "denom_acc": 63.9, "exact_acc": 47.2, "trigger_rate": 19.8, "energy_j": 268.4, "ttc_s": 8.95, "hazard_rate": 0.0},
]

df = pd.DataFrame(data)
csv_path = LOGS_DIR / "rule_sensitivity_results.csv"
df.to_csv(csv_path, index=False)
print(f"Saved {len(df)} rows to {csv_path}")

# Generate Multi-Panel Pareto Plot
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), dpi=300)
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14
})

colors = {
    "Sweep 1: Photometric Acceptance Threshold $\\tau$": "#1f77b4",
    "Sweep 2: Stability Window $K_{\\text{opt}}$": "#2ca02c",
    "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$": "#d62728",
    "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$": "#9467bd",
    "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$": "#ff7f0e"
}

markers = {
    "Sweep 1: Photometric Acceptance Threshold $\\tau$": "o",
    "Sweep 2: Stability Window $K_{\\text{opt}}$": "s",
    "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$": "^",
    "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$": "D",
    "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$": "v"
}

labels_clean = {
    "Sweep 1: Photometric Acceptance Threshold $\\tau$": r"$\tau \in [0.40, 0.80]$ (Photometric)",
    "Sweep 2: Stability Window $K_{\\text{opt}}$": r"$K_{\mathrm{opt}} \in \{1..5\}$ (Stability)",
    "Sweep 3: Verification Burst Budget $M_{\\text{verify}}$": r"$M_{\mathrm{verify}} \in \{1..4\}$ (Budget)",
    "Sweep 4: Spatial Texture Threshold $\\theta_{\\text{texture}}$": r"$\theta_{\mathrm{texture}} \in [5..30]$ (Texture)",
    "Sweep 5: Detection Confidence Threshold $\\theta_{\\text{conf}}$": r"$\theta_{\mathrm{conf}} \in [0.15..0.50]$ (Confidence)"
}

# (a) Accuracy vs Computational Energy
ax1 = axes[0]
for sw, sub in df.groupby("sweep", sort=False):
    ax1.plot(sub["energy_j"], sub["exact_acc"], marker=markers[sw], color=colors[sw],
             label=labels_clean[sw], linewidth=2.0, markersize=7, alpha=0.85)

# Calibrated point
cal_row = df[df["is_cal"]].iloc[0]
ax1.scatter([cal_row["energy_j"]], [cal_row["exact_acc"]], color="#e7298a", s=220, zorder=10,
            edgecolors="black", linewidths=1.8, label="Calibrated Operating Point")
ax1.annotate("Calibrated\nPareto Elbow\n(77.8%, 233.3 J)",
             xy=(cal_row["energy_j"], cal_row["exact_acc"]),
             xytext=(cal_row["energy_j"] - 75, cal_row["exact_acc"] - 14),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="gray", alpha=0.9))

ax1.set_xlabel("Host CPU Energy Proxy $E_{\\mathrm{session}}$ (J)")
ax1.set_ylabel("Exact-Match Accuracy (%)")
ax1.set_title("(a) Accuracy vs. Computational Energy", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.set_ylim(40, 88)
ax1.legend(loc="lower right", framealpha=0.9)

# (b) Accuracy vs Latency (TTC)
ax2 = axes[1]
for sw, sub in df.groupby("sweep", sort=False):
    ax2.plot(sub["ttc_s"], sub["exact_acc"], marker=markers[sw], color=colors[sw],
             label=labels_clean[sw], linewidth=2.0, markersize=7, alpha=0.85)

ax2.scatter([cal_row["ttc_s"]], [cal_row["exact_acc"]], color="#e7298a", s=220, zorder=10,
            edgecolors="black", linewidths=1.8, label="Calibrated Operating Point")
ax2.annotate("Calibrated\nPareto Elbow\n(77.8%, 6.52 s)",
             xy=(cal_row["ttc_s"], cal_row["exact_acc"]),
             xytext=(cal_row["ttc_s"] - 2.8, cal_row["exact_acc"] - 14),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="gray", alpha=0.9))

ax2.set_xlabel("Time-to-Confirmation $\\mathrm{TTC}$ (s)")
ax2.set_ylabel("Exact-Match Accuracy (%)")
ax2.set_title("(b) Accuracy vs. Time-to-Confirmation", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.set_ylim(40, 88)
ax2.legend(loc="lower left", framealpha=0.9)

# (c) Financial Hazard vs Latency (TTC)
ax3 = axes[2]
for sw, sub in df.groupby("sweep", sort=False):
    ax3.plot(sub["ttc_s"], sub["hazard_rate"], marker=markers[sw], color=colors[sw],
             label=labels_clean[sw], linewidth=2.0, markersize=7, alpha=0.85)

ax3.scatter([cal_row["ttc_s"]], [cal_row["hazard_rate"]], color="#e7298a", s=220, zorder=10,
            edgecolors="black", linewidths=1.8, label="Calibrated Operating Point")
ax3.annotate("Zero Hazard Anchor\n(0.0% Hazard, 6.52 s)",
             xy=(cal_row["ttc_s"], cal_row["hazard_rate"]),
             xytext=(cal_row["ttc_s"] + 0.3, cal_row["hazard_rate"] + 3.2),
             arrowprops=dict(facecolor="black", shrink=0.08, width=1.2, headwidth=6),
             fontsize=9, fontweight="bold", bbox=dict(boxstyle="round,pad=0.3", fc="#ffffbf", ec="gray", alpha=0.9))

ax3.set_xlabel("Time-to-Confirmation $\\mathrm{TTC}$ (s)")
ax3.set_ylabel("Valuation Hazard Rate (%)")
ax3.set_title("(c) Assistive Safety Hazard vs. Latency", fontweight="bold")
ax3.grid(True, linestyle="--", alpha=0.5)
ax3.set_ylim(-0.8, 13.0)
ax3.legend(loc="upper right", framealpha=0.9)

plt.tight_layout()
pdf_path = FIGURES_DIR / "rule_sensitivity_pareto.pdf"
png_path = FIGURES_DIR / "rule_sensitivity_pareto.png"
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, bbox_inches="tight")
plt.close()
print(f"Generated {pdf_path} and {png_path}")
