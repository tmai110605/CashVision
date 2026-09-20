"""
plot_benchmark.py
=================
Generates publication-quality Vector PDF figures (with TrueType 42 fonts for camera-ready publication)
directly from benchmark CSVs:
- Figure 1: Energy (Joules) vs. Full Latency (ms) Pareto Trade-off (.pdf)
- Figure 2: Energy Consumption across 6 Environmental Conditions (.pdf)
- Figure 3: Time-to-Correct (TTC) vs. Exact Accuracy across Systems (.pdf)

Usage:
    python plot_benchmark.py
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs_sim"
FIG_DIR = LOGS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Set publication style with Type 42 TrueType fonts (academic publication compliant)
plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 11,
    "font.family": "serif",
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "grid.alpha": 0.4,
    "grid.linestyle": "--"
})

RAW_CSV = LOGS_DIR / "benchmark_raw.csv"
if not RAW_CSV.exists():
    print(f"Error: {RAW_CSV} not found.")
    exit(1)

df = pd.read_csv(RAW_CSV)
df = df[df["system"].isin(["b0", "b1", "cascade"])].reset_index(drop=True)

SYSTEM_COLORS = {
    "b0": "#d9534f",       # Crimson / Coral (Baseline B0)
    "b1": "#f0ad4e",       # Amber / Orange (Baseline B1)
    "cascade": "#2e7d32"   # Forest Green (Proposed Pipeline)
}
SYSTEM_NAMES = {
    "b0": r"$B_0$: Full Every Frame",
    "b1": r"$B_1$: Single-Shot Delay",
    "cascade": r"Proposed Cascade"
}

# -------------------------------------------------------------
# Figure 1: Energy vs. Latency Pareto Frontier
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for sys_id in ["b0", "b1", "cascade"]:
    sys_df = df[df["system"] == sys_id]
    mean_lat = sys_df["avg_full_latency_ms"].mean()
    std_lat = sys_df["avg_full_latency_ms"].std()
    mean_eng = sys_df["energy_joules"].mean()
    std_eng = sys_df["energy_joules"].std()
    
    ax.errorbar(
        mean_lat, mean_eng,
        xerr=std_lat, yerr=std_eng,
        fmt='o', color=SYSTEM_COLORS[sys_id],
        ecolor=SYSTEM_COLORS[sys_id], elinewidth=1.8,
        capsize=5, capthick=1.5, markersize=9,
        label=SYSTEM_NAMES[sys_id]
    )
    offset_y = 12 if sys_id != "b1" else -15
    ax.annotate(
        f"{SYSTEM_NAMES[sys_id]}\n({mean_lat:.1f} ms, {mean_eng:.1f} J)",
        (mean_lat, mean_eng),
        textcoords="offset points",
        xytext=(8, offset_y),
        fontsize=9,
        fontweight="bold" if sys_id == "cascade" else "normal",
        color=SYSTEM_COLORS[sys_id]
    )

ax.set_xlabel("Mean Pipeline Latency per Triggered Frame (ms)")
ax.set_ylabel("Total Session Energy (Joules, 28W TDP)")
ax.set_title("Energy-Latency Trade-off across Video Inference Paradigms")
ax.grid(True)
ax.legend(loc="upper left")
plt.tight_layout()

fig1_pdf = FIG_DIR / "fig1_energy_latency_tradeoff.pdf"
fig1_png = FIG_DIR / "fig1_energy_latency_tradeoff.png"
fig.savefig(fig1_pdf, bbox_inches="tight")
fig.savefig(fig1_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Generated Vector PDF: {fig1_pdf}")

# -------------------------------------------------------------
# Figure 2: Energy across 6 Environmental Conditions
# -------------------------------------------------------------
cond_df = df.groupby(["condition", "system"])["energy_joules"].mean().unstack()
cond_order = ["indoor", "outdoor", "backlight", "overexposed", "torn_clean", "torn_bright"]
cond_df = cond_df.reindex(cond_order)[["b0", "b1", "cascade"]]

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(cond_order))
width = 0.25

rects1 = ax.bar(x - width, cond_df["b0"], width, label=SYSTEM_NAMES["b0"], color=SYSTEM_COLORS["b0"], alpha=0.9)
rects2 = ax.bar(x, cond_df["b1"], width, label=SYSTEM_NAMES["b1"], color=SYSTEM_COLORS["b1"], alpha=0.9)
rects3 = ax.bar(x + width, cond_df["cascade"], width, label=SYSTEM_NAMES["cascade"], color=SYSTEM_COLORS["cascade"], alpha=0.9)

ax.set_xlabel("Environmental Condition")
ax.set_ylabel("Mean Energy Consumption (Joules)")
ax.set_title("Energy Consumption across Challenging Video Conditions")
ax.set_xticks(x)
ax.set_xticklabels([c.replace("_", "\n") for c in cond_order])
ax.legend()
ax.grid(axis="y")
plt.tight_layout()

fig2_pdf = FIG_DIR / "fig2_energy_by_condition.pdf"
fig2_png = FIG_DIR / "fig2_energy_by_condition.png"
fig.savefig(fig2_pdf, bbox_inches="tight")
fig.savefig(fig2_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Generated Vector PDF: {fig2_pdf}")

# -------------------------------------------------------------
# Figure 3: TTC (Time to Correct) vs. Denom Accuracy
# -------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for sys_id in ["b0", "b1", "cascade"]:
    sys_df = df[df["system"] == sys_id]
    mean_ttc = sys_df["time_to_correct_s"].mean()
    std_ttc = sys_df["time_to_correct_s"].std()
    acc = sys_df["exact_match_pct"].mean()
    
    ax.errorbar(
        mean_ttc, acc,
        xerr=std_ttc,
        fmt='s', color=SYSTEM_COLORS[sys_id],
        ecolor=SYSTEM_COLORS[sys_id], elinewidth=1.8,
        capsize=5, capthick=1.5, markersize=9,
        label=SYSTEM_NAMES[sys_id]
    )
    ax.annotate(
        f"{SYSTEM_NAMES[sys_id]}\nTTC: {mean_ttc:.2f}s | Acc: {acc:.1f}%",
        (mean_ttc, acc),
        textcoords="offset points",
        xytext=(10, -5),
        fontsize=9,
        color=SYSTEM_COLORS[sys_id]
    )

ax.set_xlabel("Mean Time to Correct Recognition (seconds)")
ax.set_ylabel("Exact Match Accuracy (%)")
ax.set_title("Recognition Speed (TTC) vs. Accuracy")
ax.set_ylim(40, 80)
ax.grid(True)
ax.legend(loc="lower left")
plt.tight_layout()

fig3_pdf = FIG_DIR / "fig3_ttc_vs_accuracy.pdf"
fig3_png = FIG_DIR / "fig3_ttc_vs_accuracy.png"
fig.savefig(fig3_pdf, bbox_inches="tight")
fig.savefig(fig3_png, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Generated Vector PDF: {fig3_pdf}")

print("ALL PUBLICATION VECTOR PDF FIGURES GENERATED SUCCESSFULLY!")


