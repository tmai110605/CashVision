import json
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

from sim_engine.stats_analyzer import (
    compute_system_aggregate_metrics,
    compute_condition_breakdown,
    compute_denomination_breakdown,
    generate_latex_table,
    run_wilcoxon_tests
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs_sim"
raw_p = LOGS_DIR / "video_benchmark_raw.csv"

df = pd.read_csv(raw_p)
print(f"Loaded {len(df)} raw video runs.")
print("Systems:", df["system"].value_counts().to_dict())

def safe_write_csv(df_out, path):
    for _ in range(5):
        try:
            df_out.to_csv(path, index=False)
            return
        except PermissionError:
            time.sleep(0.5)
    # fallback
    tmp_path = path.with_suffix(f".tmp_{int(time.time()*1000)}.csv")
    df_out.to_csv(tmp_path, index=False)
    try:
        import os
        os.replace(tmp_path, path)
    except Exception:
        pass

# 1. Table 1: Overall System Performance on Real Videos
t1 = compute_system_aggregate_metrics(df)
safe_write_csv(t1, LOGS_DIR / "video_table1_systems_overall.csv")
print("Saved video_table1_systems_overall.csv (N =", t1['N_Sessions'].iloc[0], ")")

# 2. Table 2: Condition Breakdown
t2 = compute_condition_breakdown(df)
safe_write_csv(t2, LOGS_DIR / "video_table2_condition_breakdown.csv")
print("Saved video_table2_condition_breakdown.csv")

# 3. Table 3: Denomination Breakdown
t3 = compute_denomination_breakdown(df)
safe_write_csv(t3, LOGS_DIR / "video_table3_denomination_breakdown.csv")
print("Saved video_table3_denomination_breakdown.csv")

# 4. Latex table
latex_str = generate_latex_table(t1, cpu_model="Laptop CPU (28W TDP, Real-time Camera Feed)", os_name=sys.platform)
for _ in range(5):
    try:
        with open(LOGS_DIR / "video_table_latex_code.tex", "w", encoding="utf-8") as f:
            f.write(latex_str)
        break
    except PermissionError:
        time.sleep(0.5)
print("Saved video_table_latex_code.tex")

# 5. Wilcoxon Test for Real Video Energy
try:
    wilc = run_wilcoxon_tests(df, metric="energy_joules", baseline="b0", candidate="cascade")
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(x) for x in obj]
        elif hasattr(obj, "item"):
            return obj.item()
        return obj

    with open(LOGS_DIR / "video_wilcoxon_energy_results.json", "w", encoding="utf-8") as f:
        json.dump(make_serializable(wilc), f, indent=2)
    print("Saved video_wilcoxon_energy_results.json")
except Exception as e:
    print("Wilcoxon error:", e)

# 6. Comprehensive Side-by-Side Comparison: Real Video vs Synthetic 60-frame Simulation
sim_t1_p = LOGS_DIR / "table1_systems_overall.csv"
if sim_t1_p.exists():
    sim_t1 = pd.read_csv(sim_t1_p)
    comp_rows = []
    for sys_code in ["b0", "b1", "cascade"]:
        r_sim = sim_t1[sim_t1["system_code"] == sys_code].iloc[0]
        r_vid = t1[t1["system_code"] == sys_code].iloc[0]
        comp_rows.append({
            "System": r_vid["System"],
            "System_Code": sys_code,
            "Sim_Denom_Acc": r_sim.get("Denom_Accuracy_pct", "N/A"),
            "Video_Denom_Acc": r_vid.get("Denom_Accuracy_pct", "N/A"),
            "Sim_Exact_Acc": r_sim.get("Exact_Accuracy_pct", "N/A"),
            "Video_Exact_Acc": r_vid.get("Exact_Accuracy_pct", "N/A"),
            "Sim_Trigger_Rate": r_sim.get("Trigger_Rate_pct", "N/A"),
            "Video_Trigger_Rate": r_vid.get("Trigger_Rate_pct", "N/A"),
            "Sim_Energy_J": f"{r_sim.get('_raw_energy_j', 0):.1f}",
            "Video_Energy_J": f"{r_vid.get('_raw_energy_j', 0):.1f}",
            "Sim_Energy_Savings": r_sim.get("Energy_Savings_pct", "0.0%"),
            "Video_Energy_Savings": r_vid.get("Energy_Savings_pct", "0.0%"),
            "Sim_TTC_s": r_sim.get("Time_to_Correct_s", "N/A"),
            "Video_TTC_s": r_vid.get("Time_to_Correct_s", "N/A"),
            "Sim_Full_Latency_ms": r_sim.get("Full_Latency_ms", "N/A"),
            "Video_Full_Latency_ms": r_vid.get("Full_Latency_ms", "N/A"),
        })
    df_comp = pd.DataFrame(comp_rows)
    safe_write_csv(df_comp, LOGS_DIR / "video_vs_simulation_comparison.csv")
    print("Saved video_vs_simulation_comparison.csv")

print("All video tables successfully generated!")
