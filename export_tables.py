import json
import sys
from pathlib import Path
import pandas as pd

from sim_engine.stats_analyzer import (
    compute_system_aggregate_metrics,
    compute_condition_breakdown,
    compute_denomination_breakdown,
    generate_latex_table,
    run_wilcoxon_tests
)

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs_sim"
raw_p = LOGS_DIR / "benchmark_raw.csv"
df_raw = pd.read_csv(raw_p)
print(f"Loaded {len(df_raw)} raw runs.")

# Rename B2 -> B1 (Single-Shot Sampled Delay)
df = df_raw.copy()
df["system"] = df["system"].replace({"b2": "b1"})
if "log_path" in df.columns:
    df["log_path"] = df["log_path"].astype(str).str.replace("_b2.jsonl", "_b1.jsonl")

# Ensure only b0, b1, cascade exist
df = df[df["system"].isin(["b0", "b1", "cascade"])].reset_index(drop=True)

n_runs = len(df)
n_sess = df["session_id"].nunique()
print(f"Systems now: {df['system'].unique().tolist()} ({n_runs} runs across {n_sess} unique sessions).")

# Overwrite raw files with updated data
df.to_csv(raw_p, index=False)
df.to_csv(LOGS_DIR / "benchmark_matrix_summary.csv", index=False)
print("Saved updated benchmark_raw.csv and benchmark_matrix_summary.csv")

# Clean & rename jsonl logs: remove old b1 (obsolete detector continuous), rename b2 to b1
old_b1_files = list(LOGS_DIR.glob("*_b1.jsonl"))
b2_files = list(LOGS_DIR.glob("*_b2.jsonl"))
if b2_files:
    for p in old_b1_files:
        try:
            p.unlink()
        except Exception:
            pass
    for p in b2_files:
        try:
            new_p = p.with_name(p.name.replace("_b2.jsonl", "_b1.jsonl"))
            txt = p.read_text(encoding="utf-8")
            txt = txt.replace('"system": "b2"', '"system": "b1"')
            new_p.write_text(txt, encoding="utf-8")
            p.unlink()
        except Exception as e:
            pass
    print(f"Synced jsonl log files: removed obsolete b1 logs and renamed {len(b2_files)} b2 files to b1.")

# Table 1: Overall
t1 = compute_system_aggregate_metrics(df)
t1.to_csv(LOGS_DIR / "table1_systems_overall.csv", index=False)
print("Saved table1_systems_overall.csv")

# Table 2: Condition
t2 = compute_condition_breakdown(df)
t2.to_csv(LOGS_DIR / "table2_condition_breakdown.csv", index=False)
print("Saved table2_condition_breakdown.csv")

# Table 3: Denomination
t3 = compute_denomination_breakdown(df)
t3.to_csv(LOGS_DIR / "table3_denomination_breakdown.csv", index=False)
print("Saved table3_denomination_breakdown.csv")

# LaTeX Code
latex_str = generate_latex_table(t1, cpu_model="Laptop CPU (28W TDP)", os_name=sys.platform)
with open(LOGS_DIR / "table_latex_code.tex", "w", encoding="utf-8") as f:
    f.write(latex_str)
print("Saved table_latex_code.tex")

# Wilcoxon Test
wilc = run_wilcoxon_tests(df, metric="energy_joules", baseline="b0", candidate="cascade")
def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(x) for x in obj]
    elif hasattr(obj, "item"):
        return obj.item()
    return obj

with open(LOGS_DIR / "wilcoxon_energy_results.json", "w", encoding="utf-8") as f:
    json.dump(make_serializable(wilc), f, indent=2)
print("Saved wilcoxon_energy_results.json")
print("ALL PUBLICATION TABLES EXPORTED SUCCESSFULLY!")
