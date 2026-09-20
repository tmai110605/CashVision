from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon


SYSTEM_DISPLAY_NAMES = {
    "b0": "B0 - Full Every Frame (MQTone + YOLO)",
    "b1": "B1 - Single-Shot (Sampled Delay)",
    "cascade": "Cascade (Proposed Adaptive Pipeline)"
}


def compute_session_summary(df_logs: pd.DataFrame) -> pd.DataFrame:
    """Aggregates raw per-frame jsonl logs if loaded."""
    if df_logs.empty:
        return pd.DataFrame()
    return df_logs


def compute_system_aggregate_metrics(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Computes paper Table 15 aggregate metrics (Mean ± Std) across all sessions for each system.
    """
    if df_summary.empty or "system" not in df_summary.columns:
        return pd.DataFrame()

    systems_order = ["b0", "b1", "cascade"]
    rows = []

    # Get baseline B0 mean energy for savings calculation
    b0_df = df_summary[df_summary["system"] == "b0"]
    b0_mean_energy = b0_df["energy_joules"].mean() if not b0_df.empty else 1.0

    for sys_name in systems_order:
        sub = df_summary[df_summary["system"] == sys_name]
        if sub.empty:
            continue

        n_sessions = len(sub)
        avg_full_lat = sub["avg_full_latency_ms"].mean()
        std_full_lat = sub["avg_full_latency_ms"].std()

        peak_ram = sub["peak_ram_mb"].max()

        mean_energy = sub["energy_joules"].mean()
        std_energy = sub["energy_joules"].std()

        energy_savings_pct = max(0.0, ((b0_mean_energy - mean_energy) / b0_mean_energy * 100.0)) if b0_mean_energy > 0 else 0.0

        mean_dropped = sub["dropped_rate_pct"].mean()
        std_dropped = sub["dropped_rate_pct"].std()

        denom_acc = sub["denom_accuracy_pct"].mean()
        exact_acc = sub["exact_match_pct"].mean()

        # Detection metrics (mAP@0.5 and IoU)
        banknote_map = sub["banknote_map50"].mean() if "banknote_map50" in sub.columns else np.nan
        banknote_iou = sub["banknote_iou"].mean() if "banknote_iou" in sub.columns else np.nan

        # Tear mAP50 on sessions with tears
        tear_sub = sub["tear_map50"].dropna() if "tear_map50" in sub.columns else pd.Series(dtype=float)
        mean_tear_map = tear_sub.mean() if not tear_sub.empty else np.nan

        time_to_corr = sub["time_to_correct_s"].dropna()
        mean_ttc = time_to_corr.mean() if not time_to_corr.empty else np.nan
        std_ttc = time_to_corr.std() if not time_to_corr.empty else np.nan

        trigger_rate = sub["called_full_rate_pct"].mean()

        rows.append({
            "system_code": sys_name,
            "System": SYSTEM_DISPLAY_NAMES.get(sys_name, sys_name),
            "N_Sessions": n_sessions,
            "Full_Latency_ms": f"{avg_full_lat:.2f} ± {std_full_lat:.2f}" if pd.notna(std_full_lat) else f"{avg_full_lat:.2f}",
            "Peak_RAM_MB": f"{peak_ram:.1f}",
            "Energy_J": f"{mean_energy:.2f} ± {std_energy:.2f}" if pd.notna(std_energy) else f"{mean_energy:.2f}",
            "Energy_Savings_pct": f"{energy_savings_pct:.1f}%",
            "Dropped_Frames_pct": f"{mean_dropped:.2f}% ± {std_dropped:.2f}" if pd.notna(std_dropped) else f"{mean_dropped:.2f}%",
            "Denom_Accuracy_pct": f"{denom_acc:.1f}%",
            "Exact_Accuracy_pct": f"{exact_acc:.1f}%",
            "Banknote_mAP50_pct": f"{banknote_map:.1f}%" if pd.notna(banknote_map) else "N/A",
            "Banknote_IoU_pct": f"{banknote_iou:.1f}%" if pd.notna(banknote_iou) else "N/A",
            "Tear_mAP50_pct": f"{mean_tear_map:.1f}%" if pd.notna(mean_tear_map) else "N/A",
            "Time_to_Correct_s": f"{mean_ttc:.2f} ± {std_ttc:.2f}" if pd.notna(mean_ttc) else "N/A",
            "Trigger_Rate_pct": f"{trigger_rate:.1f}%",
            # Raw numerical columns for plotting
            "_raw_energy_j": mean_energy,
            "_raw_full_lat_ms": avg_full_lat,
            "_raw_exact_acc": exact_acc,
            "_raw_denom_acc": denom_acc,
            "_raw_banknote_map50": banknote_map if pd.notna(banknote_map) else 0.0,
            "_raw_banknote_iou": banknote_iou if pd.notna(banknote_iou) else 0.0,
            "_raw_tear_map50": mean_tear_map if pd.notna(mean_tear_map) else 0.0,
            "_raw_dropped_pct": mean_dropped,
            "_raw_ttc_s": mean_ttc if pd.notna(mean_ttc) else 0.0
        })

    return pd.DataFrame(rows)


def run_wilcoxon_tests(
    df_summary: pd.DataFrame,
    metric: str = "energy_joules",
    baseline: str = "b0",
    candidate: str = "cascade"
) -> Dict[str, Any]:
    """
    Performs Wilcoxon signed-rank test on paired video sessions.
    """
    if df_summary.empty or "session_id" not in df_summary.columns:
        return {"error": "DataFrame is empty or missing session_id"}

    df_base = df_summary[df_summary["system"] == baseline].set_index("session_id")[metric]
    df_cand = df_summary[df_summary["system"] == candidate].set_index("session_id")[metric]

    common_idx = df_base.index.intersection(df_cand.index)
    if len(common_idx) < 3:
        return {
            "n_pairs": len(common_idx),
            "message": "Too few paired samples for Wilcoxon test (need >= 3)"
        }

    val_base = df_base.loc[common_idx].values
    val_cand = df_cand.loc[common_idx].values

    # Check if differences are all zeros
    diff = val_cand - val_base
    if np.all(diff == 0):
        return {
            "n_pairs": len(common_idx),
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "mean_diff": 0.0,
            "badge": "No difference"
        }

    try:
        stat, p_val = wilcoxon(val_base, val_cand)
        sig = p_val < 0.05
        if p_val < 0.001:
            badge = "p < 0.001 (*** Extremely Significant)"
        elif p_val < 0.01:
            badge = "p < 0.01 (** Highly Significant)"
        elif p_val < 0.05:
            badge = "p < 0.05 (* Statistically Significant)"
        else:
            badge = "p >= 0.05 (Not Significant)"

        return {
            "metric": metric,
            "n_pairs": len(common_idx),
            "baseline": baseline,
            "candidate": candidate,
            "statistic": float(stat),
            "p_value": float(p_val),
            "significant": sig,
            "badge": badge,
            "baseline_mean": float(np.mean(val_base)),
            "candidate_mean": float(np.mean(val_cand)),
            "mean_diff": float(np.mean(diff))
        }
    except Exception as e:
        return {"error": str(e)}


def compute_condition_breakdown(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Computes performance breakdown across environmental conditions (indoor, outdoor, backlight, overexposed, torn_clean, torn_bright).
    """
    if df_summary.empty or "condition" not in df_summary.columns:
        return pd.DataFrame()

    conditions = sorted(df_summary["condition"].unique())
    systems_order = ["b0", "b1", "cascade"]
    rows = []

    for cond in conditions:
        c_sub = df_summary[df_summary["condition"] == cond]
        b0_c = c_sub[c_sub["system"] == "b0"]
        b0_energy = b0_c["energy_joules"].mean() if not b0_c.empty else 1.0

        for sys_name in systems_order:
            sc_sub = c_sub[c_sub["system"] == sys_name]
            if sc_sub.empty:
                continue

            mean_energy = sc_sub["energy_joules"].mean()
            energy_savings = max(0.0, ((b0_energy - mean_energy) / b0_energy * 100.0)) if b0_energy > 0 else 0.0

            tear_sub = sc_sub["tear_map50"].dropna() if "tear_map50" in sc_sub.columns else pd.Series(dtype=float)
            mean_tear = tear_sub.mean() if not tear_sub.empty else np.nan

            rows.append({
                "Condition": cond,
                "System": SYSTEM_DISPLAY_NAMES.get(sys_name, sys_name),
                "System_Code": sys_name,
                "N_Sessions": len(sc_sub),
                "Exact_Acc_pct": round(sc_sub["exact_match_pct"].mean(), 1),
                "Denom_Acc_pct": round(sc_sub["denom_accuracy_pct"].mean(), 1),
                "Banknote_mAP50_pct": round(sc_sub["banknote_map50"].mean(), 1) if "banknote_map50" in sc_sub.columns else 0.0,
                "Banknote_IoU_pct": round(sc_sub["banknote_iou"].mean(), 1) if "banknote_iou" in sc_sub.columns else 0.0,
                "Tear_mAP50_pct": round(mean_tear, 1) if pd.notna(mean_tear) else "N/A",
                "Energy_J": round(mean_energy, 2),
                "Energy_Savings_pct": f"{energy_savings:.1f}%",
                "Trigger_Rate_pct": f"{sc_sub['called_full_rate_pct'].mean():.1f}%"
            })

    return pd.DataFrame(rows)


def compute_denomination_breakdown(df_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Computes performance breakdown across denominations (10k, 20k, 50k, 100k, 200k, 500k).
    """
    if df_summary.empty or "denomination" not in df_summary.columns:
        return pd.DataFrame()

    norm_map = {
        10000: "10k", 20000: "20k", 50000: "50k", 100000: "100k", 200000: "200k", 500000: "500k",
        "10000": "10k", "20000": "20k", "50000": "50k", "100000": "100k", "200000": "200k", "500000": "500k",
        "10": "10k", "20": "20k", "50": "50k", "100": "100k", "200": "200k", "500": "500k",
        10: "10k", 20: "20k", 50: "50k", 100: "100k", 200: "200k", 500: "500k",
        "10k": "10k", "20k": "20k", "50k": "50k", "100k": "100k", "200k": "200k", "500k": "500k"
    }
    df_c = df_summary.copy()
    df_c["norm_denom"] = df_c["denomination"].map(norm_map).fillna(df_c["denomination"].astype(str))

    denom_order = ["10k", "20k", "50k", "100k", "200k", "500k"]
    present_denoms = [d for d in denom_order if d in df_c["norm_denom"].unique()]
    if not present_denoms:
        present_denoms = sorted(df_c["norm_denom"].astype(str).unique())

    systems_order = ["b0", "b1", "cascade"]
    rows = []

    for denom in present_denoms:
        d_sub = df_c[df_c["norm_denom"] == denom]
        b0_d = d_sub[d_sub["system"] == "b0"]
        b0_energy = b0_d["energy_joules"].mean() if not b0_d.empty else 1.0

        for sys_name in systems_order:
            sd_sub = d_sub[d_sub["system"] == sys_name]
            if sd_sub.empty:
                continue

            mean_energy = sd_sub["energy_joules"].mean()
            energy_savings = max(0.0, ((b0_energy - mean_energy) / b0_energy * 100.0)) if b0_energy > 0 else 0.0

            rows.append({
                "Denomination": denom,
                "System": SYSTEM_DISPLAY_NAMES.get(sys_name, sys_name),
                "System_Code": sys_name,
                "N_Sessions": len(sd_sub),
                "Exact_Acc_pct": round(sd_sub["exact_match_pct"].mean(), 1),
                "Denom_Acc_pct": round(sd_sub["denom_accuracy_pct"].mean(), 1),
                "Banknote_mAP50_pct": round(sd_sub["banknote_map50"].mean(), 1) if "banknote_map50" in sd_sub.columns else 0.0,
                "Banknote_IoU_pct": round(sd_sub["banknote_iou"].mean(), 1) if "banknote_iou" in sd_sub.columns else 0.0,
                "Energy_J": round(mean_energy, 2),
                "Energy_Savings_pct": f"{energy_savings:.1f}%"
            })

    return pd.DataFrame(rows)


def generate_latex_table(
    agg_df: pd.DataFrame,
    cpu_model: str = "Intel/AMD Laptop CPU",
    os_name: str = "Windows 11"
) -> str:
    """
    Generates standard LaTeX code matching Table 15 in the paper design.
    """
    if agg_df.empty:
        return "% No data available to generate LaTeX table"

    latex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Real-Time Performance, Detection Precision, and Energy-Aware Simulation Benchmark across Alternative Framework Paradigms on Laptop CPU.}",
        r"\label{tab:cascade_realtime_benchmark}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{System Paradigm} & \textbf{Full Latency (ms)} & \textbf{Energy/Session (J)} & \textbf{Energy Savings} & \textbf{Drop Rate (\%)} & \textbf{Exact Acc (\%)} & \textbf{Banknote mAP@50 (\%)} & \textbf{TTC (s)} \\",
        r"\midrule"
    ]

    for _, row in agg_df.iterrows():
        sys_label = row.get("System", row.get("system_code", ""))
        # Bold proposed cascade
        if "Cascade" in str(sys_label):
            sys_label = rf"\textbf{{{sys_label}}}"

        def _esc(val: Any) -> str:
            s = str(val).strip()
            s = s.replace("± nan", "").strip()
            if s.endswith("%"):
                return s[:-1] + r"\%"
            return s.replace("%", r"\%")

        lat = _esc(row.get("Full_Latency_ms", "-"))
        energy = _esc(row.get("Energy_J", "-"))
        savings = _esc(row.get("Energy_Savings_pct", "-"))
        drop = _esc(row.get("Dropped_Frames_pct", "-"))
        acc = _esc(row.get("Exact_Accuracy_pct", "-"))
        b_map = _esc(row.get("Banknote_mAP50_pct", "-"))
        ttc = _esc(row.get("Time_to_Correct_s", "-"))

        latex.append(f"{sys_label} & {lat} & {energy} & {savings} & {drop} & {acc} & {b_map} & {ttc} \\\\")

    latex.extend([
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        rf"\flushleft{{\footnotesize \textit{{Note: Measured across 180 simulated sessions (6 denominations $\times$ 6 conditions $\times$ 5 samples) under {cpu_model} on {os_name}.}} }}",
        r"\end{table*}"
    ])

    return "\n".join(latex)


def generate_pareto_data(df_summary: pd.DataFrame) -> pd.DataFrame:
    """Generates Pareto frontier coordinates for plotting."""
    if df_summary.empty:
        return pd.DataFrame()
    
    agg = df_summary.groupby("system").agg({
        "energy_joules": "mean",
        "avg_full_latency_ms": "mean",
        "exact_match_pct": "mean",
        "denom_accuracy_pct": "mean",
        "dropped_rate_pct": "mean"
    }).reset_index()
    
    agg["System_Name"] = agg["system"].map(SYSTEM_DISPLAY_NAMES).fillna(agg["system"])
    return agg

