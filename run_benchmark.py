"""
run_benchmark.py
================
Executes the comprehensive simulation benchmark across 3 systems:
- B0: Full Every Frame (MQTone + YOLO)
- B1: Single-Shot (Sampled Delay)
- Cascade: Proposed Adaptive Pipeline (Quality-Gate + State Machine + MQTone + YOLO)

Design:
- 6 denominations (10k, 20k, 50k, 100k, 200k, 500k)
- 6 environmental conditions (indoor, outdoor, backlight, overexposed, torn_clean, torn_bright)
- 5 randomly sampled images per cell (seed=42)
-> Total: 6 x 6 x 5 = 180 video sessions (each 60 frames = 2.0s @ 30 FPS)
-> Total runs: 180 x 3 = 540 system runs

Outputs:
- logs_sim/benchmark_raw.csv
- logs_sim/benchmark_matrix_summary.csv
- logs_sim/table1_systems_overall.csv
- logs_sim/table2_condition_breakdown.csv
- logs_sim/table3_denomination_breakdown.csv
- logs_sim/table_latex_code.tex
- logs_sim/wilcoxon_energy_results.json
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim_engine.packages import LightPackage, FullPackage, DetectorOnlyPackage
from sim_engine.frame_source import FrameSource, DatasetSequenceGenerator
from sim_engine.experiment_runner import run_session
from sim_engine.stats_analyzer import (
    compute_system_aggregate_metrics,
    compute_condition_breakdown,
    compute_denomination_breakdown,
    generate_latex_table,
    run_wilcoxon_tests
)


TARGET_DENOMINATIONS = ["10k", "20k", "50k", "100k", "200k", "500k"]
TARGET_CONDITIONS = ["indoor", "outdoor", "backlight", "overexposed", "torn_clean", "torn_bright"]
SAMPLES_PER_CELL = 5
RANDOM_SEED = 42
TOTAL_FRAMES = 60
FPS = 30.0

RAW_CSV_PATH = PROJECT_ROOT / "logs_sim" / "benchmark_raw.csv"
MATRIX_SUMMARY_CSV_PATH = PROJECT_ROOT / "logs_sim" / "benchmark_matrix_summary.csv"
TABLE1_CSV_PATH = PROJECT_ROOT / "logs_sim" / "table1_systems_overall.csv"
TABLE2_CSV_PATH = PROJECT_ROOT / "logs_sim" / "table2_condition_breakdown.csv"
TABLE3_CSV_PATH = PROJECT_ROOT / "logs_sim" / "table3_denomination_breakdown.csv"
LATEX_TEX_PATH = PROJECT_ROOT / "logs_sim" / "table_latex_code.tex"
WILCOXON_JSON_PATH = PROJECT_ROOT / "logs_sim" / "wilcoxon_energy_results.json"


def safe_to_csv(df: pd.DataFrame, path: Path, retries: int = 6, delay: float = 0.5):
    """Safely saves DataFrame to CSV on Windows, retrying if another process has a temporary lock."""
    for attempt in range(retries):
        try:
            df.to_csv(path, index=False)
            return
        except PermissionError:
            time.sleep(delay)
        except Exception:
            time.sleep(delay)
    try:
        tmp_path = path.with_suffix(f".tmp_{int(time.time() * 1000)}")
        df.to_csv(tmp_path, index=False)
        if tmp_path.exists():
            os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️ Warning saving CSV to {path}: {e}", flush=True)


DENOM_NORM = {
    10000: "10k", 20000: "20k", 50000: "50k", 100000: "100k", 200000: "200k", 500000: "500k",
    "10000": "10k", "20000": "20k", "50000": "50k", "100000": "100k", "200000": "200k", "500000": "500k",
    "10": "10k", "20": "20k", "50": "50k", "100": "100k", "200": "200k", "500": "500k",
    "10k": "10k", "20k": "20k", "50k": "50k", "100k": "100k", "200k": "200k", "500k": "500k"
}


def load_balanced_samples(df_source: pd.DataFrame, seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """
    Samples exactly SAMPLES_PER_CELL images per (denomination, condition) cell.
    Guarantees 6 denominations x 6 conditions x 5 samples = 180 sessions.
    """
    df = df_source.copy()
    df["norm_denom"] = df["denomination_class"].map(DENOM_NORM)
    df["norm_cond"] = df["condition"].astype(str).str.strip().str.lower()

    selected_samples = []
    for denom in TARGET_DENOMINATIONS:
        for cond in TARGET_CONDITIONS:
            cell_df = df[(df["norm_denom"] == denom) & (df["norm_cond"] == cond)]
            n_avail = len(cell_df)
            if n_avail == 0:
                print(f"⚠️ Warning: No samples found for ({denom}, {cond})!")
                continue

            n_pick = min(SAMPLES_PER_CELL, n_avail)
            sampled = cell_df.sample(n=n_pick, random_state=seed)
            for s_idx, (_, s_row) in enumerate(sampled.iterrows()):
                selected_samples.append({
                    "sample_id": f"{denom}_{cond}_s{s_idx + 1}",
                    "denomination": denom,
                    "condition": cond,
                    "image_path": str(s_row["img_path"]),
                    "filename": str(s_row["filename"]),
                    "is_torn": bool(str(s_row.get("is_torn", "")).lower() == "true")
                })

    print(f"✅ Total sampled sessions: {len(selected_samples)} (Target: {len(TARGET_DENOMINATIONS) * len(TARGET_CONDITIONS) * SAMPLES_PER_CELL})")
    return selected_samples


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Full Paper Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of sessions (e.g. 1 or 2 for testing)")
    args = parser.parse_args()

    print("=" * 80)
    print("🚀 CASHVISION FULL PAPER BENCHMARK (180 SESSIONS x 4 SYSTEMS = 720 RUNS)")
    print("=" * 80)
    print(f"Denominations ({len(TARGET_DENOMINATIONS)}): {TARGET_DENOMINATIONS}")
    print(f"Conditions ({len(TARGET_CONDITIONS)}): {TARGET_CONDITIONS}")
    print(f"Samples per cell: {SAMPLES_PER_CELL} | Frames per session: {TOTAL_FRAMES} (2.0s @ 30 FPS)")

    output_dir = PROJECT_ROOT / "logs_sim"
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = PROJECT_ROOT / "metadata.csv"
    if not metadata_file.exists():
        raise FileNotFoundError(f"metadata.csv not found at {metadata_file}")

    print("\n📂 Scanning dataset and matching annotations...")
    seq_gen = DatasetSequenceGenerator(metadata_path=str(metadata_file))
    samples = load_balanced_samples(seq_gen.df, seed=RANDOM_SEED)
    if args.limit:
        print(f"⚠️ [TEST MODE] Limiting run to first {args.limit} sessions.")
        samples = samples[:args.limit]

    # 1. Initialize Neural Network Packages
    print("\n📦 Loading neural network models on CPU...")
    t0_load = time.time()
    light_pkg = LightPackage(
        weights_path=str(PROJECT_ROOT / "checkpointyolov8n/quality_gate_weights.pt"),
        device="cpu"
    )
    full_pkg = FullPackage(
        weights_path=str(PROJECT_ROOT / "checkpointyolov8n/full_pipeline_weights.pt"),
        device="cpu"
    )
    detector_only_pkg = DetectorOnlyPackage(
        weights_path=str(PROJECT_ROOT / "resultsv8n/final_config_none/fold_1/full_pipeline_weights.pt"),
        device="cpu"
    )
    print(f"✅ All packages loaded in {time.time() - t0_load:.2f}s.")

    # 2. Check existing progress for incremental resume
    existing_runs = set()
    existing_summaries: List[Dict[str, Any]] = []
    if RAW_CSV_PATH.exists():
        try:
            prev_df = pd.read_csv(RAW_CSV_PATH)
            for _, r in prev_df.iterrows():
                key = (str(r["session_id"]), str(r["system"]))
                existing_runs.add(key)
                existing_summaries.append(r.to_dict())
            print(f"🔄 Resuming from existing run log: {len(existing_summaries)} runs already completed.")
        except Exception as e:
            print(f"Warning reading existing raw csv: {e}")

    systems = ["b0", "b1", "cascade"]
    cascade_params = {
        "min_confidence": 0.45,
        "quality_threshold": 0.60,
        "required_stable_frames": 8,
        "max_verify_frames": 2,
        "cooldown_frames": 20
    }

    all_summaries = list(existing_summaries)
    total_sessions = len(samples)

    start_bench_time = time.time()

    for s_idx, sample_info in enumerate(samples, 1):
        sess_id = sample_info["sample_id"]
        cond = sample_info["condition"]
        denom = sample_info["denomination"]
        img_path = sample_info["image_path"]

        # Check if all 4 systems already done for this session
        sys_needed = [sys for sys in systems if (sess_id, sys) not in existing_runs]
        if not sys_needed:
            continue

        # Generate realistic 60-frame sequence for this sample
        frames_bgr, gt_dict = seq_gen.create_condition_session(
            sample_img_path=img_path,
            condition=cond,
            denomination=denom,
            total_frames=TOTAL_FRAMES,
            fps=FPS
        )

        session_t0 = time.time()
        session_results = {}

        for sys_name in sys_needed:
            frame_source = FrameSource(
                mode="sequence",
                image_sequence=frames_bgr,
                target_fps=FPS
            )
            session_dict = {
                "session_id": sess_id,
                "source": frame_source,
                "ground_truth": gt_dict
            }

            summary = run_session(
                system=sys_name,
                source=frame_source,
                session_id=sess_id,
                ground_truth=gt_dict,
                light_pkg=light_pkg,
                full_pkg=full_pkg,
                detector_only_pkg=detector_only_pkg,
                cascade_params=cascade_params,
                tdp_watts=28.0
            )
            all_summaries.append(summary)
            existing_runs.add((sess_id, sys_name))
            session_results[sys_name] = summary

        # Write incremental CSV once per completed session and update summary tables live
        df_current = pd.DataFrame(all_summaries)
        safe_to_csv(df_current, RAW_CSV_PATH)
        safe_to_csv(df_current, MATRIX_SUMMARY_CSV_PATH)

        try:
            live_t1 = compute_system_aggregate_metrics(df_current)
            safe_to_csv(live_t1, TABLE1_CSV_PATH)
            live_t2 = compute_condition_breakdown(df_current)
            safe_to_csv(live_t2, TABLE2_CSV_PATH)
            live_t3 = compute_denomination_breakdown(df_current)
            safe_to_csv(live_t3, TABLE3_CSV_PATH)
            latex_code = generate_latex_table(live_t1, cpu_model="Laptop CPU (28W TDP)", os_name=sys.platform)
            with open(LATEX_TEX_PATH, "w", encoding="utf-8") as f:
                f.write(latex_code)
        except Exception:
            pass

        sess_duration = time.time() - session_t0
        cas_res = session_results.get("cascade")
        b0_res = session_results.get("b0")

        if cas_res and b0_res:
            cas_e = cas_res["energy_joules"]
            b0_e = b0_res["energy_joules"]
            savings = max(0.0, (b0_e - cas_e) / b0_e * 100.0) if b0_e > 0 else 0.0
            acc_icon = "🎯" if cas_res["exact_match_pct"] == 100 else "⚠️"
            print(
                f"[{s_idx:03d}/{total_sessions}] ({denom:>4}, {cond:<11}) | "
                f"Done in {sess_duration:4.1f}s | "
                f"Cascade E: {cas_e:.2f}J (Save {savings:4.1f}%) | "
                f"ExactAcc: {acc_icon} {cas_res['exact_match_pct']:3.0f}% | "
                f"Banknote mAP: {cas_res['banknote_map50']:3.0f}%",
                flush=True
            )
        else:
            print(f"[{s_idx:03d}/{total_sessions}] ({denom:>4}, {cond:<11}) | Done in {sess_duration:4.1f}s", flush=True)

    total_duration = time.time() - start_bench_time
    print(f"\n🎉 Benchmark complete in {total_duration / 60.0:.2f} minutes!")

    # 3. Generate Aggregated Tables
    final_df = pd.DataFrame(all_summaries)
    print("\n" + "=" * 80)
    print("📊 GENERATING PUBLICATION TABLES & STATISTICAL METRICS")
    print("=" * 80)

    # Table 1: Overall System Comparison
    table1 = compute_system_aggregate_metrics(final_df)
    safe_to_csv(table1, TABLE1_CSV_PATH)
    print("\n📑 TABLE 1: OVERALL SYSTEM COMPARISON (Mean ± Std)")
    cols_t1_disp = [
        "System", "Full_Latency_ms", "Peak_RAM_MB", "Energy_J",
        "Energy_Savings_pct", "Dropped_Frames_pct", "Exact_Accuracy_pct",
        "Banknote_mAP50_pct", "Tear_mAP50_pct", "Time_to_Correct_s", "Trigger_Rate_pct"
    ]
    print(table1[[c for c in cols_t1_disp if c in table1.columns]].to_string(index=False))

    # Table 2: Condition Breakdown
    table2 = compute_condition_breakdown(final_df)
    safe_to_csv(table2, TABLE2_CSV_PATH)
    print("\n📑 TABLE 2: BREAKDOWN BY ENVIRONMENTAL CONDITION (First 12 rows)")
    cols_t2_disp = ["Condition", "System", "Exact_Acc_pct", "Banknote_mAP50_pct", "Tear_mAP50_pct", "Energy_J", "Energy_Savings_pct", "Trigger_Rate_pct"]
    print(table2[[c for c in cols_t2_disp if c in table2.columns]].head(12).to_string(index=False))

    # Table 3: Denomination Breakdown
    table3 = compute_denomination_breakdown(final_df)
    safe_to_csv(table3, TABLE3_CSV_PATH)
    print("\n📑 TABLE 3: BREAKDOWN BY DENOMINATION (First 12 rows)")
    cols_t3_disp = ["Denomination", "System", "Exact_Acc_pct", "Banknote_mAP50_pct", "Energy_J", "Energy_Savings_pct"]
    print(table3[[c for c in cols_t3_disp if c in table3.columns]].head(12).to_string(index=False))

    # LaTeX Table
    latex_code = generate_latex_table(table1, cpu_model="Laptop CPU (28W TDP)", os_name=sys.platform)
    with open(LATEX_TEX_PATH, "w", encoding="utf-8") as f:
        f.write(latex_code)
    print(f"\n📄 Exported LaTeX code to: {LATEX_TEX_PATH}")

    # Wilcoxon Signed-Rank Test
    wilcoxon_res = run_wilcoxon_tests(final_df, metric="energy_joules", baseline="b0", candidate="cascade")
    with open(WILCOXON_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(wilcoxon_res, f, indent=2)
    print(f"📈 Wilcoxon Test (Cascade vs B0 Energy): p={wilcoxon_res.get('p_value', 1.0):.6e} -> {wilcoxon_res.get('badge')}")

    print("\n" + "=" * 80)
    print("✅ BENCHMARK EXECUTION AND EXPORT COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
