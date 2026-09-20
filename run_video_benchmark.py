"""
run_video_benchmark.py
======================
Executes the comprehensive real-world video benchmark across 3 systems:
- B0: Full Every Frame (MQTone + YOLO)
- B1: Single-Shot (Sampled Delay)
- Cascade: Proposed Adaptive Pipeline (Quality-Gate + State Machine + MQTone + YOLO)

Test Suite:
36 real-world test videos in `video_test/` covering:
- 6 denominations: 10k, 20k, 50k, 100k, 200k, 500k
- 6 environmental conditions: indoor, outdoor, backlight, overexposed, torn_clean, torn_bright
-> Total: 36 real video sessions x 3 systems = 108 runs

Outputs:
- logs_sim/video_benchmark_raw.csv
- logs_sim/video_table1_systems_overall.csv
- logs_sim/video_table2_condition_breakdown.csv
- logs_sim/video_table3_denomination_breakdown.csv
- logs_sim/video_table_latex_code.tex
- logs_sim/video_vs_simulation_comparison.csv
"""

import os
import sys
import time
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from sim_engine.packages import LightPackage, FullPackage, DetectorOnlyPackage
from sim_engine.frame_source import FrameSource
from sim_engine.experiment_runner import run_session
from sim_engine.stats_analyzer import (
    compute_system_aggregate_metrics,
    compute_condition_breakdown,
    compute_denomination_breakdown,
    generate_latex_table,
    SYSTEM_DISPLAY_NAMES
)

VIDEO_DIR = PROJECT_ROOT / "video_test"
OUTPUT_DIR = PROJECT_ROOT / "logs_sim"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_CSV_PATH = OUTPUT_DIR / "video_benchmark_raw.csv"
TABLE1_CSV_PATH = OUTPUT_DIR / "video_table1_systems_overall.csv"
TABLE2_CSV_PATH = OUTPUT_DIR / "video_table2_condition_breakdown.csv"
TABLE3_CSV_PATH = OUTPUT_DIR / "video_table3_denomination_breakdown.csv"
LATEX_TEX_PATH = OUTPUT_DIR / "video_table_latex_code.tex"
COMPARISON_CSV_PATH = OUTPUT_DIR / "video_vs_simulation_comparison.csv"


def safe_to_csv(df: pd.DataFrame, path: Path, retries: int = 6, delay: float = 0.5):
    """Safely saves DataFrame to CSV on Windows with retry."""
    for _ in range(retries):
        try:
            df.to_csv(path, index=False)
            return
        except Exception:
            time.sleep(delay)
    try:
        tmp_path = path.with_suffix(f".tmp_{int(time.time() * 1000)}")
        df.to_csv(tmp_path, index=False)
        if tmp_path.exists():
            os.replace(tmp_path, path)
    except Exception as e:
        print(f"⚠️ Warning saving CSV to {path}: {e}", flush=True)


def parse_video_sessions(video_dir: Path) -> List[Dict[str, Any]]:
    """Discovers and parses all 36 video files in video_test/."""
    videos = sorted(list(video_dir.glob("*.mp4")))
    sessions = []

    for v in videos:
        name = v.stem
        m = re.match(r"^(backlight|indoor|outdoor|overexposed_torn|overexposed|torn_clean)(\d+)$", name)
        if m:
            cond, denom_num = m.groups()
            is_torn = ("torn" in cond)
            cond_clean = "torn_bright" if cond == "overexposed_torn" else cond
            denom = f"{denom_num}k"
            sessions.append({
                "video_name": v.name,
                "video_path": str(v),
                "session_id": f"real_{cond_clean}_{denom}",
                "condition": cond_clean,
                "denomination": denom,
                "is_torn": is_torn
            })
        else:
            print(f"⚠️ Warning: Could not parse video filename pattern: {v.name}")

    return sessions


def build_comparison_table(video_t1: pd.DataFrame, sim_t1_path: Path) -> pd.DataFrame:
    """Generates comparison between Real Video Benchmark vs 60-frame Synthetic Simulation."""
    if not sim_t1_path.exists() or video_t1.empty:
        return pd.DataFrame()

    sim_t1 = pd.read_csv(sim_t1_path)
    comparison_rows = []

    for sys_code in ["b0", "b1", "cascade"]:
        v_sub = video_t1[video_t1["system_code"] == sys_code]
        s_sub = sim_t1[sim_t1["system_code"] == sys_code]
        if v_sub.empty or s_sub.empty:
            continue

        v_row = v_sub.iloc[0]
        s_row = s_sub.iloc[0]

        sys_label = SYSTEM_DISPLAY_NAMES.get(sys_code, sys_code)
        comparison_rows.append({
            "System": sys_label,
            "Benchmark_Type": "1-Image Synthetic (60 Frames)",
            "N_Sessions": s_row.get("N_Sessions", 180),
            "Exact_Accuracy": s_row.get("Exact_Accuracy_pct", "N/A"),
            "Denom_Accuracy": s_row.get("Denom_Accuracy_pct", "N/A"),
            "Energy_J": s_row.get("Energy_J", "N/A"),
            "Energy_Savings": s_row.get("Energy_Savings_pct", "0.0%"),
            "Full_Latency_ms": s_row.get("Full_Latency_ms", "N/A"),
            "Trigger_Rate": s_row.get("Trigger_Rate_pct", "N/A"),
            "Time_to_Correct_s": s_row.get("Time_to_Correct_s", "N/A")
        })
        comparison_rows.append({
            "System": sys_label,
            "Benchmark_Type": "Real Video Benchmark (200-280 Frames)",
            "N_Sessions": v_row.get("N_Sessions", 36),
            "Exact_Accuracy": v_row.get("Exact_Accuracy_pct", "N/A"),
            "Denom_Accuracy": v_row.get("Denom_Accuracy_pct", "N/A"),
            "Energy_J": v_row.get("Energy_J", "N/A"),
            "Energy_Savings": v_row.get("Energy_Savings_pct", "0.0%"),
            "Full_Latency_ms": v_row.get("Full_Latency_ms", "N/A"),
            "Trigger_Rate": v_row.get("Trigger_Rate_pct", "N/A"),
            "Time_to_Correct_s": v_row.get("Time_to_Correct_s", "N/A")
        })

    return pd.DataFrame(comparison_rows)


def main():
    parser = argparse.ArgumentParser(description="Run Real-World Video Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of video sessions (e.g. 2 for quick verification)")
    parser.add_argument("--cooldown", type=float, default=0.5, help="Cooldown sleep between runs in seconds")
    parser.add_argument("--fps", type=float, default=30.0, help="Target FPS for frame source replay")
    args = parser.parse_args()

    print("=" * 85, flush=True)
    print("🎬 CASHVISION REAL-WORLD VIDEO BENCHMARK (36 VIDEOS x 3 SYSTEMS = 108 RUNS)", flush=True)
    print("=" * 85, flush=True)

    sessions = parse_video_sessions(VIDEO_DIR)
    print(f"✅ Found {len(sessions)} real video sessions in {VIDEO_DIR}", flush=True)

    if args.limit:
        print(f"⚠️ [TEST MODE] Limiting run to first {args.limit} sessions.", flush=True)
        sessions = sessions[:args.limit]

    # 1. Load Neural Network Packages
    print("\n📦 Loading Neural Network Packages on CPU...", flush=True)
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
        weights_path=str(PROJECT_ROOT / "checkpointyolov8n/full_pipeline_weights.pt"),
        device="cpu"
    )
    print(f"✅ Neural network packages initialized in {time.time() - t0_load:.2f}s.\n", flush=True)

    # 2. Check existing runs for incremental resume
    existing_runs = set()
    existing_summaries: List[Dict[str, Any]] = []
    if RAW_CSV_PATH.exists():
        try:
            prev_df = pd.read_csv(RAW_CSV_PATH)
            for _, r in prev_df.iterrows():
                key = (str(r["session_id"]), str(r["system"]))
                existing_runs.add(key)
                existing_summaries.append(r.to_dict())
            print(f"🔄 Resuming from existing run log: {len(existing_summaries)} runs already completed.", flush=True)
        except Exception as e:
            print(f"Warning reading existing raw csv: {e}", flush=True)

    systems = ["b0", "b1", "cascade"]
    cascade_params = {
        "quality_threshold": 0.60,
        "required_stable_frames": 8,
        "max_verify_frames": 3,
        "confirm_conf_threshold": 0.60
    }

    all_summaries = list(existing_summaries)
    total_sessions = len(sessions)
    total_expected_runs = total_sessions * len(systems)

    t_bench_start = time.time()

    for s_idx, s_info in enumerate(sessions, 1):
        sess_id = s_info["session_id"]
        v_path = s_info["video_path"]
        cond = s_info["condition"]
        denom = s_info["denomination"]
        is_torn = s_info["is_torn"]

        gt_dict = {
            "denomination": denom,
            "condition": cond,
            "is_torn": is_torn,
            "gt_banknote_box": None,
            "gt_tear_boxes": []
        }

        sys_needed = [s for s in systems if (sess_id, s) not in existing_runs]
        if not sys_needed:
            continue

        print(f"\n[{s_idx}/{total_sessions}] ▶ Video: {s_info['video_name']} ({denom}, {cond}, torn={is_torn})", flush=True)

        for sys_name in sys_needed:
            t0_sys = time.time()
            frame_source = FrameSource(
                mode="replay",
                video_path=v_path,
                target_fps=args.fps
            )

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

            dur = time.time() - t0_sys
            full_calls = summary["called_full_count"]
            tot_frames = summary["total_frames"]
            acc = summary["exact_match_pct"]
            en_j = summary["energy_joules"]
            print(f"  • {sys_name.upper():<7} | Time: {dur:.1f}s | Frames: {tot_frames} | Calls: {full_calls} ({summary['called_full_rate_pct']:.1f}%) | Energy: {en_j:.1f}J | Acc: {acc:.0f}%", flush=True)

            if args.cooldown > 0:
                time.sleep(args.cooldown)

        # Incremental save & update summary tables after every video session
        df_current = pd.DataFrame(all_summaries)
        safe_to_csv(df_current, RAW_CSV_PATH)

        try:
            t1 = compute_system_aggregate_metrics(df_current)
            safe_to_csv(t1, TABLE1_CSV_PATH)

            t2 = compute_condition_breakdown(df_current)
            safe_to_csv(t2, TABLE2_CSV_PATH)

            t3 = compute_denomination_breakdown(df_current)
            safe_to_csv(t3, TABLE3_CSV_PATH)

            latex_str = generate_latex_table(t1)
            with open(LATEX_TEX_PATH, "w", encoding="utf-8") as f:
                f.write(latex_str)

            # Compare against 1-image synthetic simulation
            sim_t1_path = OUTPUT_DIR / "table1_systems_overall.csv"
            comp_df = build_comparison_table(t1, sim_t1_path)
            if not comp_df.empty:
                safe_to_csv(comp_df, COMPARISON_CSV_PATH)
        except Exception as e:
            print(f"Warning updating live summary: {e}", flush=True)

    elapsed_total = time.time() - t_bench_start
    print("\n" + "=" * 85, flush=True)
    print(f"🎉 REAL-WORLD VIDEO BENCHMARK COMPLETE in {elapsed_total / 60:.2f} minutes!", flush=True)
    print("=" * 85, flush=True)

    final_df = pd.DataFrame(all_summaries)
    t1_final = compute_system_aggregate_metrics(final_df)
    print("\n📊 TABLE 1: OVERALL SYSTEM PERFORMANCE ON REAL VIDEOS:")
    cols_to_print = ["System", "N_Sessions", "Exact_Accuracy_pct", "Denom_Accuracy_pct", "Energy_J", "Energy_Savings_pct", "Trigger_Rate_pct", "Time_to_Correct_s"]
    available_cols = [c for c in cols_to_print if c in t1_final.columns]
    print(t1_final[available_cols].to_string(index=False))

    comp_df = build_comparison_table(t1_final, OUTPUT_DIR / "table1_systems_overall.csv")
    if not comp_df.empty:
        print("\n📈 COMPARISON: REAL VIDEO vs. 1-IMAGE 60-FRAME SYNTHETIC SIMULATION:")
        print(comp_df.to_string(index=False))

    print(f"\n📁 All tables exported to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
