#!/usr/bin/env python3
"""
parse_phone_logs.py
Parse and analyze real-world smartphone measurement logs in logs_phone_real/files/
Export comparative summary table across paradigms: CASCADE, B0, B1, B2
"""

import sys
import json
from pathlib import Path

# Fix UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def analyze_logs():
    log_dir = Path("logs_phone_real/files")
    if not log_dir.exists():
        print(f"Directory not found: {log_dir}")
        return

    log_files = sorted(log_dir.glob("*.jsonl"))
    if not log_files:
        print("No .jsonl log files found!")
        return

    print("=" * 95)
    print(f"{'SESSION':<32} | {'MODE':<8} | {'FRAMES':<6} | {'FULL CALLS':<10} | {'LATENCY':<9} | {'POWER (mW)':<11} | {'ENERGY (J)'}")
    print(f"{'':<32} | {'':<8} | {'':<6} | {'(Count / %)':<10} | {'(Avg ms)':<9} | {'(Avg mW)':<11} | {'(Joules)'}")
    print("=" * 95)

    summaries = []

    for f in log_files:
        records = []
        with open(f, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except Exception:
                        pass

        if not records:
            continue

        mode = records[0].get("mode", "unknown").upper()
        total_frames = len(records)
        called_full = sum(1 for r in records if r.get("called_full", False))
        called_full_pct = (called_full / total_frames * 100.0) if total_frames > 0 else 0.0

        # Latency
        latencies = [r.get("latency_total_ms", 0.0) for r in records]
        avg_total_ms = sum(latencies) / len(latencies) if latencies else 0.0

        light_lats = [r.get("latency_light_ms", 0.0) for r in records]
        avg_light_ms = sum(light_lats) / len(light_lats) if light_lats else 0.0

        yolo_lats = [r.get("latency_yolo_ms", 0.0) for r in records if r.get("called_full", False)]
        avg_yolo_ms = sum(yolo_lats) / len(yolo_lats) if yolo_lats else 0.0

        # Energy
        powers = [r.get("battery_power_mw", 0.0) for r in records if r.get("battery_power_mw", 0.0) > 0]
        avg_power_mw = sum(powers) / len(powers) if powers else 0.0

        final_joules = records[-1].get("cumulative_joules", 0.0)

        # Detected denomination
        denoms = set(r.get("denomination") for r in records if r.get("denomination"))
        denom_str = ", ".join(denoms) if denoms else "None"

        summaries.append({
            "filename": f.name,
            "mode": mode,
            "frames": total_frames,
            "called_full": called_full,
            "called_pct": called_full_pct,
            "avg_latency": avg_total_ms,
            "avg_light": avg_light_ms,
            "avg_yolo": avg_yolo_ms,
            "avg_power": avg_power_mw,
            "joules": final_joules,
            "denoms": denom_str
        })

        called_str = f"{called_full}/{total_frames} ({called_full_pct:.0f}%)"
        print(f"{f.name:<32} | {mode:<8} | {total_frames:<6} | {called_str:<10} | {avg_total_ms:<9.1f} | {avg_power_mw:<11.2f} | {final_joules:.4f} J")

    print("=" * 95)
    print("\nON-DEVICE DETECTION & MODULE TELEMETRY DETAILS:")
    for s in summaries:
        print(f" • [{s['mode']}] {s['filename']}:")
        print(f"     + Detected denominations: {s['denoms']}")
        print(f"     + Average LightPackage latency: {s['avg_light']:.2f} ms")
        if s['avg_yolo'] > 0:
            print(f"     + Average YOLOv8n latency: {s['avg_yolo']:.2f} ms")
        print(f"     + Full Package trigger rate: {s['called_pct']:.1f}%")
        print(f"     + Total session energy: {s['joules']:.4f} Joules")

if __name__ == "__main__":
    analyze_logs()
