#!/usr/bin/env python3
"""
audit_phone_logs.py
Audit and aggregate detailed physical smartphone measurement logs.
"""

import glob
import json
import sys
from pathlib import Path
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

log_dir = Path('logs_phone_real/files')
files = sorted(log_dir.glob('*.jsonl'))

print("=" * 125)
print(f"{'NO':<3} | {'FILENAME':<35} | {'MODE':<8} | {'FRAMES':<8} | {'DURATION':<9} | {'FPS':<6} | {'FULL CALLS':<12} | {'AVG LAT':<8} | {'YOLO LAT':<8} | {'CURRENT mA':<8} | {'WATT':<6} | {'JOULES'}")
print("=" * 125)

all_data = {'b0': [], 'cascade': [], 'b1': []}

for idx, f in enumerate(files, 1):
    with open(f, 'r', encoding='utf-8') as fp:
        records = [json.loads(line) for line in fp if line.strip()]
    if not records:
        continue

    mode = records[0].get('mode', 'unknown')
    total_f = len(records)
    called = sum(1 for r in records if r.get('called_full', False))
    pct = (called / total_f * 100.0) if total_f > 0 else 0.0

    t0 = records[0].get('timestamp_ms', 0)
    t1 = records[-1].get('timestamp_ms', 0)
    duration_s = (t1 - t0) / 1000.0 if t1 > t0 else 30.0
    fps = total_f / duration_s

    total_lats = [r.get('latency_total_ms', 0.0) for r in records]
    avg_lat = float(np.mean(total_lats))

    yolo_lats = [r.get('latency_yolo_ms', 0.0) for r in records if r.get('called_full', False)]
    avg_yolo = float(np.mean(yolo_lats)) if yolo_lats else 0.0

    light_lats = [r.get('latency_light_ms', 0.0) for r in records]
    avg_light = float(np.mean(light_lats))

    mq_lats = [r.get('latency_mq_ms', 0.0) for r in records if r.get('called_full', False)]
    avg_mq = float(np.mean(mq_lats)) if mq_lats else 0.0

    # Analyze physical current and real-time power draw
    raw_currents = [r.get('battery_current_ma', 0.0) for r in records if r.get('battery_current_ma', 0.0) > 0]
    avg_raw_current = float(np.mean(raw_currents)) if raw_currents else 0.0
    real_current_ma = avg_raw_current * 1000.0

    raw_powers = [r.get('battery_power_mw', 0.0) for r in records if r.get('battery_power_mw', 0.0) > 0]
    avg_raw_power = float(np.mean(raw_powers)) if raw_powers else 0.0
    real_power_w = avg_raw_power

    raw_joules = records[-1].get('cumulative_joules', 0.0)
    real_joules = raw_joules * 1000.0

    denoms = [r.get('denomination') for r in records if r.get('denomination')]
    denom_set = set(denoms)

    row = {
        'file': f.name,
        'mode': mode,
        'frames': total_f,
        'duration_s': duration_s,
        'fps': fps,
        'called': called,
        'pct': pct,
        'avg_lat': avg_lat,
        'avg_light': avg_light,
        'avg_mq': avg_mq,
        'avg_yolo': avg_yolo,
        'current_ma': real_current_ma,
        'power_w': real_power_w,
        'joules': real_joules,
        'mj_per_frame': (real_joules * 1000.0) / total_f if total_f > 0 else 0.0,
        'denoms': denom_set
    }
    all_data[mode].append(row)

    call_str = f"{called}/{total_f} ({pct:.1f}%)"
    print(f"{idx:<3} | {f.name:<35} | {mode.upper():<8} | {total_f:<8} | {duration_s:<7.1f}s | {fps:<6.2f} | {call_str:<12} | {avg_lat:<8.1f} | {avg_yolo:<8.1f} | {real_current_ma:<8.1f} | {real_power_w:<6.2f} | {real_joules:<6.2f} J")

print("=" * 125)

print("\n" + "#" * 60)
print("### EMPIRICAL TELEMETRY STATISTICS BY PARADIGM (MEAN +- STD)")
print("#" * 60)

for m in ['b0', 'cascade', 'b1']:
    rows = all_data[m]
    n = len(rows)
    fps_v = [r['fps'] for r in rows]
    lat_v = [r['avg_lat'] for r in rows]
    light_v = [r['avg_light'] for r in rows]
    mq_v = [r['avg_mq'] for r in rows if r['avg_mq'] > 0]
    yolo_v = [r['avg_yolo'] for r in rows if r['avg_yolo'] > 0]
    pct_v = [r['pct'] for r in rows]
    curr_v = [r['current_ma'] for r in rows]
    pwr_v = [r['power_w'] for r in rows]
    j_v = [r['joules'] for r in rows]
    mj_v = [r['mj_per_frame'] for r in rows]
    all_denoms = set().union(*[r['denoms'] for r in rows])

    print(f"\n>>> PARADIGM: {m.upper()} (N = {n} trials)")
    print(f"  • Detected Denominations: {all_denoms if all_denoms else 'None'}")
    print(f"  • Throughput (FPS): {np.mean(fps_v):.2f} +- {np.std(fps_v):.2f} FPS")
    print(f"  • End-to-End Latency: {np.mean(lat_v):.2f} +- {np.std(lat_v):.2f} ms")
    print(f"  • Quality-Gate Latency: {np.mean(light_v):.2f} +- {np.std(light_v):.2f} ms")
    if mq_v:
        print(f"  • MQTone Latency: {np.mean(mq_v):.2f} +- {np.std(mq_v):.2f} ms")
    if yolo_v:
        print(f"  • YOLOv8n Latency: {np.mean(yolo_v):.2f} +- {np.std(yolo_v):.2f} ms")
    print(f"  • Full Pipeline Trigger Rate: {np.mean(pct_v):.2f}% +- {np.std(pct_v):.2f}%")
    print(f"  • Current Consumption: {np.mean(curr_v):.1f} +- {np.std(curr_v):.1f} mA")
    print(f"  • Active Power Draw: {np.mean(pwr_v):.2f} +- {np.std(pwr_v):.2f} Watts")
    print(f"  • Total Session Energy: {np.mean(j_v):.2f} +- {np.std(j_v):.2f} Joules")
    print(f"  • Energy per Frame: {np.mean(mj_v):.2f} +- {np.std(mj_v):.2f} mJ/frame")
