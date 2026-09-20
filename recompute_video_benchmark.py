import json
import glob
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
LOGS_DIR = PROJECT_ROOT / "logs_sim"

def compute_box_iou(b1, b2):
    if not b1 or not b2 or len(b1) < 4 or len(b2) < 4:
        return 0.0
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def recompute_all_sessions():
    all_jsonls = sorted(glob.glob(str(LOGS_DIR / "real_*.jsonl")))
    print(f"Found {len(all_jsonls)} JSONL session log files.")
    
    summaries = []
    
    for log_path in all_jsonls:
        lines = [json.loads(line) for line in open(log_path, encoding="utf-8")]
        if not lines:
            continue
            
        first = lines[0]
        sess_id = first["session_id"]
        system = first["system"]
        gt = first.get("ground_truth", {})
        gt_denom = str(gt.get("denomination", "")).strip().lower()
        gt_torn = bool(gt.get("is_torn", False))
        condition = gt.get("condition", "")
        
        total_frames = len(lines)
        dropped_count = sum(1 for e in lines if e.get("dropped"))
        called_full_count = sum(1 for e in lines if e.get("called_full"))
        full_latencies = [e["full_latency_ms"] for e in lines if e.get("called_full")]
        light_latencies = [e["light_latency_ms"] for e in lines]
        
        # Triggered frames with detections
        triggered_frames = [e for e in lines if e.get("called_full") and e.get("result")]
        
        # 1. Denomination Consensus (Confidence-weighted voting)
        denom_scores = {}
        denom_best_entry = {}
        for e in triggered_frames:
            r = e.get("result") or {}
            d_name = r.get("denomination")
            d_conf = float(r.get("confidence") or 0.0)
            if d_name:
                norm_d = str(d_name).replace("k", "").replace(".0", "").strip().lower()
                denom_scores[norm_d] = denom_scores.get(norm_d, 0.0) + d_conf
                if norm_d not in denom_best_entry or d_conf > denom_best_entry[norm_d]["conf"]:
                    denom_best_entry[norm_d] = {"entry": e, "conf": d_conf, "result": r}

        if denom_scores:
            final_pred_denom = max(denom_scores.items(), key=lambda x: x[1])[0]
            outcome_result = denom_best_entry[final_pred_denom]["result"]
        else:
            outcome_result = None
            final_pred_denom = ""

        # 2. Torn Consensus (Confirmed if detected across >= 2 frames, or >= 1 frame for single-shot)
        min_torn_frames = 1 if system == "b1" else 2
        torn_count = sum(1 for e in triggered_frames if (e.get("result") or {}).get("is_torn"))
        final_pred_torn = (torn_count >= min_torn_frames)

        norm_gt = gt_denom.replace("k", "").replace(".0", "").strip()
        norm_final_pred = final_pred_denom.replace("k", "").replace(".0", "").strip()

        denom_match = (norm_final_pred == norm_gt) if (norm_gt and norm_final_pred) else False
        torn_match = (final_pred_torn == gt_torn) if outcome_result else False
        exact_match = denom_match and torn_match
        
        # Find first correct timestamp
        first_correct_ts = None
        for e in triggered_frames:
            r = e.get("result") or {}
            cur_d = str(r.get("denomination", "")).replace("k", "").strip().lower()
            cur_t = bool(r.get("is_torn", False))
            if (cur_d == norm_gt) and (cur_t == gt_torn):
                first_correct_ts = e.get("arrival_ts")
                break
        if first_correct_ts is None and denom_match:
            # If denomination matched, find first frame with correct denom
            for e in triggered_frames:
                r = e.get("result") or {}
                cur_d = str(r.get("denomination", "")).replace("k", "").strip().lower()
                if cur_d == norm_gt:
                    first_correct_ts = e.get("arrival_ts")
                    break

        # Bounding Box IoU
        gt_banknote_box = gt.get("gt_banknote_box")
        pred_denom_box = outcome_result.get("denomination_box") if outcome_result else None
        banknote_iou = compute_box_iou(gt_banknote_box, pred_denom_box) if (gt_banknote_box and pred_denom_box) else 0.0
        banknote_map50 = 100.0 if (denom_match and banknote_iou >= 0.5) else 0.0
        banknote_map50_95 = float(banknote_iou * (banknote_map50 / 100.0) * 100.0)

        # Time-to-correct
        start_ts = lines[0].get("arrival_ts", 0.0)
        time_to_correct_s = (first_correct_ts - start_ts) if (first_correct_ts is not None) else None
        
        # Read energy from raw csv if already computed, else estimate
        # Load from existing raw csv
        raw_df = pd.read_csv(LOGS_DIR / "video_benchmark_raw.csv")
        row_prev = raw_df[(raw_df["session_id"] == sess_id) & (raw_df["system"] == system)]
        if not row_prev.empty:
            prev_dict = row_prev.iloc[0].to_dict()
            energy_joules = prev_dict.get("energy_joules", 0.0)
            avg_cpu_percent = prev_dict.get("avg_cpu_percent", 0.0)
            peak_ram_mb = prev_dict.get("peak_ram_mb", 0.0)
        else:
            energy_joules = 0.0
            avg_cpu_percent = 0.0
            peak_ram_mb = 0.0

        summaries.append({
            "session_id": sess_id,
            "system": system,
            "condition": condition,
            "denomination": gt.get("denomination", ""),
            "total_frames": total_frames,
            "dropped_frames": dropped_count,
            "dropped_rate_pct": (dropped_count / total_frames * 100.0) if total_frames else 0.0,
            "called_full_count": called_full_count,
            "called_full_rate_pct": (called_full_count / total_frames * 100.0) if total_frames else 0.0,
            "avg_light_latency_ms": float(np.mean(light_latencies)) if light_latencies else 0.0,
            "avg_full_latency_ms": float(np.mean(full_latencies)) if full_latencies else 0.0,
            "peak_ram_mb": peak_ram_mb,
            "avg_cpu_percent": avg_cpu_percent,
            "energy_joules": energy_joules,
            "denom_accuracy_pct": 100.0 if denom_match else 0.0,
            "torn_accuracy_pct": 100.0 if torn_match else 0.0,
            "exact_match_pct": 100.0 if exact_match else 0.0,
            "banknote_iou": banknote_iou,
            "banknote_map50": banknote_map50,
            "banknote_map50_95": banknote_map50_95,
            "tear_map50": None,
            "tear_avg_iou": None,
            "time_to_correct_s": time_to_correct_s,
            "log_path": log_path
        })
        
    df_out = pd.DataFrame(summaries)
    df_out.to_csv(LOGS_DIR / "video_benchmark_raw.csv", index=False)
    print(f"Updated video_benchmark_raw.csv with {len(df_out)} runs.")
    return df_out

if __name__ == "__main__":
    recompute_all_sessions()
