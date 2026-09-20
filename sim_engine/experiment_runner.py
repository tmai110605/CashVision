import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
import pandas as pd
import numpy as np

from .frame_source import FrameSource
from .controllers import CascadeController, SingleShotTrigger
from .packages import LightPackage, FullPackage, DetectorOnlyPackage, detect_note_presence_fast
from .energy_meter import CPUEnergyMeter, RAPLMeter


LOGS_DIR = Path("logs_sim")
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def compute_box_iou(box1: Optional[List[float]], box2: Optional[List[float]]) -> float:
    """Computes Intersection over Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
    if not box1 or not box2 or len(box1) < 4 or len(box2) < 4:
        return 0.0
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter_area = inter_w * inter_h
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return float(inter_area / union_area) if union_area > 0 else 0.0


def run_session(
    system: str,
    source: FrameSource,
    session_id: str,
    ground_truth: Dict[str, Any],
    light_pkg: LightPackage,
    full_pkg: FullPackage,
    detector_only_pkg: DetectorOnlyPackage,
    cascade_params: Optional[Dict[str, Any]] = None,
    single_shot_params: Optional[Dict[str, Any]] = None,
    tdp_watts: float = 28.0,
    callback_frame: Optional[Callable[[Dict[str, Any]], None]] = None,
    stop_check_fn: Optional[Callable[[], bool]] = None
) -> Dict[str, Any]:
    """
    Executes a single real-time simulation session on 1 system (b0, b1, b2, or cascade).
    Logs all per-frame metrics and returns aggregated session summary.
    """
    system = system.lower()
    cascade_params = cascade_params or {"quality_threshold": 0.6, "required_stable_frames": 3}
    single_shot_params = single_shot_params or {"min_delay_s": 1.5, "max_delay_s": 3.0}

    cascade = CascadeController(
        quality_threshold=cascade_params.get("quality_threshold", 0.6),
        required_stable_frames=cascade_params.get("required_stable_frames", 3)
    )
    single_shot = SingleShotTrigger(
        min_delay_s=single_shot_params.get("min_delay_s", 1.5),
        max_delay_s=single_shot_params.get("max_delay_s", 3.0)
    )

    energy_meter = CPUEnergyMeter(tdp_watts=tdp_watts)
    rapl_meter = RAPLMeter()

    log_file_path = LOGS_DIR / f"{session_id}_{system}.jsonl"
    frame_logs = []

    session_start_t = time.perf_counter()
    first_note_ts: Optional[float] = None
    first_correct_ts: Optional[float] = None
    last_result: Optional[Dict[str, Any]] = None
    best_result: Optional[Dict[str, Any]] = None

    source.reset()

    with energy_meter, rapl_meter:
        while True:
            if stop_check_fn and stop_check_fn():
                break

            result = source.get_next_frame()
            if result is None:
                break

            frame_bgr, frame_idx, arrival_ts = result
            t_proc_start = time.perf_counter()

            # Measure real-time deadline lag & dropped frames
            lag_ms = (t_proc_start - arrival_ts) * 1000.0
            # Frame is dropped if processing fell behind by more than 2 frame intervals
            is_dropped = lag_ms > (source.frame_interval * 1000.0 * 2.0)

            # 1. System decision logic & Package execution
            full_out = None
            full_latency_ms = 0.0

            if system == "b0":
                # Baseline 0: Full MQTone + YOLO on every frame unconditionally
                has_note = detect_note_presence_fast(frame_bgr)
                quality = 1.0
                light_out = {"latency_ms": 0.0, "quality": 1.0, "has_note": has_note, "pred_class": "n/a"}
                if has_note and first_note_ts is None:
                    first_note_ts = arrival_ts
                full_out = full_pkg.infer(frame_bgr)
                full_latency_ms = full_out["total_latency_ms"]

            elif system in ["b1", "b2"]:
                # Baseline 1: Single-Shot after blind delay once note is seen (Sampled Delay baseline)
                has_note = detect_note_presence_fast(frame_bgr)
                quality = 1.0
                light_out = {"latency_ms": 0.1, "quality": quality, "has_note": has_note, "pred_class": "n/a"}
                if has_note and first_note_ts is None:
                    first_note_ts = arrival_ts

                call_now = single_shot.should_trigger(has_note, arrival_ts)
                if call_now:
                    full_out = full_pkg.infer(frame_bgr)
                    full_latency_ms = full_out["total_latency_ms"]

            elif system == "cascade":
                # Adaptive Cascade (Proposed):
                # LightPackage evaluates quality & note presence every frame (~0.2 - 0.5 ms)
                light_out = light_pkg.infer(frame_bgr, tau=cascade.quality_threshold)
                has_note = light_out["has_note"]
                quality = light_out["quality"]
                if has_note and first_note_ts is None:
                    first_note_ts = arrival_ts

                call_now = cascade.decide(has_note, quality)
                if call_now:
                    full_out = full_pkg.infer(frame_bgr)
                    full_latency_ms = full_out["total_latency_ms"]
                    if full_out.get("denomination_name") is not None:
                        conf = float(full_out.get("denomination_conf", 0.0))
                        cascade.on_detection_result(has_detection=True, conf=conf)
                    else:
                        cascade.on_detection_result(has_detection=False, conf=0.0)
            else:
                light_out = {"latency_ms": 0.0, "quality": 0.0, "has_note": False, "pred_class": "unknown"}
                has_note = False
                quality = 0.0

            if full_out is not None:
                last_result = full_out
                if full_out.get("denomination_name") is not None:
                    if best_result is None or float(full_out.get("denomination_conf", 0.0)) > float(best_result.get("denomination_conf", 0.0)):
                        best_result = full_out

                # Check correctness vs ground truth
                gt_denom = str(ground_truth.get("denomination", "")).strip().lower()
                gt_torn = bool(ground_truth.get("is_torn", False))

                pred_denom = str(full_out.get("denomination_name", "")).strip().lower()
                pred_torn = bool(full_out.get("is_torn", False))

                norm_gt = gt_denom.replace("k", "").replace(".0", "").strip()
                norm_pred = pred_denom.replace("k", "").replace(".0", "").strip()

                denom_correct = (norm_pred == norm_gt) if (norm_gt and norm_pred) else False
                torn_correct = (pred_torn == gt_torn)
                is_correct = denom_correct and torn_correct

                if is_correct and first_correct_ts is None:
                    first_correct_ts = arrival_ts

            t_proc_end = time.perf_counter()
            total_step_ms = (t_proc_end - t_proc_start) * 1000.0
            energy_meter.record_step()

            # Record frame log entry
            log_entry = {
                "session_id": session_id,
                "system": system,
                "frame_index": frame_idx,
                "arrival_ts": float(arrival_ts),
                "lag_ms": float(lag_ms),
                "dropped": bool(is_dropped),
                "light_latency_ms": float(light_out["latency_ms"]),
                "full_latency_ms": float(full_latency_ms),
                "total_step_ms": float(total_step_ms),
                "called_full": bool(full_out is not None),
                "quality": float(quality),
                "has_note": bool(has_note),
                "pred_class": light_out["pred_class"],
                "stable_count": cascade.stable_count if system == "cascade" else 0,
                "result": {
                    "denomination": full_out.get("denomination_name") if full_out else None,
                    "confidence": full_out.get("denomination_conf") if full_out else None,
                    "is_torn": full_out.get("is_torn") if full_out else None,
                    "speech_text": full_out.get("speech_text") if full_out else None
                } if full_out else None,
                "ground_truth": ground_truth
            }
            frame_logs.append(log_entry)

            if callback_frame:
                callback_frame({
                    "frame_bgr": frame_bgr,
                    "log_entry": log_entry,
                    "light_out": light_out,
                    "full_out": full_out,
                    "system": system,
                    "corrected_image": full_out.get("corrected_image") if full_out else None
                })

    # Save session JSONL
    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            for entry in frame_logs:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Error writing log file {log_file_path}: {e}")

    # Compute Session Summary
    total_frames = len(frame_logs)
    dropped_count = sum(1 for e in frame_logs if e["dropped"])
    called_full_count = sum(1 for e in frame_logs if e["called_full"])
    full_latencies = [e["full_latency_ms"] for e in frame_logs if e["called_full"]]
    light_latencies = [e["light_latency_ms"] for e in frame_logs]

    # Accuracy & Bounding Box mAP check on session outcome via Temporal Consensus
    gt_denom = str(ground_truth.get("denomination", "")).strip().lower()
    gt_torn = bool(ground_truth.get("is_torn", False))

    # Gather all triggered frames that produced detections
    triggered_frames = [e for e in frame_logs if e.get("called_full") and e.get("result")]
    
    # 1. Denomination Consensus (Confidence-weighted voting across triggered detections)
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
        outcome_result = best_result if best_result is not None else last_result
        final_pred_denom = str(outcome_result.get("denomination_name", "")).strip().lower() if outcome_result else ""

    # 2. Torn Consensus (Confirmed if detected across >= 2 frames, or >= 1 frame for single-shot)
    min_torn_frames = 1 if system == "b1" else 2
    torn_count = sum(1 for e in triggered_frames if (e.get("result") or {}).get("is_torn"))
    final_pred_torn = (torn_count >= min_torn_frames)

    norm_gt = gt_denom.replace("k", "").replace(".0", "").strip()
    norm_final_pred = final_pred_denom.replace("k", "").replace(".0", "").strip()

    denom_match = (norm_final_pred == norm_gt) if (norm_gt and norm_final_pred) else False
    torn_match = (final_pred_torn == gt_torn) if outcome_result else False
    exact_match = denom_match and torn_match

    # Bounding Box IoU & mAP50 Evaluation
    gt_banknote_box = ground_truth.get("gt_banknote_box")
    gt_tear_boxes = ground_truth.get("gt_tear_boxes", []) or []

    pred_denom_box = outcome_result.get("denomination_box") if outcome_result else None
    pred_tears = outcome_result.get("tear_detections", []) if outcome_result else []

    banknote_iou = compute_box_iou(gt_banknote_box, pred_denom_box) if (gt_banknote_box and pred_denom_box) else 0.0
    banknote_map50 = 100.0 if (denom_match and banknote_iou >= 0.5) else 0.0
    banknote_map50_95 = float(banknote_iou * (banknote_map50 / 100.0) * 100.0)

    # Tear mAP50 Evaluation
    has_real_tears = len(gt_tear_boxes) > 0
    if has_real_tears:
        matched_tears = 0
        tear_ious = []
        for gt_tbox in gt_tear_boxes:
            best_tiou = 0.0
            for pred_t in pred_tears:
                tiou = compute_box_iou(gt_tbox, pred_t.get("box"))
                if tiou > best_tiou:
                    best_tiou = tiou
            tear_ious.append(best_tiou)
            if best_tiou >= 0.5:
                matched_tears += 1
        tear_map50 = (matched_tears / len(gt_tear_boxes)) * 100.0
        tear_avg_iou = float(np.mean(tear_ious)) * 100.0 if tear_ious else 0.0
    else:
        tear_map50 = None
        tear_avg_iou = None

    # Time to correct result
    time_to_correct_s = None
    if first_note_ts is not None and first_correct_ts is not None:
        time_to_correct_s = max(0.0, first_correct_ts - first_note_ts)

    energy_j = rapl_meter.energy_joules if rapl_meter.available else energy_meter.energy_joules

    summary = {
        "session_id": session_id,
        "system": system,
        "condition": ground_truth.get("condition", "unknown"),
        "denomination": ground_truth.get("denomination", "unknown"),
        "total_frames": total_frames,
        "dropped_frames": dropped_count,
        "dropped_rate_pct": (dropped_count / total_frames * 100.0) if total_frames > 0 else 0.0,
        "called_full_count": called_full_count,
        "called_full_rate_pct": (called_full_count / total_frames * 100.0) if total_frames > 0 else 0.0,
        "avg_light_latency_ms": float(np.mean(light_latencies)) if light_latencies else 0.0,
        "avg_full_latency_ms": float(np.mean(full_latencies)) if full_latencies else 0.0,
        "peak_ram_mb": energy_meter.peak_ram_mb,
        "avg_cpu_percent": energy_meter.avg_cpu_percent,
        "energy_joules": energy_j,
        "denom_accuracy_pct": 100.0 if denom_match else 0.0,
        "torn_accuracy_pct": 100.0 if torn_match else 0.0,
        "exact_match_pct": 100.0 if exact_match else 0.0,
        "banknote_iou": round(float(banknote_iou * 100.0), 2),
        "banknote_map50": round(float(banknote_map50), 2),
        "banknote_map50_95": round(float(banknote_map50_95), 2),
        "tear_map50": round(float(tear_map50), 2) if tear_map50 is not None else None,
        "tear_avg_iou": round(float(tear_avg_iou), 2) if tear_avg_iou is not None else None,
        "time_to_correct_s": time_to_correct_s,
        "log_path": str(log_file_path)
    }

    return summary


def run_matrix_benchmark(
    sessions_list: List[Dict[str, Any]],
    light_pkg: LightPackage,
    full_pkg: FullPackage,
    detector_only_pkg: DetectorOnlyPackage,
    systems: Optional[List[str]] = None,
    cascade_params: Optional[Dict[str, Any]] = None,
    tdp_watts: float = 28.0,
    cooldown_seconds: float = 2.0,
    progress_callback: Optional[Callable[[int, int, str], None]] = None
) -> pd.DataFrame:
    """
    Runs full 4-system matrix across all defined test sessions.
    Returns comprehensive DataFrame of summary results.
    """
    systems = systems or ["b0", "b1", "cascade"]
    all_summaries = []
    total_runs = len(sessions_list) * len(systems)
    curr_run = 0

    for sess in sessions_list:
        source_obj = sess["source"]
        sess_id = sess["session_id"]
        gt = sess["ground_truth"]

        for sys_name in systems:
            curr_run += 1
            if progress_callback:
                progress_callback(curr_run, total_runs, f"Running {sys_name.upper()} on {sess_id}")

            summ = run_session(
                system=sys_name,
                source=source_obj,
                session_id=sess_id,
                ground_truth=gt,
                light_pkg=light_pkg,
                full_pkg=full_pkg,
                detector_only_pkg=detector_only_pkg,
                cascade_params=cascade_params,
                tdp_watts=tdp_watts
            )
            all_summaries.append(summ)

            if cooldown_seconds > 0:
                time.sleep(cooldown_seconds)

    summary_df = pd.DataFrame(all_summaries)
    summary_csv_path = LOGS_DIR / "benchmark_matrix_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    return summary_df
