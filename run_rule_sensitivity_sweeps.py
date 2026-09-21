"""
run_rule_sensitivity_sweeps.py
==============================
Executes the empirical One-at-a-Time (OAT) parameter sweeps across the 36 continuous
real-world video streams in `video_test/` as specified in Section 3.5 of the CashVision
ESWA manuscript.

Swept Parameters (5 Expert Decision Thresholds):
1. Photometric Acceptance Threshold tau in [0.40, 0.80], step 0.05 (9 points)
2. Temporal Buffer Stability Window K_opt in {1, 2, 3, 4, 5} frames (5 points)
3. Verification Burst Budget M_verify in {1, 2, 3, 4} attempts (4 points)
4. Spatial Texture Filter Threshold theta_texture in [5.0, 30.0], step 5.0 (6 points)
5. Post-NMS Detection Confidence Threshold theta_conf in [0.15, 0.50], step 0.05 (8 points)
Total configurations: 32 points (28 unique).

Metrics computed across 36 video streams for each configuration:
- Denomination Accuracy (Acc_denom, %)
- Exact-Match Accuracy (Acc_exact, %)
- Active Trigger Rate (P_active, %)
- Host CPU Energy Work Proxy (E_session, J)
- Interaction Time-to-Confirmation (TTC, s)
- Financial Valuation Hazard Rate (Hazard, %)

Outputs:
- logs_sim/rule_sensitivity_results.csv
- logs_sim/table6_rule_sensitivity_populated.tex
- figures/rule_sensitivity_pareto.pdf
- figures/rule_sensitivity_pareto.png
"""

import os
import sys
import time
import re
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from quality_gate import QualityGate
from mqtone import MQTone
from train_c2 import C2DetectionPipeline, DENOM_CLASSES, TORN_CLASS_ID
from sim_engine.controllers import CascadeController, CascadeState

VIDEO_DIR = PROJECT_ROOT / "video_test"
OUTPUT_DIR = PROJECT_ROOT / "logs_sim"
FIGURES_DIR = PROJECT_ROOT / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

CACHE_FILE = OUTPUT_DIR / "rule_sensitivity_frame_cache.pkl"
RESULTS_CSV = OUTPUT_DIR / "rule_sensitivity_results.csv"
TABLE_TEX = OUTPUT_DIR / "table6_rule_sensitivity_populated.tex"
PARETO_PDF = FIGURES_DIR / "rule_sensitivity_pareto.pdf"
PARETO_PNG = FIGURES_DIR / "rule_sensitivity_pareto.png"


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
    return sessions


def build_or_load_frame_cache(
    sessions: List[Dict[str, Any]],
    weights_qg: str = "checkpointyolov8n/quality_gate_weights.pt",
    weights_full: str = "checkpointyolov8n/full_pipeline_weights.pt",
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Extracts and caches per-frame neural features across all 36 continuous videos.
    Runs QualityGate on thumbnails and MQTone+YOLOv8n with conf=0.10 on candidate frames.
    """
    if CACHE_FILE.exists():
        print(f"📦 Loading precomputed frame cache from {CACHE_FILE}...", flush=True)
        try:
            with open(CACHE_FILE, "rb") as f:
                cache = pickle.load(f)
            if len(cache) == len(sessions):
                print(f"✅ Loaded frame cache for all {len(cache)} sessions.", flush=True)
                return cache
        except Exception as e:
            print(f"⚠️ Cache loading failed ({e}). Rebuilding cache...", flush=True)

    print(f"🔄 Building frame cache across {len(sessions)} videos on {device.upper()}...", flush=True)
    t0_cache = time.time()

    # Load QualityGate
    qg_model = QualityGate(thumbnail_size=64).to(device)
    qg_model.eval()
    p_qg = Path(weights_qg)
    if p_qg.exists():
        qg_model.load_state_dict(torch.load(str(p_qg), map_location=device))
        print("✅ QualityGate weights loaded.")
    else:
        print(f"⚠️ QualityGate weights not found at {weights_qg}!")

    # Load Full Pipeline (MQTone + YOLO)
    full_pipeline = C2DetectionPipeline(weights_path="yolov8n.pt", correction_method='mqtone').to(device)
    full_pipeline.eval()
    p_full = Path(weights_full)
    if p_full.exists():
        full_pipeline.load_state_dict(torch.load(str(p_full), map_location=device))
        print("✅ FullPipeline weights loaded.")
    else:
        print(f"⚠️ FullPipeline weights not found at {weights_full}!")

    cache = {}

    for s_idx, s in enumerate(sessions, 1):
        v_path = s["video_path"]
        sess_id = s["session_id"]
        cap = cv2.VideoCapture(v_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_interval = 1.0 / fps

        frames_data = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Texture measure
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            std_dev = float(np.std(gray))

            # 2. Quality-Gate inference on 64x64 thumbnail
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            thumb = cv2.resize(rgb, (64, 64))
            thumb_tensor = torch.from_numpy(thumb).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

            with torch.no_grad():
                logits = qg_model(thumb_tensor)
                probs = torch.softmax(logits, dim=-1)[0]
                conf_val, pred_cls = torch.max(probs, dim=-1)
                prob_good = float(probs[0].item())
                pred_cls_idx = int(pred_cls.item())
                conf_val = float(conf_val.item())

            # Baseline quality score
            is_passed_default = not ((pred_cls_idx != 0) and (conf_val > 0.60))
            quality_score = prob_good if is_passed_default else (1.0 - conf_val)

            # 3. Candidate full detection check
            # We run full pipeline on frames that could be triggered in any parameter sweep:
            # i.e. std_dev >= 4.0 and prob_good >= 0.25 (or conf_val < 0.85)
            yolo_dets = []
            full_lat_ms = 0.0

            if std_dev >= 4.0:
                t_f0 = time.perf_counter()
                resized_640 = cv2.resize(rgb, (640, 640))
                img_t = torch.from_numpy(resized_640).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0

                with torch.no_grad():
                    if full_pipeline.use_correction and full_pipeline.corrector is not None:
                        corr_t = full_pipeline.corrector(img_t)
                    else:
                        corr_t = img_t
                    res = full_pipeline.yolo.predict(corr_t, conf=0.10, device=device, verbose=False)[0]

                t_f1 = time.perf_counter()
                full_lat_ms = (t_f1 - t_f0) * 1000.0

                for b in res.boxes:
                    cid = int(b.cls.item())
                    c_val = float(b.conf.item())
                    xyxy = b.xyxy[0].tolist()
                    yolo_dets.append({
                        "cls_id": cid,
                        "conf": c_val,
                        "box": xyxy,
                        "is_denom": (cid in DENOM_CLASSES),
                        "is_torn": (cid == TORN_CLASS_ID)
                    })

            frames_data.append({
                "frame_idx": frame_idx,
                "arrival_ts": frame_idx * frame_interval,
                "std_dev": std_dev,
                "prob_good": prob_good,
                "pred_cls_idx": pred_cls_idx,
                "qg_conf": conf_val,
                "quality_score": quality_score,
                "detections": yolo_dets,
                "full_lat_ms": full_lat_ms
            })
            frame_idx += 1

        cap.release()
        cache[sess_id] = {
            "session_info": s,
            "total_frames": len(frames_data),
            "frame_interval": frame_interval,
            "frames": frames_data
        }
        print(f"  [{s_idx:02d}/36] Cached {s['video_name']}: {len(frames_data)} frames", flush=True)

    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)
    print(f"✅ Frame cache built and saved in {(time.time() - t0_cache) / 60:.2f} min.\n", flush=True)
    return cache


def evaluate_configuration(
    cache: Dict[str, Any],
    tau: float = 0.60,
    k_opt: int = 3,
    m_verify: int = 2,
    theta_texture: float = 15.0,
    theta_conf: float = 0.25,
    tdp_watts: float = 28.0,
    light_lat_ms: float = 1.70,
    base_full_lat_ms: float = 107.79
) -> Dict[str, float]:
    """
    Simulates the CascadeController decision layer on the cached video frames.
    """
    total_sessions = len(cache)
    denom_correct_count = 0
    exact_correct_count = 0
    hazard_count = 0
    total_trigger_rate_pct = []
    total_session_energy_j = []
    valid_ttc_list = []

    for sess_id, s_data in cache.items():
        s_info = s_data["session_info"]
        frames = s_data["frames"]
        gt_denom = s_info["denomination"].replace("k", "").strip().lower()
        gt_torn = s_info["is_torn"]

        cascade = CascadeController(
            quality_threshold=tau,
            required_stable_frames=k_opt,
            max_verify_frames=m_verify
        )

        called_full_count = 0
        triggered_results = []
        first_note_ts = None
        first_correct_ts = None

        for f in frames:
            arrival_ts = f["arrival_ts"]
            has_note = (f["std_dev"] > theta_texture)

            if has_note and first_note_ts is None:
                first_note_ts = arrival_ts

            # Optical quality score under tau
            pred_cls_idx = f["pred_cls_idx"]
            qg_conf = f["qg_conf"]
            prob_good = f["prob_good"]
            is_blocked = (pred_cls_idx != 0) and (qg_conf > tau)
            quality = prob_good if not is_blocked else (1.0 - qg_conf)

            call_now = cascade.decide(has_note, quality)
            if call_now:
                called_full_count += 1
                # Filter detections at theta_conf
                active_dets = [d for d in f["detections"] if d["conf"] >= theta_conf]
                denom_dets = [d for d in active_dets if d["is_denom"]]
                has_torn = any(d["is_torn"] for d in active_dets)

                if denom_dets:
                    best_d = max(denom_dets, key=lambda x: x["conf"])
                    cid = best_d["cls_id"]
                    cname = DENOM_CLASSES.get(cid, "unknown")
                    cascade.on_detection_result(has_detection=True, conf=best_d["conf"])
                    triggered_results.append({
                        "denom": str(cname).replace("k", "").strip().lower(),
                        "conf": best_d["conf"],
                        "is_torn": has_torn,
                        "ts": arrival_ts
                    })
                    if str(cname).replace("k", "").strip().lower() == gt_denom and (has_torn == gt_torn):
                        if first_correct_ts is None:
                            first_correct_ts = arrival_ts
                else:
                    cascade.on_detection_result(has_detection=False, conf=0.0)

        # Session Consensus
        final_denom = None
        final_torn = False
        if triggered_results:
            scores = {}
            for tr in triggered_results:
                d = tr["denom"]
                scores[d] = scores.get(d, 0.0) + tr["conf"]
            final_denom = max(scores.items(), key=lambda x: x[1])[0]
            torn_hits = sum(1 for tr in triggered_results if tr["is_torn"])
            final_torn = (torn_hits >= 2)

        # Outcome assessment
        denom_match = (final_denom == gt_denom) if final_denom else False
        torn_match = (final_torn == gt_torn) if final_denom else False
        exact_match = denom_match and torn_match

        if denom_match:
            denom_correct_count += 1
        if exact_match:
            exact_correct_count += 1

        # Valuation Hazard: confirmed denomination that is WRONG
        if final_denom is not None and (final_denom != gt_denom):
            hazard_count += 1

        # Trigger rate
        p_active = (called_full_count / len(frames) * 100.0) if frames else 0.0
        total_trigger_rate_pct.append(p_active)

        # Host CPU Energy Proxy
        # E = (Light_Time + Triggered * Full_Time) * TDP_Watts * (CPU_Util / 100)
        t_light_sec = len(frames) * (light_lat_ms / 1000.0)
        t_full_sec = called_full_count * (base_full_lat_ms / 1000.0)
        t_total_work = t_light_sec + t_full_sec
        # Average session active energy based on measured 28W host CPU baseline
        e_session = t_total_work * tdp_watts * 0.72  # 72% average multi-core CPU load
        total_session_energy_j.append(e_session)

        # Time to confirmation
        if first_note_ts is not None and first_correct_ts is not None:
            ttc = max(0.5, first_correct_ts - first_note_ts)
            valid_ttc_list.append(ttc)
        elif triggered_results:
            ttc = max(0.5, triggered_results[0]["ts"] - (first_note_ts or 0.0))
            valid_ttc_list.append(ttc)
        else:
            valid_ttc_list.append(len(frames) * s_data["frame_interval"])

    denom_acc_pct = (denom_correct_count / total_sessions) * 100.0
    exact_acc_pct = (exact_correct_count / total_sessions) * 100.0
    hazard_rate_pct = (hazard_count / total_sessions) * 100.0
    mean_trigger_rate = float(np.mean(total_trigger_rate_pct))
    mean_energy_j = float(np.mean(total_session_energy_j))
    mean_ttc_s = float(np.mean(valid_ttc_list)) if valid_ttc_list else 6.52

    return {
        "denom_acc": denom_acc_pct,
        "exact_acc": exact_acc_pct,
        "trigger_rate": mean_trigger_rate,
        "energy_j": mean_energy_j,
        "ttc_s": mean_ttc_s,
        "hazard_rate": hazard_rate_pct
    }


def run_all_sweeps(cache: Dict[str, Any]) -> pd.DataFrame:
    """Executes all 5 OAT parameter sweeps across 32 configuration points."""
    print("=" * 80, flush=True)
    print("🚀 EXECUTING 5-AXIS ONE-AT-A-TIME PARAMETER SWEEPS (32 CONFIGURATIONS)", flush=True)
    print("=" * 80, flush=True)

    rows = []

    # Baseline Calibrated Defaults
    DEF_TAU = 0.60
    DEF_K = 3
    DEF_M = 2
    DEF_THETA_TEX = 15.0
    DEF_THETA_CONF = 0.25

    # Sweep 1: Photometric Acceptance Threshold tau in [0.40, 0.80], step 0.05
    tau_vals = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    for tau in tau_vals:
        res = evaluate_configuration(cache, tau=tau, k_opt=DEF_K, m_verify=DEF_M, theta_texture=DEF_THETA_TEX, theta_conf=DEF_THETA_CONF)
        label = f"{tau:.2f} (calibrated)" if abs(tau - DEF_TAU) < 1e-4 else f"{tau:.2f}"
        rows.append({
            "Sweep": "Sweep 1: Photometric Acceptance Threshold tau",
            "Parameter": "$\\tau$",
            "Swept_Value": label,
            "raw_val": tau,
            **res
        })

    # Sweep 2: Temporal Buffer Stability Window K_opt in {1, 2, 3, 4, 5}
    k_vals = [1, 2, 3, 4, 5]
    for k in k_vals:
        res = evaluate_configuration(cache, tau=DEF_TAU, k_opt=k, m_verify=DEF_M, theta_texture=DEF_THETA_TEX, theta_conf=DEF_THETA_CONF)
        label = f"{k} frames (calibrated)" if k == DEF_K else f"{k} frames"
        rows.append({
            "Sweep": "Sweep 2: Stability Window K_opt",
            "Parameter": "$K_{\\text{opt}}$",
            "Swept_Value": label,
            "raw_val": k,
            **res
        })

    # Sweep 3: Verification Burst Budget M_verify in {1, 2, 3, 4}
    m_vals = [1, 2, 3, 4]
    for m in m_vals:
        res = evaluate_configuration(cache, tau=DEF_TAU, k_opt=DEF_K, m_verify=m, theta_texture=DEF_THETA_TEX, theta_conf=DEF_THETA_CONF)
        label = f"{m} attempts (calibrated)" if m == DEF_M else f"{m} attempts"
        rows.append({
            "Sweep": "Sweep 3: Verification Burst Budget M_verify",
            "Parameter": "$M_{\\text{verify}}$",
            "Swept_Value": label,
            "raw_val": m,
            **res
        })

    # Sweep 4: Spatial Texture Filter Threshold theta_texture in [5.0, 30.0], step 5.0
    tex_vals = [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    for tex in tex_vals:
        res = evaluate_configuration(cache, tau=DEF_TAU, k_opt=DEF_K, m_verify=DEF_M, theta_texture=tex, theta_conf=DEF_THETA_CONF)
        label = f"{tex:.1f} (calibrated)" if abs(tex - DEF_THETA_TEX) < 1e-4 else f"{tex:.1f}"
        rows.append({
            "Sweep": "Sweep 4: Spatial Texture Threshold theta_texture",
            "Parameter": "$\\theta_{\\text{texture}}$",
            "Swept_Value": label,
            "raw_val": tex,
            **res
        })

    # Sweep 5: Detection Confidence Threshold theta_conf in [0.15, 0.50], step 0.05
    conf_vals = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    for conf in conf_vals:
        res = evaluate_configuration(cache, tau=DEF_TAU, k_opt=DEF_K, m_verify=DEF_M, theta_texture=DEF_THETA_TEX, theta_conf=conf)
        label = f"{conf:.2f} (calibrated)" if abs(conf - DEF_THETA_CONF) < 1e-4 else f"{conf:.2f}"
        rows.append({
            "Sweep": "Sweep 5: Detection Confidence Threshold theta_conf",
            "Parameter": "$\\theta_{\\text{conf}}$",
            "Swept_Value": label,
            "raw_val": conf,
            **res
        })

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"✅ Sweep results saved to {RESULTS_CSV}", flush=True)
    return df


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generates the populated LaTeX code for Table 6."""
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Rule Base Sensitivity Analysis: Empirical One-at-a-Time (OAT) Sweeps of Decision Thresholds across Accuracy, Energy, Latency, and Assistive Safety Metrics on the 36 Continuous Video Benchmark Streams.}")
    latex.append(r"\label{tab:rule_sensitivity}")
    latex.append(r"\footnotesize")
    latex.append(r"\renewcommand{\arraystretch}{1.15}")
    latex.append(r"\begin{tabular}{llcccccc}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Parameter} & \textbf{Swept Value} & \textbf{Denom Acc (\%)} & \textbf{Exact Acc (\%)} & \textbf{Trigger Rate $P_{\text{active}}$ (\%)} & \textbf{Energy Proxy $E_{\text{session}}$ (J)} & \textbf{Est.\ TTC (s)} & \textbf{Hazard Rate (\%)} \\")
    latex.append(r"\midrule")

    sweeps = df["Sweep"].unique()
    for s_idx, sw in enumerate(sweeps, 1):
        sub = df[df["Sweep"] == sw]
        sub_title = sw.replace("_", r"\_")
        latex.append(f"\\multicolumn{{8}}{{l}}{{\\textit{{{sub_title}}}}} \\\\")
        for _, r in sub.iterrows():
            param = r["Parameter"]
            val_str = r["Swept_Value"]
            if "(calibrated)" in val_str:
                val_fmt = val_str.replace("(calibrated)", r"\textit{(calibrated)}")
            else:
                val_fmt = val_str
            denom = f"{r['denom_acc']:.1f}"
            exact = f"{r['exact_acc']:.1f}"
            trig = f"{r['trigger_rate']:.1f}"
            energy = f"{r['energy_j']:.1f}"
            ttc = f"{r['ttc_s']:.2f}"
            hazard = f"{r['hazard_rate']:.1f}"
            latex.append(f"{param} & {val_fmt} & {denom} & {exact} & {trig} & {energy} & {ttc} & {hazard} \\\\")
        if s_idx < len(sweeps):
            latex.append(r"\midrule")

    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")

    tex_str = "\n".join(latex)
    with open(TABLE_TEX, "w", encoding="utf-8") as f:
        f.write(tex_str)
    print(f"✅ Populated LaTeX table exported to {TABLE_TEX}", flush=True)
    return tex_str


def plot_pareto_curves(df: pd.DataFrame):
    """Plots multi-panel Accuracy-Energy-Latency Pareto curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"

    colors = {
        "Sweep 1: Photometric Acceptance Threshold tau": "#1f77b4",
        "Sweep 2: Stability Window K_opt": "#2ca02c",
        "Sweep 3: Verification Burst Budget M_verify": "#d62728",
        "Sweep 4: Spatial Texture Threshold theta_texture": "#9467bd",
        "Sweep 5: Detection Confidence Threshold theta_conf": "#ff7f0e"
    }

    # Panel 1: Accuracy vs Energy
    ax1 = axes[0]
    for sw, sub in df.groupby("Sweep"):
        lbl = sw.split(":")[1].strip().replace("theta_", "θ_").replace("tau", "τ").replace("K_opt", "K")
        ax1.plot(sub["energy_j"], sub["exact_acc"], marker="o", label=lbl, color=colors.get(sw, "black"), alpha=0.85)
        # Highlight calibrated point
        cal = sub[sub["Swept_Value"].str.contains("calibrated")]
        if not cal.empty:
            ax1.scatter(cal["energy_j"], cal["exact_acc"], color="gold", edgecolors="black", s=120, zorder=5)

    ax1.set_xlabel("Host CPU Energy Work Proxy $E_{session}$ (J)", fontsize=11)
    ax1.set_ylabel("Exact-Match Accuracy (%)", fontsize=11)
    ax1.set_title("(a) Accuracy vs. Computational Energy", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=8, loc="lower right")

    # Panel 2: Accuracy vs Latency (TTC)
    ax2 = axes[1]
    for sw, sub in df.groupby("Sweep"):
        lbl = sw.split(":")[1].strip().replace("theta_", "θ_").replace("tau", "τ").replace("K_opt", "K")
        ax2.plot(sub["ttc_s"], sub["exact_acc"], marker="s", label=lbl, color=colors.get(sw, "black"), alpha=0.85)
        cal = sub[sub["Swept_Value"].str.contains("calibrated")]
        if not cal.empty:
            ax2.scatter(cal["ttc_s"], cal["exact_acc"], color="gold", edgecolors="black", s=120, zorder=5)

    ax2.set_xlabel("Time-to-Confirmation (TTC, s)", fontsize=11)
    ax2.set_ylabel("Exact-Match Accuracy (%)", fontsize=11)
    ax2.set_title("(b) Accuracy vs. Time-to-Confirmation", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=8, loc="lower right")

    # Panel 3: Financial Valuation Hazard Rate vs Energy
    ax3 = axes[2]
    for sw, sub in df.groupby("Sweep"):
        lbl = sw.split(":")[1].strip().replace("theta_", "θ_").replace("tau", "τ").replace("K_opt", "K")
        ax3.plot(sub["energy_j"], sub["hazard_rate"], marker="^", label=lbl, color=colors.get(sw, "black"), alpha=0.85)
        cal = sub[sub["Swept_Value"].str.contains("calibrated")]
        if not cal.empty:
            ax3.scatter(cal["energy_j"], cal["hazard_rate"], color="gold", edgecolors="black", s=120, zorder=5)

    ax3.set_xlabel("Host CPU Energy Work Proxy $E_{session}$ (J)", fontsize=11)
    ax3.set_ylabel("Financial Valuation Hazard Rate (%)", fontsize=11)
    ax3.set_title("(c) Assistive Safety Hazard vs. Energy", fontsize=12, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.5)
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(PARETO_PDF, dpi=300)
    plt.savefig(PARETO_PNG, dpi=300)
    plt.close()
    print(f"✅ Pareto trade-off figure saved to {PARETO_PDF} and {PARETO_PNG}", flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Rule Base Sensitivity Sweeps")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of video sessions for testing (e.g. 2)")
    parser.add_argument("--recompute", action="store_true", help="Force recomputation of frame cache")
    args = parser.parse_args()

    if args.recompute and CACHE_FILE.exists():
        CACHE_FILE.unlink()
        print(f"🗑️ Cleared existing cache at {CACHE_FILE}")

    sessions = parse_video_sessions(VIDEO_DIR)
    if not sessions:
        print(f"❌ Error: No video sessions found in {VIDEO_DIR}!")
        return

    if args.limit:
        print(f"⚠️ [TEST MODE] Limiting run to first {args.limit} sessions.", flush=True)
        sessions = sessions[:args.limit]

    print(f"🎬 Processing {len(sessions)} test videos.")
    cache = build_or_load_frame_cache(sessions)
    df = run_all_sweeps(cache)
    generate_latex_table(df)
    plot_pareto_curves(df)
    print("\n🎉 RULE SENSITIVITY PARAMETER SWEEP COMPLETE!", flush=True)


if __name__ == "__main__":
    main()
