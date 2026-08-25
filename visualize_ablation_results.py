#!/usr/bin/env python3
"""
visualize_ablation_results.py — Trực quan hoá kết quả Ablation 2x2 (Config A/B/C/D)
của CashVision C2 Pipeline.

Đọc results_c2/config_A/metrics_per_condition.csv (và B, C, D nếu có), xuất ra:
  - Heatmap Accuracy Denom theo Condition x Config
  - Heatmap mAP50 Tear theo Condition x Config (chỉ áp dụng torn_clean/torn_bright)
  - Heatmap False Alarm Rate theo Condition x Config
  - Bar chart Overall (weighted theo num_samples) cho từng metric, so sánh 4 config
  - Gộp tất cả vào 1 file HTML duy nhất để mở bằng trình duyệt

Cách chạy (từ thư mục ~/CashVision, sau khi đã chạy xong run_c2.py --stage ablation):
    python visualize_ablation_results.py --results_dir results_c2

Không cần GPU / không cần load lại model — chỉ đọc CSV kết quả đã có sẵn.
"""

import argparse
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # không cần display, chỉ xuất ảnh
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CONFIG_COLORS = {
    'A': '#e74c3c',  # đỏ - baseline
    'B': '#3498db',  # xanh dương - IC-Net
    'C': '#2ecc71',  # xanh lá - Consistency Loss
    'D': '#9b59b6',  # tím - cả hai
}
CONFIG_LABELS = {
    'A': 'A: Baseline',
    'B': 'B: +IC-Net',
    'C': 'C: +Consistency',
    'D': 'D: +IC-Net+Consistency',
}
CONDITION_ORDER = ['indoor', 'outdoor', 'backlight', 'overexposed', 'torn_clean', 'torn_bright']


def load_configs(results_dir: Path) -> dict:
    data = {}
    for cfg in ['A', 'B', 'C', 'D']:
        csv_path = results_dir / f"config_{cfg}" / "metrics_per_condition.csv"
        if csv_path.exists():
            data[cfg] = pd.read_csv(csv_path)
        else:
            print(f"  (bỏ qua) Không tìm thấy: {csv_path}")
    return data


def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return encoded


def plot_heatmap(pivot_df: pd.DataFrame, title: str, cmap="RdYlGn", vmin=0, vmax=100):
    n_cols = max(len(pivot_df.columns), 1)
    n_rows = max(len(pivot_df.index), 1)
    fig, ax = plt.subplots(figsize=(1.7 * n_cols + 2.2, 0.9 * n_rows + 2))
    data = pivot_df.values.astype(float)
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels([CONFIG_LABELS.get(c, c) for c in pivot_df.columns], rotation=15, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index, fontsize=10)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            text = "—" if np.isnan(val) else f"{val:.1f}"
            ax.text(j, i, text, ha="center", va="center", color="black", fontsize=11, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig


def plot_bar(overall_df: pd.DataFrame, value_col: str, title: str, ylabel: str):
    plot_df = overall_df.dropna(subset=[value_col])
    fig, ax = plt.subplots(figsize=(6, 4.5))
    configs = plot_df["config"].tolist()
    values = plot_df[value_col].tolist()
    colors = [CONFIG_COLORS.get(c, "#888") for c in configs]
    bars = ax.bar([CONFIG_LABELS.get(c, c) for c in configs], values, color=colors)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel(ylabel)
    if values:
        ax.set_ylim(0, max(values) * 1.2)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right")
    fig.tight_layout()
    return fig


def compute_overall(data: dict) -> pd.DataFrame:
    rows = []
    for cfg, df in data.items():
        total_n = df["num_samples"].sum()
        weighted_acc = (df["accuracy_denom"] * df["num_samples"]).sum() / total_n if total_n > 0 else np.nan
        weighted_fa = (df["false_alarm_rate"] * df["num_samples"]).sum() / total_n if total_n > 0 else np.nan

        tear_rows = df[df["mAP50_tear"].notna()]
        tear_n = tear_rows["num_samples"].sum()
        weighted_tear_map = (
            (tear_rows["mAP50_tear"] * tear_rows["num_samples"]).sum() / tear_n if tear_n > 0 else np.nan
        )

        rows.append({
            "config": cfg,
            "total_samples": int(total_n),
            "overall_accuracy_denom": round(weighted_acc, 2) if not np.isnan(weighted_acc) else np.nan,
            "overall_false_alarm_rate": round(weighted_fa, 2) if not np.isnan(weighted_fa) else np.nan,
            "overall_mAP50_tear": round(weighted_tear_map, 2) if not np.isnan(weighted_tear_map) else np.nan,
        })
    return pd.DataFrame(rows).sort_values("config").reset_index(drop=True)


def build_pivot(data: dict, value_col: str, only_notna: bool = False) -> pd.DataFrame:
    rows = []
    for cfg, df in data.items():
        for _, r in df.iterrows():
            if only_notna and pd.isna(r[value_col]):
                continue
            rows.append({"condition": r["condition"], "config": cfg, value_col: r[value_col]})
    if not rows:
        return pd.DataFrame()
    long_df = pd.DataFrame(rows)
    pivot = long_df.pivot(index="condition", columns="config", values=value_col)
    ordered_idx = [c for c in CONDITION_ORDER if c in pivot.index]
    return pivot.reindex(ordered_idx)


def main():
    parser = argparse.ArgumentParser(description="Trực quan hoá kết quả Ablation 2x2 (CashVision C2)")
    parser.add_argument("--results_dir", type=str, default="results_c2", help="Thư mục chứa config_A/, config_B/, ...")
    parser.add_argument("--output", type=str, default="ablation_report.html", help="Tên file HTML báo cáo đầu ra")
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    print(f"🔍 Đang tìm kết quả trong: {results_dir}")
    data = load_configs(results_dir)

    if not data:
        print(f"❌ Không tìm thấy metrics_per_condition.csv nào trong {results_dir}/config_*/")
        print("   Hãy chạy 'python run_c2.py --stage ablation ...' trước.")
        return

    print(f"✅ Tìm thấy kết quả cho config: {list(data.keys())}")

    overall_df = compute_overall(data)
    print("\n=== TỔNG QUAN (weighted theo num_samples mỗi điều kiện) ===")
    print(overall_df.to_string(index=False))

    acc_pivot = build_pivot(data, "accuracy_denom")
    tear_pivot = build_pivot(data, "mAP50_tear", only_notna=True)
    fa_pivot = build_pivot(data, "false_alarm_rate")

    images = {}
    images["acc_heatmap"] = fig_to_base64(
        plot_heatmap(acc_pivot, "Độ chính xác Mệnh giá (%) — Điều kiện x Config", cmap="RdYlGn", vmin=0, vmax=100)
    )
    if not tear_pivot.empty:
        images["tear_heatmap"] = fig_to_base64(
            plot_heatmap(tear_pivot, "mAP50 Phát hiện Rách (%) — Điều kiện x Config", cmap="RdYlGn", vmin=0, vmax=100)
        )
    if not fa_pivot.empty:
        fa_max = max(10, np.nanmax(fa_pivot.values))
        images["fa_heatmap"] = fig_to_base64(
            plot_heatmap(fa_pivot, "Tỷ lệ Báo động giả (%) — càng THẤP càng tốt", cmap="RdYlGn_r", vmin=0, vmax=fa_max)
        )

    images["overall_acc_bar"] = fig_to_base64(
        plot_bar(overall_df, "overall_accuracy_denom", "Độ chính xác Mệnh giá — Tổng thể", "Accuracy (%)")
    )
    if overall_df["overall_mAP50_tear"].notna().any():
        images["overall_tear_bar"] = fig_to_base64(
            plot_bar(overall_df, "overall_mAP50_tear", "mAP50 Phát hiện Rách — Tổng thể", "mAP50 (%)")
        )
    images["overall_fa_bar"] = fig_to_base64(
        plot_bar(overall_df, "overall_false_alarm_rate", "Tỷ lệ Báo động giả — Tổng thể (thấp hơn = tốt hơn)", "False Alarm (%)")
    )

    html = ["<html><head><meta charset='utf-8'><title>CashVision C2 — Ablation Report</title><style>",
            "body{font-family:Arial,Helvetica,sans-serif;max-width:1100px;margin:30px auto;padding:0 20px;background:#fafafa;color:#222}",
            "h1{color:#2c3e50} h2{color:#34495e;border-bottom:2px solid #eee;padding-bottom:6px;margin-top:40px}",
            "table{border-collapse:collapse;width:100%;margin:15px 0;font-size:14px}",
            "th,td{border:1px solid #ddd;padding:7px;text-align:center}",
            "th{background:#2c3e50;color:white} tr:nth-child(even){background:#f2f2f2}",
            "img{max-width:100%;display:block;margin:15px auto;box-shadow:0 2px 8px rgba(0,0,0,.15);border-radius:6px}",
            ".grid{display:flex;flex-wrap:wrap;gap:20px;justify-content:center}",
            ".grid>div{flex:1;min-width:340px}",
            "</style></head><body>",
            "<h1>📊 CashVision C2 — Báo cáo Ablation 2x2 (Config A/B/C/D)</h1>",
            f"<p>Nguồn dữ liệu: <code>{results_dir}</code></p>",
            "<h2>1. Tổng quan (weighted theo số ảnh mỗi điều kiện)</h2>",
            overall_df.to_html(index=False, na_rep="—"),
            "<div class='grid'>",
            f"<div><h3>Accuracy Mệnh giá</h3><img src='data:image/png;base64,{images['overall_acc_bar']}'></div>"]
    if "overall_tear_bar" in images:
        html.append(f"<div><h3>mAP50 Rách</h3><img src='data:image/png;base64,{images['overall_tear_bar']}'></div>")
    html.append(f"<div><h3>False Alarm Rate</h3><img src='data:image/png;base64,{images['overall_fa_bar']}'></div>")
    html.append("</div>")

    html.append("<h2>2. Chi tiết theo từng Điều kiện (Heatmap)</h2>")
    html.append(f"<img src='data:image/png;base64,{images['acc_heatmap']}'>")
    if "tear_heatmap" in images:
        html.append(f"<img src='data:image/png;base64,{images['tear_heatmap']}'>")
    if "fa_heatmap" in images:
        html.append(f"<img src='data:image/png;base64,{images['fa_heatmap']}'>")

    html.append("<h2>3. Bảng dữ liệu đầy đủ theo từng Config</h2>")
    for cfg in sorted(data.keys()):
        html.append(f"<h3>Config {cfg} ({CONFIG_LABELS.get(cfg, cfg)})</h3>")
        html.append(data[cfg].to_html(index=False, na_rep="—"))

    html.append("</body></html>")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = results_dir / args.output
    out_path.write_text("\n".join(html), encoding="utf-8")

    print(f"\n✅ Đã xuất báo cáo trực quan: {out_path}")
    print("   Mở file này bằng trình duyệt để xem. Nếu đang SSH vào server không có GUI,")
    print("   copy file về máy local bằng lệnh (chạy trên máy local):")
    print(f"   scp thaimq@<server_ip>:{out_path} .")


if __name__ == "__main__":
    main()
