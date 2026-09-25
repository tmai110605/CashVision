#!/usr/bin/env python3
"""
create_metadata.py — Generate metadata.csv catalog from dataset file naming conventions and annotations.

Filename structure: {split}_{denom}_{condition}_{index}_jpg.rf.{hash}.jpg
Example:
  - train_100k_backlight_0001_jpg.rf.WMHOUSo8P4ZMzNHK4O6P.jpg -> condition: backlight, is_torn: false, denom: 100000
  - train_500k_torn_overexposed_0023_jpg.rf.qiXs34...jpg      -> condition: torn_bright, is_torn: true, denom: 500000
  - train_10k_torn_clean_0001_jpg.rf...jpg                   -> condition: torn_clean, is_torn: true, denom: 10000
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

# Fix UTF-8 encoding on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

DENOM_MAP = {
    '10k': 10000,
    '20k': 20000,
    '50k': 50000,
    '100k': 100000,
    '200k': 200000,
    '500k': 500000
}

# Mapping from filename patterns to 6 standard environmental conditions
CONDITION_MAP = {
    'indoor': 'indoor',
    'outdoor': 'outdoor',
    'backlight': 'backlight',
    'overexposed': 'overexposed',
    'torn_clean': 'torn_clean',
    'torn_overexposed': 'torn_bright',
    'torn_bright': 'torn_bright'
}


def parse_filename(filename: str):
    """
    Parse filename to extract (denomination_class, condition, is_torn).
    """
    stem = filename.lower()
    
    # 1. Extract denomination
    denom_val = None
    for d_key, d_num in DENOM_MAP.items():
        if f"_{d_key}_" in stem or f"_{d_key}" in stem:
            denom_val = d_num
            break

    # 2. Extract condition
    matched_cond = None
    if "torn_overexposed" in stem or "torn_bright" in stem:
        matched_cond = "torn_bright"
    elif "torn_clean" in stem:
        matched_cond = "torn_clean"
    elif "backlight" in stem:
        matched_cond = "backlight"
    elif "overexposed" in stem:
        matched_cond = "overexposed"
    elif "outdoor" in stem:
        matched_cond = "outdoor"
    elif "indoor" in stem:
        matched_cond = "indoor"

    # 3. is_torn
    is_torn = "true" if ("torn" in stem) else "false"

    return denom_val, matched_cond, is_torn


def check_label_is_torn(label_path: Path, torn_class_id: int = 6):
    """Check whether YOLO label file contains tear defect class."""
    if not label_path.exists():
        return False
    with open(label_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == torn_class_id:
                return True
    return False


def generate_metadata(data_dir: str, output_csv: str, include_splits: list = None):
    data_path = Path(data_dir).resolve()
    if include_splits is None:
        include_splits = ['train', 'valid', 'test']

    rows = []
    skipped = 0

    print(f"[INFO] Scanning dataset in: {data_path}")
    print(f"[INFO] Scanning directories: {include_splits}")

    image_exts = {'.jpg', '.jpeg', '.png'}

    for split in include_splits:
        split_img_dir = data_path / split / "images"
        split_lbl_dir = data_path / split / "labels"

        if not split_img_dir.exists():
            print(f"Directory not found: {split_img_dir}, skipping.")
            continue

        for img_file in sorted(split_img_dir.iterdir()):
            if img_file.suffix.lower() not in image_exts:
                continue

            fname = img_file.name
            denom, cond, is_torn_by_name = parse_filename(fname)

            # Check ground truth label file
            lbl_file = split_lbl_dir / f"{img_file.stem}.txt"
            has_torn_label = check_label_is_torn(lbl_file)
            
            # Determine is_torn status combining filename and bounding box annotations
            is_torn_final = "true" if (is_torn_by_name == "true" or has_torn_label) else "false"

            if cond is None or denom is None:
                print(f"Warning: Failed to parse file: {fname} (denom={denom}, cond={cond})")
                skipped += 1
                continue

            rows.append({
                'filename': fname,
                'denomination_class': denom,
                'condition': cond,
                'is_torn': is_torn_final
            })

    # Ghi file metadata.csv
    out_path = Path(output_csv).resolve()
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'denomination_class', 'condition', 'is_torn'])
        writer.writeheader()
        writer.writerows(rows)

    # Summary statistics
    print("\n" + "=" * 60)
    print(f"CATALOG GENERATED: {out_path}")
    print("=" * 60)
    print(f"  Total images indexed: {len(rows)}")
    print(f"  Skipped files:       {skipped}")
    
    # Distribution by condition
    cond_counts = {}
    torn_counts = {}
    for r in rows:
        c = r['condition']
        cond_counts[c] = cond_counts.get(c, 0) + 1
        if r['is_torn'] == 'true':
            torn_counts[c] = torn_counts.get(c, 0) + 1

    print("\nDISTRIBUTION BY CONDITION:")
    for cond, count in sorted(cond_counts.items()):
        torn_c = torn_counts.get(cond, 0)
        print(f"  - {cond:<16}: {count:>4} images (torn: {torn_c:>3})")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate metadata.csv from dataset filenames and YOLO labels.")
    parser.add_argument("--data_dir", type=str, default=".", help="Root directory containing train/, valid/, test/ splits")
    parser.add_argument("--output", type=str, default="metadata.csv", help="Output path for metadata.csv")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], help="Dataset splits to scan")

    args = parser.parse_args()
    generate_metadata(data_dir=args.data_dir, output_csv=args.output, include_splits=args.splits)


if __name__ == "__main__":
    main()
