#!/usr/bin/env python3
"""
Final Report Card Generator for SMART Benchmark.
Reads the calibration summary JSON and the three result JSON files per dataset
(SMART, SLM-only, LLM-only) and prints a comprehensive report card table.
"""

import json
import os
import sys
import glob
import argparse

DATASETS = ["boolq", "gsm8k", "math500"]

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

def find_result_file(results_dir, tag):
    """Search results_dir and its alternate sibling (output vs outputs) for a jsonl file containing the tag."""
    search_dirs = [results_dir]
    normalized_dir = results_dir.replace("\\", "/")
    if "outputs/" in normalized_dir:
        search_dirs.append(results_dir.replace("outputs/", "output/"))
    elif "output/" in normalized_dir:
        search_dirs.append(results_dir.replace("output/", "outputs/"))
        
    matched_files = []
    for s_dir in search_dirs:
        if not os.path.exists(s_dir):
            continue
        pattern = os.path.join(s_dir, "**", "*.jsonl")
        files = glob.glob(pattern, recursive=True)
        for f in files:
            if os.path.isdir(f):
                continue
            normalized_path = f.replace("\\", "/")
            if tag in normalized_path:
                matched_files.append(f)
    if not matched_files:
        return None
    # Pick the most recently modified file
    return sorted(matched_files, key=os.path.getmtime)[-1]

def extract_accuracy(result_path):
    """
    Reads a results JSONL and computes accuracy from the 'correct' field
    (written by evaluate.py). Falls back to reading acc from a metrics JSON if present.
    """
    if result_path is None:
        return None
    # Try metrics JSON first
    metrics_path = result_path.replace(".jsonl", "_metrics.json")
    if os.path.exists(metrics_path):
        m = load_json(metrics_path)
        if m:
            # Handle both {"acc": X} and {"acc": {"pred": X}} shapes
            acc = m.get("acc", m.get("accuracy", None))
            if isinstance(acc, dict):
                # Take the first numeric value
                acc = next(iter(acc.values()), None)
            if acc is not None:
                return float(acc)
    # Fall back: compute from individual lines
    try:
        correct = 0
        total = 0
        with open(result_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                total += 1
                if row.get("correct", False):
                    correct += 1
        if total > 0:
            return round((correct / total) * 100, 1)
    except Exception:
        pass
    return None

def main():
    parser = argparse.ArgumentParser()
    
    # Look for calibration summary in both output and outputs
    calib_options = [
        "outputs/calibration/calibration_summary.json",
        "outputs/calibration_summary.json",
        "output/calibration/calibration_summary.json",
        "output/calibration_summary.json"
    ]
    default_calib = None
    for opt in calib_options:
        if os.path.exists(opt):
            default_calib = opt
            break
            
    default_dir = "outputs" if os.path.exists("outputs") else "output"
    if not default_calib:
        default_calib = os.path.join(default_dir, "calibration/calibration_summary.json")
        
    parser.add_argument("--calibration_json", default=default_calib)
    parser.add_argument("--output_dir", default=default_dir)
    parser.add_argument("--smart_tag", default="smart_results", help="File tag for SMART results")
    parser.add_argument("--slm_tag", default="slm_baseline", help="File tag for SLM-only results")
    parser.add_argument("--llm_tag", default="llm_baseline", help="File tag for LLM-only results")
    args = parser.parse_args()

    calib = load_json(args.calibration_json)

    print("\n" + "=" * 90)
    print("                    📊  SMART BENCHMARK FINAL REPORT CARD")
    print("=" * 90)
    print(f"{'Dataset':<10} | {'Method':<14} | {'Thresh':>7} | "
          f"{'Proj Cost%':>10} | {'Proj Acc%':>9} | "
          f"{'SLM 1-shot%':>11} | {'LLM 1-shot%':>11} | "
          f"{'SMART Acc%':>10} | {'SMART Cost%':>11}")
    print("-" * 90)

    for ds in DATASETS:
        # --- Calibration info ---
        if calib and ds in calib:
            bc = calib[ds].get("best_config", {})
            method    = bc.get("method", "N/A")
            threshold = bc.get("threshold", float("nan"))
            proj_acc  = bc.get("acc", None)
            proj_cost = bc.get("cost", None)
        else:
            method = threshold = "N/A"
            proj_acc = proj_cost = None

        # --- Read result files ---
        ds_dir = os.path.join(args.output_dir, ds)

        smart_file = find_result_file(ds_dir, args.smart_tag)
        slm_file   = find_result_file(ds_dir, args.slm_tag)
        llm_file   = find_result_file(ds_dir, args.llm_tag)

        smart_acc = extract_accuracy(smart_file)
        slm_acc   = extract_accuracy(slm_file)
        llm_acc   = extract_accuracy(llm_file)

        # --- SMART cost (actual): read from JSONL if available ---
        smart_cost = None
        if smart_file and os.path.exists(smart_file):
            try:
                llm_tok = 0
                total_tok = 0
                with open(smart_file) as f:
                    for line in f:
                        row = json.loads(line.strip()) if line.strip() else {}
                        llm_tok   += sum(row.get("llm_tokens", []) or [0])
                        total_tok += row.get("total_tokens", 0)
                if total_tok > 0:
                    smart_cost = round((llm_tok / total_tok) * 100, 2)
            except Exception:
                pass

        def fmt(v, fmt_str="{:.1f}"):
            return fmt_str.format(v) if v is not None else "N/A"

        thresh_str = f"{threshold:.6f}" if isinstance(threshold, float) else str(threshold)

        print(f"{ds:<10} | {method:<14} | {thresh_str:>7} | "
              f"{fmt(proj_cost, '{:.2f}'):>10} | {fmt(proj_acc):>9} | "
              f"{fmt(slm_acc):>11} | {fmt(llm_acc):>11} | "
              f"{fmt(smart_acc):>10} | {fmt(smart_cost, '{:.2f}'):>11}")

    print("=" * 90)
    print("  Proj = Projected from calibration sweep | Cost = % of tokens generated by LLM")
    print("=" * 90 + "\n")

if __name__ == "__main__":
    main()
