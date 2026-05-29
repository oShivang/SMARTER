#!/bin/bash
# Master Script for Full SMART Pipeline Benchmark
# Runs calibration, baseline generation, and evaluation in one consolidated runner.
# Optimized for H100 or memory-efficient sequential runner.

set -e

export PYTHONWARNINGS="ignore"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "=================================================="
echo "      🚀 STARTING FINAL SMART BENCHMARK"
echo "=================================================="

# Run the consolidated benchmark runner
python scripts/run_benchmarks.py \
    --config_calibrate recipes/qwen_calibrate.yaml \
    --config_test recipes/qwen_test.yaml \
    --num_calibration_samples 100 \
    --num_full_samples -1 \
    --datasets "gsm8k,math500,boolq" \
    --elbow_method "utility" \
    "$@"

# ── Report Card ──
echo ""
echo "======================================================"
echo "  PHASE 3: GENERATING FINAL REPORT CARD"
echo "======================================================"
python scripts/generate_report.py \
    --calibration_json outputs/calibration/calibration_summary.json \
    --output_dir outputs \
    --smart_tag smart_results \
    --slm_tag slm_baseline \
    --llm_tag llm_baseline

echo "=================================================="
echo "✅ BENCHMARK COMPLETE. All results in outputs/"
echo "=================================================="
