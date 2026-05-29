#!/usr/bin/env bash
set -e

echo "=========================================================="
echo "🚀 Running Complete SMARTER Calibration & Evaluation Pipeline"
echo "=========================================================="

# Run full pipeline with offline variables
# --num_calibration_samples: 300 samples for threshold sweep
# --num_full_samples -1: runs evaluation on entire/full datasets
# --datasets: gsm8k, math500, boolq
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python scripts/run_final_pipeline.py \
    --num_calibration_samples 300 \
    --num_full_samples -1 \
    --datasets "gsm8k,math500,boolq" \
    --elbow_method "utility" \
    "$@"
