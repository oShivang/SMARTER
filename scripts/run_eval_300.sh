#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "================================================================="
echo "🚀 Starting 300-Sample Baseline & SMART Pipeline Evaluation"
echo "================================================================="

# Run the python script
# --num_samples: exactly 300 data points of each dataset
# --datasets: gsm8k,math500,boolq
python scripts/run_eval_300.py \
    --num_samples 300 \
    --datasets "gsm8k,math500,boolq" \
    --config_test "recipes/qwen_test.yaml"

echo "================================================================="
echo "✅ Pipeline Evaluation and Report Generation Completed!"
echo "================================================================="
