#!/bin/bash
# Test script for BoolQ on T4 GPU
# Verifies data mapping (Passage + Question) and yes/no evaluation

set -e

echo "=================================================="
echo "      🧪 TESTING BOOLQ ON T4 (2 SAMPLES)"
echo "=================================================="

# 1. Run Calibration (Tiny sweep)
echo "PHASE 1: CALIBRATION"
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples 2 \
    --num_full_samples 0 \
    --datasets "boolq" \
    --gpu_memory_utilization 0.6

# 2. Run Evaluation
echo "--------------------------------------------------"
echo "📊 PHASE 2: EVALUATION"
echo "--------------------------------------------------"

METHOD=$(python3 -c "import json; print(json.load(open('outputs/calibration/calibration_summary.json'))['boolq']['best_config']['method'])")
THRESHOLD=$(python3 -c "import json; print(json.load(open('outputs/calibration/calibration_summary.json'))['boolq']['best_config']['threshold'])")

python scripts/test_time_compute.py \
    recipes/qwen_test.yaml \
    --dataset_name="boolq" \
    --smart_search=True \
    --score_method="conf" \
    --conf_strategy="$METHOD" \
    --threshold="$THRESHOLD" \
    --num_samples=2 \
    --gpu_memory_utilization=0.6 \
    --enforce_eager=True

echo "=================================================="
echo "✅ BOOLQ T4 TEST COMPLETE"
echo "=================================================="
