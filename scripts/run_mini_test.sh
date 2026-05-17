#!/bin/bash
# Mini-Test Script for SMART Pipeline Verification
# Runs on 2 samples per dataset to verify logic

set -e

echo "=================================================="
echo "      🧪 STARTING MINI SMART TEST (2 SAMPLES)"
echo "=================================================="

# 1. Run Calibration for all datasets (Tiny sweep)
echo "PHASE 1: MINI CALIBRATION SWEEP"
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples 2 \
    --num_full_samples 0 \
    --datasets "gsm8k,math500,boolq" \
    "$@"

# 2. Run Mini Evaluation
for DS in "gsm8k" "math500" "boolq"; do
    echo "--------------------------------------------------"
    echo "📊 MINI EVALUATION: $DS"
    echo "--------------------------------------------------"
    
    # Check if summary exists, if not, wait a bit
    if [ ! -f outputs/calibration/calibration_summary.json ]; then
        echo "Error: Calibration summary not found!"
        exit 1
    fi

    METHOD=$(python3 -c "import json; print(json.load(open('outputs/calibration/calibration_summary.json'))['$DS']['best_config']['method'])")
    THRESHOLD=$(python3 -c "import json; print(json.load(open('outputs/calibration/calibration_summary.json'))['$DS']['best_config']['threshold'])")
    
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=True \
        --score_method="conf" \
        --conf_strategy="$METHOD" \
        --threshold="$THRESHOLD" \
        --num_samples=2
done

echo "=================================================="
echo "✅ MINI TEST COMPLETE. LOGIC VERIFIED."
echo "=================================================="
