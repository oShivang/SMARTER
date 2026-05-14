#!/bin/bash
# Master Script for Full SMART Pipeline Benchmark
# Optimized for H100 (Parallel Loading)

set -e # Exit on error

echo "=================================================="
echo "      🚀 STARTING FINAL SMART BENCHMARK"
echo "=================================================="

# 1. Run Calibration for all datasets
# We use 100 samples for calibration to get a robust threshold
echo "PHASE 1: CALIBRATION SWEEP (GSM8K, MATH500, BOOLQ)"
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples 100 \
    --num_full_samples 0 \
    --datasets "gsm8k,math500,boolq"

echo "Calibration Complete. Results saved in outputs/calibration/"

# 2. Run Full Scale Evaluation
# Note: num_full_samples 0 in calibrate.py skips the mini-run, 
# so we run the actual full benchmarks here for total control.

# Get thresholds from the summary (we can also just let calibrate.py launch them, 
# but running them manually here ensures we don't miss any logs)

for DS in "gsm8k" "math500" "boolq"; do
    echo "--------------------------------------------------"
    echo "📊 EVALUATING FULL DATASET: $DS"
    echo "--------------------------------------------------"
    
    # Check if calibration file exists
    if [ ! -f "outputs/calibration/calibration_summary.json" ]; then
        echo "❌ ERROR: outputs/calibration/calibration_summary.json not found! Calibration must have failed."
        exit 1
    fi

    # Extract the winner method and threshold for this dataset
    METHOD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d['$DS']['best_config']['method'] if '$DS' in d else 'probs_mean')")
    THRESHOLD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d['$DS']['best_config']['threshold'] if '$DS' in d else '0.5')")
    
    if [ "$METHOD" == "None" ] || [ -z "$METHOD" ]; then
        echo "⚠️ Warning: No best method found for $DS, defaulting to probs_mean"
        METHOD="probs_mean"
        THRESHOLD="0.5"
    fi

    echo "Using Optimal Strategy: $METHOD at Threshold: $THRESHOLD"
    
    # Use half dataset for BoolQ (approx 1635 samples), full for others
    SAMPLES="-1"
    if [ "$DS" == "boolq" ]; then
        SAMPLES="1635"
        echo "Note: Running half of BoolQ ($SAMPLES samples)"
    fi

    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=True \
        --score_method="conf" \
        --conf_strategy="$METHOD" \
        --threshold="$THRESHOLD" \
        --num_samples="$SAMPLES"
done

echo "=================================================="
echo "✅ BENCHMARK COMPLETE. ALL RESULTS IN outputs/"
echo "=================================================="
