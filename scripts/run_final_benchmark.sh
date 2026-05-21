#!/bin/bash
# Master Script for Full SMART Pipeline Benchmark
# For each dataset runs:
#   1. SMART (SLM + LLM intervention)
#   2. SLM-only  (best_of_n, no SMART, no CoT diversity — n=1)
#   3. LLM-only  (best_of_n, no SMART, model = LLM — n=1)
# Then prints a final report card comparing all three.
# Optimized for H100.

set -e

export PYTHONWARNINGS="ignore"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "=================================================="
echo "      🚀 STARTING FINAL SMART BENCHMARK"
echo "=================================================="

# ---- Model paths ----
LLM_MODEL="Qwen/Qwen2.5-7B-Instruct"
SLM_MODEL="Qwen/Qwen2.5-1.5B-Instruct"

# ---- Phase 1: Calibration (100 samples per dataset) ----
echo ""
echo "PHASE 1: CALIBRATION SWEEP (boolq, gsm8k, math500)"
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples 100 \
    --num_full_samples 0 \
    --datasets "boolq,gsm8k,math500" \
    --elbow_method "utility" \
    "$@"

echo ""
echo "✅ Calibration complete."

# ---- Phase 2: Three evaluations per dataset ----
for DS in "boolq" "gsm8k" "math500"; do
    echo ""
    echo "======================================================"
    echo "  📊 DATASET: $DS"
    echo "======================================================"

    if [ ! -f "outputs/calibration/calibration_summary.json" ]; then
        echo "❌ ERROR: calibration_summary.json not found!"
        exit 1
    fi

    METHOD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d.get('$DS', {}).get('best_config', {}).get('method', 'probs_mean'))")
    THRESHOLD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d.get('$DS', {}).get('best_config', {}).get('threshold', '0.5'))")

    if [ "$METHOD" == "None" ] || [ -z "$METHOD" ]; then
        METHOD="probs_mean"
        THRESHOLD="0.5"
        echo "⚠️  No calibrated method for $DS, defaulting to probs_mean@0.5"
    fi

    echo "  Optimal strategy: $METHOD @ threshold=$THRESHOLD"

    # Sample size — half BoolQ, full for others
    SAMPLES="-1"
    if [ "$DS" == "boolq" ]; then
        SAMPLES="1635"
    fi

    # ── 2a. SMART evaluation ──
    echo ""
    echo "  [2a] Running SMART (SLM + LLM intervention)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=True \
        --score_method="conf" \
        --conf_strategy="$METHOD" \
        --threshold="$THRESHOLD" \
        --num_samples="$SAMPLES" \
        --output_dir="outputs/$DS/smart_results"

    # ── 2b. SLM-only one-shot baseline ──
    echo ""
    echo "  [2b] Running SLM-only (1-shot, no CoT)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=False \
        --score_method="conf" \
        --n=1 \
        --num_iterations=1 \
        --model_path="$SLM_MODEL" \
        --draft_model_path="$SLM_MODEL" \
        --num_samples="$SAMPLES" \
        --output_dir="outputs/$DS/slm_baseline"

    # ── 2c. LLM-only one-shot baseline ──
    echo ""
    echo "  [2c] Running LLM-only (1-shot, no CoT)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=False \
        --score_method="conf" \
        --n=1 \
        --num_iterations=1 \
        --model_path="$LLM_MODEL" \
        --draft_model_path="$LLM_MODEL" \
        --num_samples="$SAMPLES" \
        --output_dir="outputs/$DS/llm_baseline"

    echo ""
    echo "  ✅ $DS complete."
done

# ---- Phase 3: Print Final Report Card ----
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
