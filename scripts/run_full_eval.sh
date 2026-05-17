#!/bin/bash
# ============================================================
#   Full Evaluation Script — Uses calibration already done
#   Thresholds sourced from completed calibration run
# ============================================================

set -e
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=================================================="
echo "   📊 RUNNING FULL SMART EVALUATION"
echo "=================================================="

# ── Thresholds from calibration ──────────────────────────
BOOLQ_METHOD="top_2_diff"
BOOLQ_THRESHOLD="0.9004992053840244"
BOOLQ_SAMPLES="1635"

GSM8K_METHOD="mean_least_3"
GSM8K_THRESHOLD="0.000089"
GSM8K_SAMPLES="-1"

MATH500_METHOD="probs_mean"
MATH500_THRESHOLD="0.862487"
MATH500_SAMPLES="-1"
# ─────────────────────────────────────────────────────────

run_eval() {
    DS=$1
    METHOD=$2
    THRESHOLD=$3
    SAMPLES=$4

    echo ""
    echo "--------------------------------------------------"
    echo "📊 EVALUATING: $DS"
    echo "   Strategy : $METHOD"
    echo "   Threshold: $THRESHOLD"
    echo "   Samples  : $SAMPLES"
    echo "--------------------------------------------------"

    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=True \
        --score_method="conf" \
        --conf_strategy="$METHOD" \
        --threshold="$THRESHOLD" \
        --num_samples="$SAMPLES"

    echo "✅ $DS done."
}

run_eval "boolq"   "$BOOLQ_METHOD"   "$BOOLQ_THRESHOLD"   "$BOOLQ_SAMPLES"
run_eval "gsm8k"   "$GSM8K_METHOD"   "$GSM8K_THRESHOLD"   "$GSM8K_SAMPLES"
run_eval "math500" "$MATH500_METHOD" "$MATH500_THRESHOLD" "$MATH500_SAMPLES"

echo ""
echo "=================================================="
echo "✅ ALL EVALUATIONS COMPLETE. Results in outputs/"
echo "=================================================="
