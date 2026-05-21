#!/bin/bash
# Mini-Test Script for SMART Pipeline Verification
# Phase 1: Tune threshold on 5 questions
# Phase 2: Three evaluations per dataset (SMART, SLM 1-shot, LLM 1-shot)
# Phase 3: Print report card
# All on 5 samples — Both SLM and LLM set to Qwen2.5-1.5B-Instruct

set -e

export PYTHONWARNINGS="ignore"
export VLLM_LOGGING_LEVEL="WARNING"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"
export HF_HUB_DISABLE_TELEMETRY="1"

MINI_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
N_SAMPLES=5

echo "=================================================="
echo "      🧪 MINI SMART TEST ($N_SAMPLES SAMPLES)"
echo "  SLM + LLM = $MINI_MODEL"
echo "=================================================="

# ── Phase 1: Calibration ──
echo ""
echo "PHASE 1: MINI CALIBRATION SWEEP ($N_SAMPLES samples)"
python scripts/calibrate.py \
    --config recipes/qwen_calibrate.yaml \
    --num_calibration_samples $N_SAMPLES \
    --num_full_samples 0 \
    --datasets "gsm8k,math500,boolq" \
    --elbow_method "utility" \
    --model_path="$MINI_MODEL" \
    --draft_model_path="$MINI_MODEL" \
    "$@"

echo ""
echo "✅ Calibration complete."

# ── Phase 2: Three evaluations per dataset ──
for DS in "gsm8k" "math500" "boolq"; do
    echo ""
    echo "======================================================"
    echo "  📊 DATASET: $DS"
    echo "======================================================"

    if [ ! -f outputs/calibration/calibration_summary.json ]; then
        echo "❌ Error: Calibration summary not found!"
        exit 1
    fi

    METHOD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d.get('$DS', {}).get('best_config', {}).get('method', 'probs_mean'))")
    THRESHOLD=$(python3 -c "import json; d=json.load(open('outputs/calibration/calibration_summary.json')); print(d.get('$DS', {}).get('best_config', {}).get('threshold', '0.5'))")

    if [ "$METHOD" == "None" ] || [ -z "$METHOD" ]; then
        METHOD="probs_mean"
        THRESHOLD="0.5"
        echo "⚠️  Defaulting to probs_mean@0.5"
    fi

    echo "  Strategy : $METHOD  |  Threshold: $THRESHOLD"

    # 2a. SMART
    echo ""
    echo "  [2a] SMART (SLM + LLM intervention)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=True \
        --score_method="conf" \
        --conf_strategy="$METHOD" \
        --threshold="$THRESHOLD" \
        --num_samples=$N_SAMPLES \
        --model_path="$MINI_MODEL" \
        --draft_model_path="$MINI_MODEL" \
        --output_dir="outputs/$DS/smart_results"

    # 2b. SLM-only 1-shot
    echo ""
    echo "  [2b] SLM-only (1-shot, no CoT)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=False \
        --score_method="conf" \
        --n=1 \
        --num_iterations=1 \
        --model_path="$MINI_MODEL" \
        --draft_model_path="$MINI_MODEL" \
        --num_samples=$N_SAMPLES \
        --output_dir="outputs/$DS/slm_baseline"

    # 2c. LLM-only 1-shot (same 1.5B model in mini-test)
    echo ""
    echo "  [2c] LLM-only (1-shot, no CoT)..."
    python scripts/test_time_compute.py \
        recipes/qwen_test.yaml \
        --dataset_name="$DS" \
        --smart_search=False \
        --score_method="conf" \
        --n=1 \
        --num_iterations=1 \
        --model_path="$MINI_MODEL" \
        --draft_model_path="$MINI_MODEL" \
        --num_samples=$N_SAMPLES \
        --output_dir="outputs/$DS/llm_baseline"

    echo "  ✅ $DS done."
done

# ── Phase 3: Report Card ──
echo ""
echo "======================================================"
echo "  PHASE 3: FINAL REPORT CARD"
echo "======================================================"
python scripts/generate_report.py \
    --calibration_json outputs/calibration/calibration_summary.json \
    --output_dir outputs \
    --smart_tag smart_results \
    --slm_tag slm_baseline \
    --llm_tag llm_baseline

echo ""
echo "=================================================="
echo "✅ MINI TEST COMPLETE ($N_SAMPLES samples each)"
echo "   SLM + LLM (intervention): $MINI_MODEL"
echo "=================================================="


