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

# Run the consolidated benchmark runner
python scripts/run_benchmarks.py \
    --config_calibrate recipes/qwen_calibrate.yaml \
    --config_test recipes/qwen_test.yaml \
    --num_calibration_samples $N_SAMPLES \
    --num_full_samples $N_SAMPLES \
    --datasets "gsm8k,math500,boolq" \
    --elbow_method "utility" \
    --model_path="$MINI_MODEL" \
    --draft_model_path="$MINI_MODEL" \
    "$@"

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


