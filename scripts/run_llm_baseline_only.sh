#!/bin/bash
# Dedicated shell script to run the optimized LLM baseline and print final report card.
# Run this after you kill the slow LLM baseline phase of the original script.

set -e

export PYTHONWARNINGS="ignore"
export HF_HUB_DISABLE_SYMLINKS_WARNING="1"

echo "=================================================="
echo "      🚀 STARTING OPTIMIZED LLM BASELINE RUN"
echo "=================================================="

# Run the optimized LLM baseline generation and report script
python scripts/run_llm_only.py \
    --config_test recipes/qwen_test.yaml \
    --num_full_samples -1 \
    --datasets "gsm8k,math500,boolq" \
    "$@"
