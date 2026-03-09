#!/bin/bash
#
# Test LLM-as-Judge on a single sample
# Quick test before running full evaluation
#

set -e

BASE_DIR=$(pwd)

# Check API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set: export OPENAI_API_KEY='your-key'"
    exit 1
fi

echo "======================================================"
echo "Testing LLM-as-Judge Evaluation"
echo "======================================================"
echo ""

# Test on 1 sample
echo "Testing on 1 sample from CoT predictions..."
echo ""

python3 eval/llm_judge_evaluation_v2.py \
    --input results/predictions_test.jsonl \
    --output results/test_eval.json \
    --model claude-opus-4-6 \
    --max-samples 1

echo ""
echo "======================================================"
echo "Test Complete!"
echo "======================================================"
echo ""
echo "Check the output:"
echo "  cat results/test_eval.json"
echo ""
echo "If this works, run the full evaluation:"
echo "  bash scripts/run_complete_evaluation.sh"
echo ""
