#!/bin/bash
#
# Complete Evaluation Pipeline
# Runs inference and evaluation for all three models: Baseline, SFT, CoT
#

set -e  # Exit on error

BASE_DIR=$(pwd)
RESULTS_DIR="$BASE_DIR/results"

# Configuration
EVAL_MODEL="gpt-4o"  # Model for LLM judging
EVAL_TYPE="quick"     # "quick" or "full"
MAX_SAMPLES=""        # Leave empty for all, or set number for testing

# API Configuration
if [ -z "$OPENAI_API_KEY" ]; then
    echo "Error: OPENAI_API_KEY not set"
    echo "Please set your API key:"
    echo "  export OPENAI_API_KEY='your-api-key-here'"
    exit 1
fi

echo "======================================================"
echo "Complete Evaluation Pipeline"
echo "======================================================"
echo "Evaluation Model: $EVAL_MODEL"
echo "Evaluation Type: $EVAL_TYPE"
echo "Results Directory: $RESULTS_DIR"
echo "======================================================"
echo ""

mkdir -p "$RESULTS_DIR"

# ============================================
# Step 1: Run Inference (if not already done)
# ============================================

echo "Step 1: Checking inference results..."
echo ""

if [ ! -f "$RESULTS_DIR/baseline_predictions.jsonl" ]; then
    echo "⚠️  Baseline predictions not found"
    echo "Please run: bash scripts/run_baseline_inference.sh"
else
    echo "✓ Baseline predictions found"
fi

if [ ! -f "$RESULTS_DIR/sft_predictions.jsonl" ]; then
    echo "⚠️  SFT predictions not found"
    echo "Please run: bash scripts/run_sft_inference.sh"
else
    echo "✓ SFT predictions found"
fi

if [ ! -f "$RESULTS_DIR/predictions_test.jsonl" ]; then
    echo "⚠️  CoT predictions not found"
    echo "Please run: bash scripts/run_cot_inference.sh"
else
    echo "✓ CoT predictions found"
fi

echo ""

# ============================================
# Step 2: Run LLM-as-Judge Evaluation
# ============================================

echo "Step 2: Running LLM-as-Judge Evaluation..."
echo ""

if [ "$EVAL_TYPE" == "quick" ]; then
    EVAL_SCRIPT="llm_judge_quick.py"
else
    EVAL_SCRIPT="llm_judge_evaluation.py"
fi

# Set max samples flag
SAMPLES_FLAG=""
if [ ! -z "$MAX_SAMPLES" ]; then
    SAMPLES_FLAG="--max-samples $MAX_SAMPLES"
fi

# Evaluate Baseline
if [ -f "$RESULTS_DIR/baseline_predictions.jsonl" ]; then
    echo "Evaluating Baseline model..."
    python3 scripts/$EVAL_SCRIPT \
        --input "$RESULTS_DIR/baseline_predictions.jsonl" \
        --output "$RESULTS_DIR/eval_baseline.json" \
        --model "$EVAL_MODEL" \
        $SAMPLES_FLAG
    echo "✓ Baseline evaluation complete"
    echo ""
fi

# Evaluate SFT
if [ -f "$RESULTS_DIR/sft_predictions.jsonl" ]; then
    echo "Evaluating SFT model..."
    python3 scripts/$EVAL_SCRIPT \
        --input "$RESULTS_DIR/sft_predictions.jsonl" \
        --output "$RESULTS_DIR/eval_sft.json" \
        --model "$EVAL_MODEL" \
        $SAMPLES_FLAG
    echo "✓ SFT evaluation complete"
    echo ""
fi

# Evaluate CoT
if [ -f "$RESULTS_DIR/predictions_test.jsonl" ]; then
    echo "Evaluating CoT model..."
    python3 scripts/$EVAL_SCRIPT \
        --input "$RESULTS_DIR/predictions_test.jsonl" \
        --output "$RESULTS_DIR/eval_cot.json" \
        --model "$EVAL_MODEL" \
        $SAMPLES_FLAG
    echo "✓ CoT evaluation complete"
    echo ""
fi

# ============================================
# Step 3: Generate Comparison Report
# ============================================

echo "Step 3: Generating comparison report..."
echo ""

python3 scripts/compare_evaluations.py \
    --baseline "$RESULTS_DIR/eval_baseline.json" \
    --sft "$RESULTS_DIR/eval_sft.json" \
    --cot "$RESULTS_DIR/eval_cot.json" \
    --output "$RESULTS_DIR/comparison_report.md" \
    --csv-prefix "$RESULTS_DIR/detailed"

echo "✓ Comparison report generated"
echo ""

# ============================================
# Step 4: Display Results
# ============================================

echo "======================================================"
echo "Evaluation Complete!"
echo "======================================================"
echo ""
echo "Results saved to:"
echo "  - $RESULTS_DIR/eval_baseline.json"
echo "  - $RESULTS_DIR/eval_sft.json"
echo "  - $RESULTS_DIR/eval_cot.json"
echo "  - $RESULTS_DIR/comparison_report.md"
echo "  - $RESULTS_DIR/detailed_comparison.csv"
echo ""
echo "View the comparison report:"
echo "  cat $RESULTS_DIR/comparison_report.md"
echo ""

# Display quick summary
if command -v jq &> /dev/null; then
    echo "Quick Summary:"
    echo ""

    if [ -f "$RESULTS_DIR/eval_baseline.json" ]; then
        baseline_score=$(jq -r '.statistics.mean_average_score // .dataset_statistics.mean_grand_avg // 0' "$RESULTS_DIR/eval_baseline.json")
        echo "  Baseline: $baseline_score"
    fi

    if [ -f "$RESULTS_DIR/eval_sft.json" ]; then
        sft_score=$(jq -r '.statistics.mean_average_score // .dataset_statistics.mean_grand_avg // 0' "$RESULTS_DIR/eval_sft.json")
        echo "  SFT:      $sft_score"
    fi

    if [ -f "$RESULTS_DIR/eval_cot.json" ]; then
        cot_score=$(jq -r '.statistics.mean_average_score // .dataset_statistics.mean_grand_avg // 0' "$RESULTS_DIR/eval_cot.json")
        echo "  CoT:      $cot_score"
    fi

    echo ""
else
    echo "Tip: Install jq for quick summary display"
    echo "  brew install jq  # macOS"
    echo "  apt install jq   # Ubuntu"
    echo ""
fi

echo "======================================================"
