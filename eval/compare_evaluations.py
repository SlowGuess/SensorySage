#!/usr/bin/env python3
"""
Compare evaluation results from multiple models (Baseline, SFT, CoT)

Generates a comprehensive comparison report with statistics and visualizations.
"""

import json
import argparse
from typing import Dict, List
import pandas as pd


def load_evaluation(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_statistics(eval_data: Dict) -> Dict:
    """Extract key statistics from evaluation results."""

    stats = {}

    # Check if it's quick evaluation or full evaluation
    if "statistics" in eval_data:
        # Quick evaluation format
        stats_data = eval_data["statistics"]
        stats["average_score"] = stats_data.get("mean_average_score", 0)
        stats["quality"] = stats_data.get("mean_Overall_Quality", 0)
        stats["data_usage"] = stats_data.get("mean_Data_Usage", 0)
        stats["domain_knowledge"] = stats_data.get("mean_Domain_Knowledge", 0)
        stats["personalization"] = stats_data.get("mean_Personalization", 0)
        stats["readability"] = stats_data.get("mean_Overall_Readability", 0)
        stats["harm"] = stats_data.get("mean_Overall_Harm", 0)

    elif "dataset_statistics" in eval_data:
        # Full evaluation format
        stats_data = eval_data["dataset_statistics"]
        stats["grand_avg"] = stats_data.get("mean_grand_avg", 0)
        stats["section_avg"] = stats_data.get("mean_section_avg", 0)
        stats["overall_avg"] = stats_data.get("mean_overall_avg", 0)
        stats["insights_avg"] = stats_data.get("mean_insights_avg", 0)
        stats["etiology_avg"] = stats_data.get("mean_etiology_avg", 0)
        stats["recommendations_avg"] = stats_data.get("mean_recommendations_avg", 0)

    stats["num_cases"] = eval_data.get("metadata", {}).get("num_cases", 0)
    stats["model"] = eval_data.get("metadata", {}).get("model", "unknown")

    return stats


def generate_markdown_report(baseline_stats: Dict, sft_stats: Dict, cot_stats: Dict, output_file: str):
    """Generate a comprehensive markdown comparison report."""

    report = []

    report.append("# Sleep Coaching Model Evaluation Comparison")
    report.append("")
    report.append("## Overview")
    report.append("")
    report.append("| Model | Type | Num Cases | Avg Score |")
    report.append("|-------|------|-----------|-----------|")

    # Determine which score to use based on available data
    baseline_score = baseline_stats.get("average_score") or baseline_stats.get("grand_avg", 0)
    sft_score = sft_stats.get("average_score") or sft_stats.get("grand_avg", 0)
    cot_score = cot_stats.get("average_score") or cot_stats.get("grand_avg", 0)

    report.append(f"| Baseline | Original Llama3 + Cascade | {baseline_stats['num_cases']} | {baseline_score:.3f} |")
    report.append(f"| SFT | Three-stage SFT + Cascade | {sft_stats['num_cases']} | {sft_score:.3f} |")
    report.append(f"| CoT | CoT Distillation + One-shot | {cot_stats['num_cases']} | {cot_score:.3f} |")
    report.append("")

    # Calculate improvements
    sft_improvement = ((sft_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
    cot_improvement = ((cot_score - baseline_score) / baseline_score * 100) if baseline_score > 0 else 0
    cot_vs_sft = ((cot_score - sft_score) / sft_score * 100) if sft_score > 0 else 0

    report.append("## Performance Improvements")
    report.append("")
    report.append(f"- **SFT vs Baseline**: {sft_improvement:+.1f}%")
    report.append(f"- **CoT vs Baseline**: {cot_improvement:+.1f}%")
    report.append(f"- **CoT vs SFT**: {cot_vs_sft:+.1f}%")
    report.append("")

    # Quick evaluation comparison
    if "quality" in baseline_stats:
        report.append("## Detailed Metrics (Quick Evaluation)")
        report.append("")
        report.append("| Metric | Baseline | SFT | CoT | Best |")
        report.append("|--------|----------|-----|-----|------|")

        metrics = [
            ("Overall Quality", "quality"),
            ("Data Usage", "data_usage"),
            ("Domain Knowledge", "domain_knowledge"),
            ("Personalization", "personalization"),
            ("Readability", "readability"),
            ("Safety (no harm)", "harm")
        ]

        for metric_name, metric_key in metrics:
            b_val = baseline_stats.get(metric_key, 0)
            s_val = sft_stats.get(metric_key, 0)
            c_val = cot_stats.get(metric_key, 0)

            best = "Baseline"
            if s_val > b_val and s_val >= c_val:
                best = "SFT"
            elif c_val > b_val and c_val > s_val:
                best = "CoT"

            report.append(f"| {metric_name} | {b_val:.3f} | {s_val:.3f} | {c_val:.3f} | **{best}** |")

        report.append("")

    # Full evaluation comparison
    elif "section_avg" in baseline_stats:
        report.append("## Detailed Metrics (Full Evaluation)")
        report.append("")
        report.append("| Metric | Baseline | SFT | CoT | Best |")
        report.append("|--------|----------|-----|-----|------|")

        metrics = [
            ("Section Average (Q1-Q12)", "section_avg"),
            ("Overall Average (Overall Q1-Q3)", "overall_avg"),
            ("Grand Average (All 39 Qs)", "grand_avg"),
            ("Insights Section", "insights_avg"),
            ("Etiology Section", "etiology_avg"),
            ("Recommendations Section", "recommendations_avg")
        ]

        for metric_name, metric_key in metrics:
            b_val = baseline_stats.get(metric_key, 0)
            s_val = sft_stats.get(metric_key, 0)
            c_val = cot_stats.get(metric_key, 0)

            best = "Baseline"
            if s_val > b_val and s_val >= c_val:
                best = "SFT"
            elif c_val > b_val and c_val > s_val:
                best = "CoT"

            report.append(f"| {metric_name} | {b_val:.3f} | {s_val:.3f} | {c_val:.3f} | **{best}** |")

        report.append("")

    # Analysis
    report.append("## Analysis")
    report.append("")

    if cot_score > sft_score and sft_score > baseline_score:
        report.append("### Key Findings")
        report.append("")
        report.append("1. **CoT Distillation shows the best performance**: The one-shot generation approach after CoT training outperforms both baseline and three-stage SFT methods.")
        report.append("")
        report.append("2. **Three-stage SFT improves over baseline**: Training on expert-annotated data with cascade inference shows improvements over the untrained base model.")
        report.append("")
        report.append("3. **CoT advantages**:")
        report.append("   - Simpler inference (one-shot vs cascade)")
        report.append("   - Better coherence across sections")
        report.append("   - Captures reasoning process from teacher model")
        report.append("")
    elif sft_score > cot_score and sft_score > baseline_score:
        report.append("### Key Findings")
        report.append("")
        report.append("1. **Three-stage SFT shows the best performance**: Cascade inference with specialized training for each stage works well.")
        report.append("")
        report.append("2. **CoT underperforms**: Possible reasons:")
        report.append("   - Insufficient training data quality")
        report.append("   - Teacher model (Gemini/GPT) may not provide consistent reasoning")
        report.append("   - One-shot generation may miss specialized knowledge for each stage")
        report.append("")
        report.append("3. **Recommendations**:")
        report.append("   - Improve CoT data generation with better prompts")
        report.append("   - Use stronger teacher models")
        report.append("   - Consider hybrid approach: CoT training + cascade inference")
        report.append("")

    # Methodology comparison
    report.append("## Methodology Comparison")
    report.append("")
    report.append("| Aspect | Baseline | SFT | CoT |")
    report.append("|--------|----------|-----|-----|")
    report.append("| Training Data | None | Expert annotations | Teacher model generations |")
    report.append("| Training Method | None | Three-stage SFT | Single-stage CoT SFT |")
    report.append("| Inference | Cascade (3 steps) | Cascade (3 steps) | One-shot |")
    report.append("| Model Params | 8B | 8B | 8B |")
    report.append("| Data Source | - | Human experts | GPT/Gemini |")
    report.append("| Coherence | Medium (cascade) | Medium (cascade) | High (one-shot) |")
    report.append("| Inference Speed | Slow (3 calls) | Slow (3 calls) | Fast (1 call) |")
    report.append("")

    # Recommendations
    report.append("## Recommendations")
    report.append("")

    if cot_score > sft_score:
        report.append("**Best approach**: CoT Distillation")
        report.append("")
        report.append("For production deployment:")
        report.append("1. Use CoT-trained model with one-shot inference")
        report.append("2. Continue improving CoT training data quality")
        report.append("3. Consider ensemble with SFT model for critical cases")
        report.append("")
    else:
        report.append("**Best approach**: Three-stage SFT with Cascade")
        report.append("")
        report.append("For production deployment:")
        report.append("1. Use SFT-trained model with cascade inference")
        report.append("2. Investigate why CoT underperformed")
        report.append("3. Consider improving CoT data generation pipeline")
        report.append("")

    report.append("## Next Steps")
    report.append("")
    report.append("1. **Error Analysis**: Manually review low-scoring cases to identify failure patterns")
    report.append("2. **Ablation Studies**: Test different teacher models, prompts, training epochs")
    report.append("3. **Hybrid Approaches**: Combine CoT reasoning with cascade structure")
    report.append("4. **Human Evaluation**: Validate LLM-judge scores with expert reviews")
    report.append("5. **Production Testing**: A/B test with real users")
    report.append("")

    # Technical details
    report.append("## Technical Details")
    report.append("")
    report.append(f"- **Judge Model**: {baseline_stats['model']}")
    report.append(f"- **Evaluation Framework**: Based on PH-LLM paper (Supplementary Table 9)")
    report.append(f"- **Total Cases Evaluated**: {baseline_stats['num_cases']}")
    report.append("")

    # Write report
    with open(output_file, 'w') as f:
        f.write('\n'.join(report))

    print(f"Report saved to {output_file}")


def generate_csv_export(baseline_data: Dict, sft_data: Dict, cot_data: Dict, output_prefix: str):
    """Export detailed results to CSV for further analysis."""

    # Try to get individual results
    baseline_results = baseline_data.get("individual_results") or baseline_data.get("results", [])
    sft_results = sft_data.get("individual_results") or sft_data.get("results", [])
    cot_results = cot_data.get("individual_results") or cot_data.get("results", [])

    # Create comparison DataFrame
    comparison_data = []

    for b_res, s_res, c_res in zip(baseline_results, sft_results, cot_results):
        case_id = b_res.get("case_study_id", "unknown")

        row = {"case_study_id": case_id}

        # Quick eval format
        if "average_score" in b_res:
            row["baseline_avg"] = b_res.get("average_score", 0)
            row["sft_avg"] = s_res.get("average_score", 0)
            row["cot_avg"] = c_res.get("average_score", 0)

            row["baseline_quality"] = b_res.get("Overall_Quality", 0)
            row["sft_quality"] = s_res.get("Overall_Quality", 0)
            row["cot_quality"] = c_res.get("Overall_Quality", 0)

        # Full eval format
        elif "aggregate" in b_res:
            row["baseline_grand"] = b_res["aggregate"].get("grand_avg", 0)
            row["sft_grand"] = s_res["aggregate"].get("grand_avg", 0)
            row["cot_grand"] = c_res["aggregate"].get("grand_avg", 0)

            row["baseline_insights"] = b_res["aggregate"].get("insights_avg", 0)
            row["sft_insights"] = s_res["aggregate"].get("insights_avg", 0)
            row["cot_insights"] = c_res["aggregate"].get("insights_avg", 0)

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_file = f"{output_prefix}_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed comparison saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results from multiple models")
    parser.add_argument("--baseline", required=True, help="Baseline evaluation JSON")
    parser.add_argument("--sft", required=True, help="SFT evaluation JSON")
    parser.add_argument("--cot", required=False, help="CoT evaluation JSON")
    parser.add_argument("--output", default="results/comparison_report.md", help="Output markdown report")
    parser.add_argument("--csv-prefix", help="Prefix for CSV export (optional)")

    args = parser.parse_args()

    # Load evaluation results
    print(f"Loading baseline from {args.baseline}...")
    baseline_data = load_evaluation(args.baseline)

    print(f"Loading SFT from {args.sft}...")
    sft_data = load_evaluation(args.sft)

    print(f"Loading CoT from {args.cot}...")
    cot_data = load_evaluation(args.cot)

    # Extract statistics
    baseline_stats = extract_statistics(baseline_data)
    sft_stats = extract_statistics(sft_data)
    cot_stats = extract_statistics(cot_data)

    # Generate report
    print("Generating comparison report...")
    generate_markdown_report(baseline_stats, sft_stats, cot_stats, args.output)

    # Optional CSV export
    if args.csv_prefix:
        print("Exporting detailed CSV...")
        generate_csv_export(baseline_data, sft_data, cot_data, args.csv_prefix)

    print("\nComparison complete!")
    print(f"\nQuick Summary:")
    print(f"  Baseline: {baseline_stats.get('average_score') or baseline_stats.get('grand_avg', 0):.3f}")
    print(f"  SFT:      {sft_stats.get('average_score') or sft_stats.get('grand_avg', 0):.3f}")
    print(f"  CoT:      {cot_stats.get('average_score') or cot_stats.get('grand_avg', 0):.3f}")


if __name__ == "__main__":
    main()
