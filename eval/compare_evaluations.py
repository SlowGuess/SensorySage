#!/usr/bin/env python3
"""
Compare evaluation results from multiple models (Baseline, SFT, CoT, and Ground Truth)
Generates a comprehensive comparison report with statistics and visualizations.
"""

import json
import argparse
from typing import Dict, List, Optional
import pandas as pd

def load_evaluation(filepath: str) -> Dict:
    """Load evaluation results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def extract_statistics(eval_data: Dict) -> Dict:
    """Extract key statistics from evaluation results."""
    stats = {}
    if not eval_data:
        return stats

    if "statistics" in eval_data:
        stats_data = eval_data["statistics"]
        stats["average_score"] = stats_data.get("mean_average_score", 0)
        stats["quality"] = stats_data.get("mean_Overall_Quality", 0)
        stats["data_usage"] = stats_data.get("mean_Data_Usage", 0)
        stats["domain_knowledge"] = stats_data.get("mean_Domain_Knowledge", 0)
        stats["personalization"] = stats_data.get("mean_Personalization", 0)
        stats["readability"] = stats_data.get("mean_Overall_Readability", 0)
        stats["harm"] = stats_data.get("mean_Overall_Harm", 0)

    elif "dataset_statistics" in eval_data:
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

def generate_markdown_report(baseline_stats: Dict, sft_stats: Dict, cot_stats: Dict, gt_stats: Dict, output_file: str):
    """Generate a comprehensive markdown comparison report including Ground Truth."""
    report = []
    report.append("# Sleep Coaching Model Evaluation Comparison")
    report.append("")
    report.append("## Overview")
    report.append("")
    
    # 动态表格头
    has_gt = bool(gt_stats)
    header = "| Model | Type | Num Cases | Avg Score |"
    separator = "|-------|------|-----------|-----------|"
    report.append(header)
    report.append(separator)

    baseline_score = baseline_stats.get("average_score") or baseline_stats.get("grand_avg", 0)
    sft_score = sft_stats.get("average_score") or sft_stats.get("grand_avg", 0)
    cot_score = cot_stats.get("average_score") or cot_stats.get("grand_avg", 0)
    gt_score = (gt_stats.get("average_score") or gt_stats.get("grand_avg", 0)) if has_gt else 0

    if has_gt:
        report.append(f"| **Ground Truth** | **Expert Human Baseline** | {gt_stats.get('num_cases', 0)} | **{gt_score:.3f}** |")
    
    report.append(f"| Baseline | Original Model + Cascade | {baseline_stats.get('num_cases', 0)} | {baseline_score:.3f} |")
    report.append(f"| SFT | Three-stage SFT + Cascade | {sft_stats.get('num_cases', 0)} | {sft_score:.3f} |")
    report.append(f"| CoT | CoT Distillation + One-shot | {cot_stats.get('num_cases', 0)} | {cot_score:.3f} |")
    report.append("")

    report.append("## Performance Improvements & Gaps")
    report.append("")
    report.append(f"- **SFT vs Baseline**: {((sft_score - baseline_score) / baseline_score * 100) if baseline_score else 0:+.1f}%")
    report.append(f"- **CoT vs Baseline**: {((cot_score - baseline_score) / baseline_score * 100) if baseline_score else 0:+.1f}%")
    report.append(f"- **CoT vs SFT**: {((cot_score - sft_score) / sft_score * 100) if sft_score else 0:+.1f}%")
    
    if has_gt and gt_score > 0:
        best_model_score = max(sft_score, cot_score)
        report.append(f"- **Best Model vs Ground Truth Gap**: {((best_model_score - gt_score) / gt_score * 100):+.1f}%")
    report.append("")

    if "section_avg" in baseline_stats:
        report.append("## Detailed Metrics (Full Evaluation)")
        report.append("")
        if has_gt:
            report.append("| Metric | Ground Truth | Baseline | SFT | CoT | Best Model |")
            report.append("|--------|--------------|----------|-----|-----|------------|")
        else:
            report.append("| Metric | Baseline | SFT | CoT | Best Model |")
            report.append("|--------|----------|-----|-----|------------|")

        metrics = [
            ("Section Average (Q1-Q12)", "section_avg"),
            ("Overall Average (Overall Q1-Q3)", "overall_avg"),
            ("Grand Average (All 39 Qs)", "grand_avg"),
            ("Insights Section", "insights_avg"),
            ("Etiology Section", "etiology_avg"),
            ("Recommendations Section", "recommendations_avg")
        ]

        for metric_name, metric_key in metrics:
            g_val = gt_stats.get(metric_key, 0) if has_gt else 0
            b_val = baseline_stats.get(metric_key, 0)
            s_val = sft_stats.get(metric_key, 0)
            c_val = cot_stats.get(metric_key, 0)

            best = "Baseline"
            if s_val > b_val and s_val >= c_val:
                best = "SFT"
            elif c_val > b_val and c_val > s_val:
                best = "CoT"

            if has_gt:
                report.append(f"| {metric_name} | {g_val:.3f} | {b_val:.3f} | {s_val:.3f} | {c_val:.3f} | **{best}** |")
            else:
                report.append(f"| {metric_name} | {b_val:.3f} | {s_val:.3f} | {c_val:.3f} | **{best}** |")
        report.append("")

    with open(output_file, 'w') as f:
        f.write('\n'.join(report))
    print(f"Report saved to {output_file}")

def generate_csv_export(baseline_data: Dict, sft_data: Dict, cot_data: Dict, gt_data: Optional[Dict], output_prefix: str):
    """Export detailed results to CSV for further analysis."""
    baseline_results = baseline_data.get("individual_results") or baseline_data.get("results", [])
    sft_results = sft_data.get("individual_results") or sft_data.get("results", [])
    cot_results = cot_data.get("individual_results") or cot_data.get("results", [])
    gt_results = (gt_data.get("individual_results") or gt_data.get("results", [])) if gt_data else []

    comparison_data = []
    
    # 构建 ID 到 GT 结果的映射表
    gt_map = {res.get("case_study_id"): res for res in gt_results}

    for b_res, s_res, c_res in zip(baseline_results, sft_results, cot_results):
        case_id = b_res.get("case_study_id", "unknown")
        g_res = gt_map.get(case_id, {})
        row = {"case_study_id": case_id}

        if "aggregate" in b_res:
            if gt_data: row["gt_grand"] = g_res.get("aggregate", {}).get("grand_avg", 0)
            row["baseline_grand"] = b_res["aggregate"].get("grand_avg", 0)
            row["sft_grand"] = s_res["aggregate"].get("grand_avg", 0)
            row["cot_grand"] = c_res["aggregate"].get("grand_avg", 0)

            if gt_data: row["gt_insights"] = g_res.get("aggregate", {}).get("insights_avg", 0)
            row["baseline_insights"] = b_res["aggregate"].get("insights_avg", 0)
            row["sft_insights"] = s_res["aggregate"].get("insights_avg", 0)
            row["cot_insights"] = c_res["aggregate"].get("insights_avg", 0)

        comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    csv_file = f"{output_prefix}_comparison.csv"
    df.to_csv(csv_file, index=False)
    print(f"Detailed comparison saved to {csv_file}")

def main():
    parser = argparse.ArgumentParser(description="Compare evaluation results from models and Ground Truth")
    parser.add_argument("--baseline", required=True, help="Baseline evaluation JSON")
    parser.add_argument("--sft", required=True, help="SFT evaluation JSON")
    parser.add_argument("--cot", required=True, help="CoT evaluation JSON")
    parser.add_argument("--gt", required=False, help="Ground Truth evaluation JSON (Optional)")
    parser.add_argument("--output", default="results/comparison_report.md", help="Output markdown report")
    parser.add_argument("--csv-prefix", help="Prefix for CSV export (optional)")
    args = parser.parse_args()

    print("Loading Baseline, SFT, and CoT data...")
    baseline_data = load_evaluation(args.baseline)
    sft_data = load_evaluation(args.sft)
    cot_data = load_evaluation(args.cot)
    gt_data = load_evaluation(args.gt) if args.gt else None

    baseline_stats = extract_statistics(baseline_data)
    sft_stats = extract_statistics(sft_data)
    cot_stats = extract_statistics(cot_data)
    gt_stats = extract_statistics(gt_data) if gt_data else {}

    print("Generating comparison report...")
    generate_markdown_report(baseline_stats, sft_stats, cot_stats, gt_stats, args.output)

    if args.csv_prefix:
        print("Exporting detailed CSV...")
        generate_csv_export(baseline_data, sft_data, cot_data, gt_data, args.csv_prefix)

    print("\nComparison complete!")
    if gt_data:
        print(f"  Ground Truth: {gt_stats.get('grand_avg', 0):.3f}")
    print(f"  Baseline:     {baseline_stats.get('grand_avg', 0):.3f}")
    print(f"  SFT:          {sft_stats.get('grand_avg', 0):.3f}")
    print(f"  CoT:          {cot_stats.get('grand_avg', 0):.3f}")

if __name__ == "__main__":
    main()