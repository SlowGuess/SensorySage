#!/usr/bin/env python3
"""
对比baseline和微调后模型的性能
"""
import json
import numpy as np
from collections import defaultdict

def load_and_evaluate(file_path):
    """加载预测并计算指标"""
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    except ImportError:
        print("Error: rouge-score not installed")
        return None

    with open(file_path, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    metrics = defaultdict(list)

    for pred in predictions:
        for task in ['insights', 'etiology', 'recommendations']:
            if task not in pred['predictions'] or task not in pred['ground_truth']:
                continue

            pred_text = pred['predictions'][task]
            ref_text = pred['ground_truth'][task]

            # ROUGE-L
            score = scorer.score(ref_text, pred_text)
            metrics[f'{task}_rouge'].append(score['rougeL'].fmeasure)

            # 长度
            metrics[f'{task}_length'].append(len(pred_text))
            metrics[f'{task}_ref_length'].append(len(ref_text))

    # 计算平均值
    results = {}
    for key, values in metrics.items():
        results[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'count': len(values)
        }

    return results

def main():
    print("="*80)
    print("Baseline vs Fine-tuned Model Comparison")
    print("="*80)

    # 加载结果
    print("\nLoading predictions...")
    baseline = load_and_evaluate('results/baseline_predictions.jsonl')
    finetuned = load_and_evaluate('results/test_predictions.jsonl')

    if baseline is None or finetuned is None:
        print("Error loading predictions")
        return

    # 对比展示
    print("\n" + "="*80)
    print("ROUGE-L Comparison")
    print("="*80)

    for task in ['insights', 'etiology', 'recommendations']:
        key = f'{task}_rouge'

        baseline_score = baseline[key]['mean']
        finetuned_score = finetuned[key]['mean']
        improvement = ((finetuned_score - baseline_score) / baseline_score) * 100

        print(f"\n{task.upper()}:")
        print(f"  Baseline:   {baseline_score:.4f}")
        print(f"  Fine-tuned: {finetuned_score:.4f}")
        print(f"  Improvement: {improvement:+.1f}% {'✓' if improvement > 0 else '✗'}")

    # 总体对比
    print("\n" + "="*80)
    print("Overall Comparison")
    print("="*80)

    baseline_avg = np.mean([
        baseline['insights_rouge']['mean'],
        baseline['etiology_rouge']['mean'],
        baseline['recommendations_rouge']['mean']
    ])

    finetuned_avg = np.mean([
        finetuned['insights_rouge']['mean'],
        finetuned['etiology_rouge']['mean'],
        finetuned['recommendations_rouge']['mean']
    ])

    overall_improvement = ((finetuned_avg - baseline_avg) / baseline_avg) * 100

    print(f"\nAverage ROUGE-L:")
    print(f"  Baseline:   {baseline_avg:.4f}")
    print(f"  Fine-tuned: {finetuned_avg:.4f}")
    print(f"  Improvement: {overall_improvement:+.1f}%")

    print("\n" + "="*80)
    print("Conclusion:")
    if overall_improvement > 10:
        print("✓ Fine-tuning shows SIGNIFICANT improvement!")
    elif overall_improvement > 5:
        print("✓ Fine-tuning shows NOTABLE improvement")
    elif overall_improvement > 0:
        print("✓ Fine-tuning shows improvement")
    else:
        print("✗ Fine-tuning did not improve performance")
    print("="*80)

if __name__ == '__main__':
    main()
