#!/usr/bin/env python3
"""
可视化baseline vs 微调模型的对比
"""
import json
import matplotlib.pyplot as plt
import numpy as np

def load_rouge_scores(file_path):
    """提取ROUGE分数"""
    from rouge_score import rouge_scorer
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    with open(file_path, 'r') as f:
        predictions = [json.loads(line) for line in f if line.strip()]

    scores = {'insights': [], 'etiology': [], 'recommendations': []}

    for pred in predictions:
        for task in scores.keys():
            if task in pred['predictions'] and task in pred['ground_truth']:
                score = scorer.score(
                    pred['ground_truth'][task],
                    pred['predictions'][task]
                )
                scores[task].append(score['rougeL'].fmeasure)

    return scores

# 加载数据
baseline = load_rouge_scores('results/baseline_predictions.jsonl')
finetuned = load_rouge_scores('results/test_predictions.jsonl')

# 创建对比图
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
tasks = ['insights', 'etiology', 'recommendations']

for i, task in enumerate(tasks):
    ax = axes[i]

    data = [baseline[task], finetuned[task]]
    labels = ['Baseline', 'Fine-tuned']

    bp = ax.boxplot(data, labels=labels, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightgreen')

    ax.set_ylabel('ROUGE-L Score')
    ax.set_title(task.capitalize())
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/baseline_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 可视化结果已保存: results/baseline_comparison.png")
