# Sleep Coaching Model Evaluation Comparison

## Overview

| Model | Type | Num Cases | Avg Score |
|-------|------|-----------|-----------|
| **Ground Truth** | **Expert Human Baseline** | 69 | **3.707** |
| Baseline | Original Model + Cascade | 69 | 3.225 |
| SFT | Three-stage SFT + Cascade | 69 | 3.782 |
| CoT | CoT Distillation + One-shot | 69 | 3.700 |

## Performance Improvements & Gaps

- **SFT vs Baseline**: +17.3%
- **CoT vs Baseline**: +14.7%
- **CoT vs SFT**: -2.2%
- **Best Model vs Ground Truth Gap**: +2.0%

## Detailed Metrics (Full Evaluation)

| Metric | Ground Truth | Baseline | SFT | CoT | Best Model |
|--------|--------------|----------|-----|-----|------------|
| Section Average (Q1-Q12) | 3.717 | 3.197 | 3.786 | 3.666 | **SFT** |
| Overall Average (Overall Q1-Q3) | 3.585 | 3.556 | 3.729 | 4.111 | **CoT** |
| Grand Average (All 39 Qs) | 3.707 | 3.225 | 3.782 | 3.700 | **SFT** |
| Insights Section | 3.792 | 3.292 | 3.905 | 3.715 | **SFT** |
| Etiology Section | 3.694 | 3.048 | 3.780 | 3.612 | **SFT** |
| Recommendations Section | 3.664 | 3.251 | 3.674 | 3.670 | **SFT** |
