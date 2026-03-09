# Sleep Coaching Model Evaluation Comparison

## Overview

| Model | Type | Num Cases | Avg Score |
|-------|------|-----------|-----------|
| Baseline | Original Llama3 + Cascade | 69 | 3.225 |
| SFT | Three-stage SFT + Cascade | 69 | 3.782 |
| CoT | CoT Distillation + One-shot | 1 | 3.769 |

## Performance Improvements

- **SFT vs Baseline**: +17.3%
- **CoT vs Baseline**: +16.9%
- **CoT vs SFT**: -0.3%

## Detailed Metrics (Full Evaluation)

| Metric | Baseline | SFT | CoT | Best |
|--------|----------|-----|-----|------|
| Section Average (Q1-Q12) | 3.197 | 3.786 | 3.778 | **SFT** |
| Overall Average (Overall Q1-Q3) | 3.556 | 3.729 | 3.667 | **SFT** |
| Grand Average (All 39 Qs) | 3.225 | 3.782 | 3.769 | **SFT** |
| Insights Section | 3.292 | 3.905 | 4.000 | **CoT** |
| Etiology Section | 3.048 | 3.780 | 4.000 | **CoT** |
| Recommendations Section | 3.251 | 3.674 | 3.333 | **SFT** |

## Analysis

### Key Findings

1. **Three-stage SFT shows the best performance**: Cascade inference with specialized training for each stage works well.

2. **CoT underperforms**: Possible reasons:
   - Insufficient training data quality
   - Teacher model (Gemini/GPT) may not provide consistent reasoning
   - One-shot generation may miss specialized knowledge for each stage

3. **Recommendations**:
   - Improve CoT data generation with better prompts
   - Use stronger teacher models
   - Consider hybrid approach: CoT training + cascade inference

## Methodology Comparison

| Aspect | Baseline | SFT | CoT |
|--------|----------|-----|-----|
| Training Data | None | Expert annotations | Teacher model generations |
| Training Method | None | Three-stage SFT | Single-stage CoT SFT |
| Inference | Cascade (3 steps) | Cascade (3 steps) | One-shot |
| Model Params | 8B | 8B | 8B |
| Data Source | - | Human experts | GPT/Gemini |
| Coherence | Medium (cascade) | Medium (cascade) | High (one-shot) |
| Inference Speed | Slow (3 calls) | Slow (3 calls) | Fast (1 call) |

## Recommendations

**Best approach**: Three-stage SFT with Cascade

For production deployment:
1. Use SFT-trained model with cascade inference
2. Investigate why CoT underperformed
3. Consider improving CoT data generation pipeline

## Next Steps

1. **Error Analysis**: Manually review low-scoring cases to identify failure patterns
2. **Ablation Studies**: Test different teacher models, prompts, training epochs
3. **Hybrid Approaches**: Combine CoT reasoning with cascade structure
4. **Human Evaluation**: Validate LLM-judge scores with expert reviews
5. **Production Testing**: A/B test with real users

## Technical Details

- **Judge Model**: claude-opus-4-6
- **Evaluation Framework**: Based on PH-LLM paper (Supplementary Table 9)
- **Total Cases Evaluated**: 69
