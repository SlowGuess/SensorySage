#!/usr/bin/env python3
"""
LLM-as-Judge Evaluation System (Optimized Version)

Based on PH-LLM paper's evaluation rubric (Supplementary Table 9).
Uses SINGLE API call per case study to evaluate all 39 dimensions.

Evaluation Framework:
- 12 section-specific questions (Q1-Q12) for each of 3 sections
- 3 overall quality questions
- Total: 39 evaluation dimensions
- Uses 1 API call per case (not 39!)
"""

import json
from openai import OpenAI
from typing import Dict
import os
from tqdm import tqdm
import time


def create_full_evaluation_prompt(case_id: str, predictions: dict, user_data: str) -> str:
    """Create a single prompt to evaluate all 39 dimensions at once."""

    full_response = f"""**Insights Section**:
{predictions.get('insights', 'N/A')}

**Etiology Section**:
{predictions.get('etiology', 'N/A')}

**Recommendations Section**:
{predictions.get('recommendations', 'N/A')}
"""

    prompt = f"""You are an expert sleep and fitness coach evaluator. Evaluate the following case study on multiple dimensions.

**Case Study ID**: {case_id}

**User Data**:
{user_data}

**Model Response**:
{full_response}

**Evaluation Task**: Rate each dimension below on a 1-5 scale based on the criteria.

## Section-Specific Questions (Q1-Q12)
Evaluate EACH of the three sections (Insights, Etiology, Recommendations) on these 12 questions:

**Q1 - References all IMPORTANT user data**
- 1: None referenced, 2: Some referenced, 3: Half referenced, 4: Most referenced, 5: All referenced

**Q2 - Does NOT reference UNIMPORTANT user data**
- 1: Only unimportant data, 2: Many unimportant, 3: Several unimportant, 4: Few unimportant, 5: No unimportant

**Q3 - Does NOT reference INCORRECT user data (hallucinations)**
- 1: Only incorrect data, 2: Many incorrect, 3: Several incorrect, 4: Few incorrect, 5: No incorrect

**Q4 - Contains all IMPORTANT interpretations (personalization)**
- 1: None present, 2: Many missing, 3: Several missing, 4: Few missing, 5: All present

**Q5 - Does NOT contain UNIMPORTANT interpretations**
- 1: All unimportant, 2: Many unimportant, 3: Some unimportant, 4: Few unimportant, 5: None unimportant

**Q6 - IMPORTANT interpretations are correct**
- 1: All incorrect, 2: Many incorrect, 3: Some incorrect, 4: Few incorrect, 5: None incorrect

**Q7 - UNIMPORTANT interpretations are correct**
- 1: All incorrect, 2: Many incorrect, 3: Some incorrect, 4: Few incorrect, 5: None incorrect

**Q8 - No assumptions beyond provided data (no stereotypes)**
- 1: Many assumptions, 2: Several assumptions, 3: Few assumptions, 4: One assumption, 5: No assumptions

**Q9 - Contains IMPORTANT domain knowledge**
- 1: None referenced, 2: Some referenced, 3: Half referenced, 4: Most referenced, 5: All referenced

**Q10 - Does NOT contain UNIMPORTANT domain knowledge**
- 1: Only unimportant, 2: Many unimportant, 3: Several unimportant, 4: Few unimportant, 5: No unimportant

**Q11 - Does NOT contain INCORRECT domain knowledge**
- 1: Only incorrect, 2: Many incorrect, 3: Several incorrect, 4: Few incorrect, 5: No incorrect

**Q12 - No hallucinations or fabricated knowledge**
- 1: Only hallucinations, 2: Many hallucinations, 3: Several hallucinations, 4: Few hallucinations, 5: No hallucinations

## Overall Questions

**Overall_Q1 - Does NOT contain harmful information**
- 1: Likely serious harm, 2: Likely some harm, 3: May cause harm, 4: Potentially harmful but not significant, 5: No harmful information

**Overall_Q2 - Easy to read, proper formatting, spelling, punctuation**
- 1: Very difficult to read, 2: Many issues, 3: Some issues, 4: Few issues, 5: Easy to read, no issues

**Overall_Q3 - Overall quality**
- 1: Very Poor (needs rewrite), 2: Poor (significant errors), 3: Fair (could improve), 4: Good (slight improvements possible), 5: Excellent (no changes needed)

**Output Format** (JSON):
{{
    "insights_Q1": <1-5>,
    "insights_Q2": <1-5>,
    "insights_Q3": <1-5>,
    "insights_Q4": <1-5>,
    "insights_Q5": <1-5>,
    "insights_Q6": <1-5>,
    "insights_Q7": <1-5>,
    "insights_Q8": <1-5>,
    "insights_Q9": <1-5>,
    "insights_Q10": <1-5>,
    "insights_Q11": <1-5>,
    "insights_Q12": <1-5>,

    "etiology_Q1": <1-5>,
    "etiology_Q2": <1-5>,
    "etiology_Q3": <1-5>,
    "etiology_Q4": <1-5>,
    "etiology_Q5": <1-5>,
    "etiology_Q6": <1-5>,
    "etiology_Q7": <1-5>,
    "etiology_Q8": <1-5>,
    "etiology_Q9": <1-5>,
    "etiology_Q10": <1-5>,
    "etiology_Q11": <1-5>,
    "etiology_Q12": <1-5>,

    "recommendations_Q1": <1-5>,
    "recommendations_Q2": <1-5>,
    "recommendations_Q3": <1-5>,
    "recommendations_Q4": <1-5>,
    "recommendations_Q5": <1-5>,
    "recommendations_Q6": <1-5>,
    "recommendations_Q7": <1-5>,
    "recommendations_Q8": <1-5>,
    "recommendations_Q9": <1-5>,
    "recommendations_Q10": <1-5>,
    "recommendations_Q11": <1-5>,
    "recommendations_Q12": <1-5>,

    "Overall_Q1": <1-5>,
    "Overall_Q2": <1-5>,
    "Overall_Q3": <1-5>,

    "overall_comment": "<2-3 sentences summarizing the evaluation>"
}}

Provide ONLY the JSON output, no additional text.
"""
    return prompt


def evaluate_case(client: OpenAI, case_id: str, predictions: dict, user_data: str, model: str, debug: bool = False) -> dict:
    """Evaluate a single case study with ONE API call."""

    prompt = create_full_evaluation_prompt(case_id, predictions, user_data)

    try:
        # Prepare API call parameters
        api_params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are an expert evaluator providing structured JSON responses with numerical scores."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.0
        }

        # Only add response_format for OpenAI models (not Claude)
        if not model.startswith("claude"):
            api_params["response_format"] = {"type": "json_object"}

        response = client.chat.completions.create(**api_params)

        raw_content = response.choices[0].message.content

        if debug:
            print(f"\n=== DEBUG: Raw API Response ===")
            print(raw_content)
            print("=" * 40)

        # Clean markdown code blocks if present (Claude often returns ```json ... ```)
        import re
        cleaned_content = raw_content.strip()
        cleaned_content = re.sub(r'^```json\s*', '', cleaned_content)
        cleaned_content = re.sub(r'^```\s*', '', cleaned_content)
        cleaned_content = re.sub(r'\s*```$', '', cleaned_content)

        scores = json.loads(cleaned_content)

        # Calculate aggregate scores
        aggregate = calculate_aggregate_scores(scores)
        scores['aggregate'] = aggregate

        return scores

    except Exception as e:
        print(f"Error evaluating {case_id}: {e}")
        # Return default scores
        default_scores = {}
        for section in ["insights", "etiology", "recommendations"]:
            for q in range(1, 13):
                default_scores[f"{section}_Q{q}"] = 3
        default_scores["Overall_Q1"] = 3
        default_scores["Overall_Q2"] = 3
        default_scores["Overall_Q3"] = 3
        default_scores["overall_comment"] = f"Evaluation failed: {str(e)}"
        default_scores["aggregate"] = calculate_aggregate_scores(default_scores)
        return default_scores


def calculate_aggregate_scores(scores: dict) -> dict:
    """Calculate aggregate statistics from individual scores."""

    aggregate = {}

    # Per-section averages
    for section in ["insights", "etiology", "recommendations"]:
        section_scores = [v for k, v in scores.items()
                         if k.startswith(f"{section}_Q") and isinstance(v, (int, float))]
        if section_scores:
            aggregate[f"{section}_avg"] = sum(section_scores) / len(section_scores)

    # Overall section average (Q1-Q12 across all sections)
    all_section_scores = [v for k, v in scores.items()
                          if "_Q" in k and not k.startswith("Overall") and isinstance(v, (int, float))]
    if all_section_scores:
        aggregate["section_avg"] = sum(all_section_scores) / len(all_section_scores)

    # Overall quality average (Overall_Q1-Q3)
    overall_scores = [v for k, v in scores.items()
                     if k.startswith("Overall_Q") and isinstance(v, (int, float))]
    if overall_scores:
        aggregate["overall_avg"] = sum(overall_scores) / len(overall_scores)

    # Grand average (all 39 questions)
    all_scores = [v for k, v in scores.items()
                  if k.endswith(tuple([f"_Q{i}" for i in range(1, 13)] + ["_Q1", "_Q2", "_Q3"]))
                  and isinstance(v, (int, float))]
    if all_scores:
        aggregate["grand_avg"] = sum(all_scores) / len(all_scores)

    return aggregate


def main():
    import argparse

    parser = argparse.ArgumentParser(description="LLM-as-Judge Full Evaluation (Optimized)")
    parser.add_argument("--input", required=True, help="Input JSONL file with predictions")
    parser.add_argument("--output", required=True, help="Output JSON file with results")
    parser.add_argument("--api-key", help="API key (defaults to OPENAI_API_KEY env)")
    parser.add_argument("--base-url", help="Base URL for API")
    parser.add_argument("--model", default="gpt-4o", help="Model for evaluation")
    parser.add_argument("--max-samples", type=int, help="Limit number of samples")
    parser.add_argument("--user-data-key", default="prompt", help="Key for user data in input")
    parser.add_argument("--debug", action="store_true", help="Print raw API responses for debugging")

    args = parser.parse_args()

    # Setup client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key required via --api-key or OPENAI_API_KEY env var")

    client_kwargs = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url

    client = OpenAI(**client_kwargs)

    # Load data
    print(f"Loading predictions from {args.input}...")
    with open(args.input, 'r') as f:
        data = [json.loads(line) for line in f]

    if args.max_samples:
        data = data[:args.max_samples]
        print(f"Evaluating first {args.max_samples} samples")

    print(f"Loaded {len(data)} case studies")
    print(f"Using model: {args.model}")
    print(f"API calls needed: {len(data)} (1 per case)")
    print("")

    # Evaluate
    results = []
    for item in tqdm(data, desc="Evaluating"):
        case_id = item.get("case_study_id", "unknown")
        predictions = item.get("predictions", {})
        user_data = item.get(args.user_data_key, "No user data")

        eval_result = evaluate_case(client, case_id, predictions, user_data, args.model, debug=args.debug)
        eval_result["case_study_id"] = case_id

        results.append(eval_result)

        # Print summary
        agg = eval_result.get("aggregate", {})
        print(f"\n{case_id}:")
        print(f"  Grand Avg: {agg.get('grand_avg', 0):.2f}")
        print(f"  Insights: {agg.get('insights_avg', 0):.2f}, Etiology: {agg.get('etiology_avg', 0):.2f}, Recommendations: {agg.get('recommendations_avg', 0):.2f}")
        print(f"  Overall Quality (Q3): {eval_result.get('Overall_Q3', 'N/A')}")

        time.sleep(1)  # Rate limiting

    # Calculate dataset statistics
    dataset_stats = calculate_dataset_statistics(results)

    # Save
    output_data = {
        "metadata": {
            "model": args.model,
            "num_cases": len(results),
            "api_calls_used": len(results),  # Much better than 39*len(results)!
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "dataset_statistics": dataset_stats,
        "individual_results": results
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("\n=== Dataset Statistics ===")
    for key in ["mean_grand_avg", "mean_section_avg", "mean_overall_avg",
                "mean_insights_avg", "mean_etiology_avg", "mean_recommendations_avg"]:
        if key in dataset_stats:
            print(f"{key}: {dataset_stats[key]:.3f}")

    print(f"\nResults saved to {args.output}")
    print(f"Total API calls: {len(results)} (optimized!)")


def calculate_dataset_statistics(results: list) -> dict:
    """Calculate statistics across all case studies."""

    stats = {}

    # Collect all aggregate scores
    all_aggregates = [r.get("aggregate", {}) for r in results]

    # Calculate means for aggregate metrics
    for key in ["section_avg", "overall_avg", "grand_avg", "insights_avg", "etiology_avg", "recommendations_avg"]:
        values = [agg.get(key, 0) for agg in all_aggregates if key in agg]
        if values:
            stats[f"mean_{key}"] = sum(values) / len(values)
            stats[f"min_{key}"] = min(values)
            stats[f"max_{key}"] = max(values)

    # Per-question statistics
    all_scores = {}
    for result in results:
        for score_key, score_val in result.items():
            if isinstance(score_val, (int, float)) and ("_Q" in score_key):
                if score_key not in all_scores:
                    all_scores[score_key] = []
                all_scores[score_key].append(score_val)

    for score_key, scores in all_scores.items():
        if scores:
            stats[f"mean_{score_key}"] = sum(scores) / len(scores)

    return stats


if __name__ == "__main__":
    main()
