#!/usr/bin/env python3
"""
Add user data (input field) from sleep_case_studies.all.jsonl to prediction files
"""

import json
import argparse


def main():
    parser = argparse.ArgumentParser(description="Add user data to predictions")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file")
    parser.add_argument("--source", default="data/sleep_case_studies.all.jsonl",
                       help="Source file with user data")
    parser.add_argument("--output", required=True, help="Output JSONL file")

    args = parser.parse_args()

    # Load source data - create case_id -> input mapping
    print(f"Loading source data from {args.source}...")
    user_data_map = {}

    with open(args.source, 'r') as f:
        for line in f:
            item = json.loads(line)
            case_id = item.get('case_study_id')
            user_input = item.get('input')
            if case_id and user_input:
                user_data_map[case_id] = user_input

    print(f"Loaded {len(user_data_map)} cases from source")

    # Load predictions
    print(f"Loading predictions from {args.predictions}...")
    with open(args.predictions, 'r') as f:
        predictions = [json.loads(line) for line in f]

    print(f"Loaded {len(predictions)} predictions")

    # Add user data
    updated = 0
    missing = 0

    for pred in predictions:
        case_id = pred.get('case_study_id')

        if case_id in user_data_map:
            pred['prompt'] = user_data_map[case_id]
            updated += 1
        else:
            print(f"⚠️  Warning: No user data for {case_id}")
            pred['prompt'] = "No user data available"
            missing += 1

    # Save
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print(f"\n✅ Done!")
    print(f"   Updated: {updated}")
    print(f"   Missing: {missing}")

    # Show sample
    if predictions:
        print(f"\n=== Sample ===")
        sample = predictions[0]
        print(f"case_study_id: {sample.get('case_study_id')}")
        print(f"prompt length: {len(sample.get('prompt', ''))} chars")
        print(f"predictions keys: {list(sample.get('predictions', {}).keys())}")


if __name__ == "__main__":
    main()
