"""
将混合训练的JSONL数据转换为Parquet格式
支持Insights, Etiology, Recommendations三个任务的混合训练
"""
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm


def convert_jsonl_to_parquet(input_file, output_file):
    """
    将JSONL格式转换为Parquet格式

    输入JSONL格式:
    {
        "case_study_id": "SC16543",
        "task_type": "insights",
        "prompt": "You are a sleep medicine expert...",
        "completion": "**Sleep Routine:**..."
    }

    输出Parquet格式:
    {
        "prompt": "You are a sleep medicine expert...",
        "response": "**Sleep Routine:**..."
    }
    """
    print(f"Reading {input_file}...")

    rows = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Processing"):
            if not line.strip():
                continue

            try:
                record = json.loads(line)

                # 提取必要字段
                prompt = record.get('prompt', '')
                completion = record.get('completion', '')

                if not prompt or not completion:
                    print(f"Warning: Skipping record with missing prompt or completion")
                    continue

                # 转换为训练格式
                rows.append({
                    "prompt": prompt,
                    "response": completion,
                    "case_study_id": record.get('case_study_id', ''),
                    "task_type": record.get('task_type', '')
                })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    print(f"Total valid records: {len(rows)}")

    # 保存为Parquet
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"Saved to {output_file}")

    return len(rows)


def main():
    parser = argparse.ArgumentParser(description='Convert mixed training JSONL to Parquet')
    parser.add_argument('--input_dir', default='dataset/raw', help='Input directory containing JSONL files')
    parser.add_argument('--output_dir', default='dataset/parquet/sleep_mixed', help='Output directory for Parquet files')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 定义文件映射
    file_mapping = {
        'train': 'sleep_train_with_ids.jsonl',
        'val': 'sleep_validation_with_ids.jsonl',
        'test': 'sleep_test_with_ids.jsonl'
    }

    print("=" * 60)
    print("Converting Mixed Sleep Training Data to Parquet Format")
    print("=" * 60)

    stats = {}
    for split_name, filename in file_mapping.items():
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, f'{split_name}.parquet')

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue

        print(f"\n[{split_name.upper()}]")
        count = convert_jsonl_to_parquet(input_path, output_path)
        stats[split_name] = count

    print("\n" + "=" * 60)
    print("Conversion Summary:")
    print("=" * 60)
    for split_name, count in stats.items():
        print(f"  {split_name:10s}: {count:6d} samples")
    print("=" * 60)
    print("\nDone! You can now use these Parquet files for training.")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    main()
