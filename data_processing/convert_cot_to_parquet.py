"""
将CoT (Chain-of-Thought) 训练数据从JSONL转换为Parquet格式
用于训练Llama3学习思维链推理模式
"""
import json
import os
import pandas as pd
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split


def convert_jsonl_to_parquet(input_file, output_file):
    """
    将CoT JSONL格式转换为Parquet格式

    输入JSONL格式:
    {
        "case_study_id": "SC16543",
        "prompt": "You are a sleep medicine expert...",
        "response": "<THINKING>...<INSIGHTS>...<ETIOLOGY>...<RECOMMENDATIONS>..."
    }

    输出Parquet格式:
    {
        "prompt": "You are a sleep medicine expert...",
        "response": "<THINKING>...<INSIGHTS>...<ETIOLOGY>...<RECOMMENDATIONS>..."
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
                response = record.get('response', '')
                case_id = record.get('case_study_id', '')

                if not prompt or not response:
                    print(f"Warning: Skipping record {case_id} with missing prompt or response")
                    continue

                # 检查response是否包含CoT标记
                if '<THINKING>' not in response:
                    print(f"Warning: Record {case_id} missing <THINKING> tag")

                # 转换为训练格式
                rows.append({
                    "prompt": prompt,
                    "response": response,
                    "case_study_id": case_id
                })

            except json.JSONDecodeError as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    print(f"Total valid records: {len(rows)}")

    # 统计数据长度
    if rows:
        prompt_lengths = [len(r['prompt']) for r in rows]
        response_lengths = [len(r['response']) for r in rows]
        print(f"\n数据统计:")
        print(f"  平均Prompt长度: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
        print(f"  平均Response长度: {sum(response_lengths)/len(response_lengths):.0f} chars")
        print(f"  最大Prompt长度: {max(prompt_lengths)} chars")
        print(f"  最大Response长度: {max(response_lengths)} chars")

        # 估算token数（粗略估计：1 token ≈ 4 chars）
        max_total_chars = max([len(r['prompt']) + len(r['response']) for r in rows])
        estimated_max_tokens = max_total_chars / 4
        print(f"  估算最大token数: {estimated_max_tokens:.0f} tokens")
        print(f"  建议max_length: {int(estimated_max_tokens * 1.1)}")  # 加10%余量

    # 保存为Parquet
    df = pd.DataFrame(rows)
    df.to_parquet(output_file, engine='pyarrow', index=False)
    print(f"\n已保存到: {output_file}")

    return len(rows)


def split_and_convert(input_file, output_dir, train_ratio=0.9, val_ratio=0.05, test_ratio=0.05, random_state=42):
    """
    读取完整数据，分割为train/val/test并转换为parquet

    Args:
        input_file: CoT SFT JSONL文件
        output_dir: 输出目录
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        random_state: 随机种子
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "比例之和必须为1"

    print(f"Reading {input_file}...")

    # 读取所有数据
    all_records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    record = json.loads(line)
                    if record.get('prompt') and record.get('response'):
                        all_records.append(record)
                except:
                    continue

    print(f"Total records: {len(all_records)}")

    # 分割数据
    # 先分出测试集
    train_val, test = train_test_split(
        all_records,
        test_size=test_ratio,
        random_state=random_state
    )

    # 再从train_val中分出验证集
    val_size_from_train_val = val_ratio / (train_ratio + val_ratio)
    train, val = train_test_split(
        train_val,
        test_size=val_size_from_train_val,
        random_state=random_state
    )

    splits = {
        'train': train,
        'val': val,
        'test': test
    }

    print(f"\n数据分割:")
    print(f"  训练集: {len(train)} samples ({len(train)/len(all_records)*100:.1f}%)")
    print(f"  验证集: {len(val)} samples ({len(val)/len(all_records)*100:.1f}%)")
    print(f"  测试集: {len(test)} samples ({len(test)/len(all_records)*100:.1f}%)")

    # 保存各个分割
    os.makedirs(output_dir, exist_ok=True)
    stats = {}

    for split_name, records in splits.items():
        # 转换为DataFrame
        rows = []
        for record in records:
            rows.append({
                "prompt": record['prompt'],
                "response": record['response'],
                "case_study_id": record.get('case_study_id', '')
            })

        df = pd.DataFrame(rows)
        output_path = os.path.join(output_dir, f'{split_name}.parquet')
        df.to_parquet(output_path, engine='pyarrow', index=False)
        print(f"  已保存 {split_name}: {output_path}")
        stats[split_name] = len(rows)

    return stats


def main():
    parser = argparse.ArgumentParser(description='Convert CoT JSONL to Parquet for training')
    parser.add_argument('--input_file', required=True,
                        help='Input CoT SFT JSONL file (e.g., data/cot_training_full_sft.jsonl)')
    parser.add_argument('--output_dir', default='dataset/parquet/cot',
                        help='Output directory for Parquet files')
    parser.add_argument('--split', action='store_true',
                        help='Split data into train/val/test')
    parser.add_argument('--train_ratio', type=float, default=0.9,
                        help='Training set ratio (default: 0.9)')
    parser.add_argument('--val_ratio', type=float, default=0.05,
                        help='Validation set ratio (default: 0.05)')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='Test set ratio (default: 0.05)')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for splitting')

    args = parser.parse_args()

    print("=" * 60)
    print("Converting CoT Training Data to Parquet Format")
    print("=" * 60)

    if args.split:
        # 分割并转换
        stats = split_and_convert(
            input_file=args.input_file,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            random_state=args.random_state
        )

        print("\n" + "=" * 60)
        print("Conversion Summary:")
        print("=" * 60)
        for split_name, count in stats.items():
            print(f"  {split_name:10s}: {count:6d} samples")
    else:
        # 直接转换整个文件
        output_file = os.path.join(args.output_dir, 'full.parquet')
        os.makedirs(args.output_dir, exist_ok=True)
        count = convert_jsonl_to_parquet(args.input_file, output_file)

        print("\n" + "=" * 60)
        print(f"Converted {count} samples to {output_file}")

    print("=" * 60)
    print("\nDone! You can now use these Parquet files for training.")
    print(f"Output directory: {args.output_dir}")
    print("\n下一步:")
    print("  1. 查看生成的parquet文件")
    print("  2. 运行CoT训练脚本")


if __name__ == '__main__':
    main()
