#!/usr/bin/env python3
"""
将CoT测试数据转换为推断脚本需要的chat格式
"""
import pandas as pd
import json

def convert_to_chat_format(input_file, output_file):
    """
    将prompt字符串转换为chat模板格式

    输入格式：
    {
        "prompt": "You are a sleep medicine expert...",
        "response": "...",
        "case_study_id": "..."
    }

    输出格式：
    {
        "prompt": [{"role": "user", "content": "You are a sleep medicine expert..."}],
        "response": "...",
        "case_study_id": "..."
    }
    """
    print(f"Reading {input_file}...")
    df = pd.read_parquet(input_file)

    print(f"Loaded {len(df)} records")
    print(f"Columns: {df.columns.tolist()}")

    # 转换prompt为chat格式
    def to_chat(prompt_text):
        return [{"role": "user", "content": prompt_text}]

    df['prompt'] = df['prompt'].apply(to_chat)

    # 保存
    df.to_parquet(output_file, index=False)
    print(f"\nConverted and saved to: {output_file}")

    # 显示示例
    print(f"\nExample converted prompt:")
    print(df.iloc[0]['prompt'])

    return df

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Convert CoT test data to chat format')
    parser.add_argument('--input', default='dataset/parquet/cot/test.parquet',
                        help='Input parquet file')
    parser.add_argument('--output', default='dataset/parquet/cot/test_chat.parquet',
                        help='Output parquet file')

    args = parser.parse_args()

    convert_to_chat_format(args.input, args.output)
