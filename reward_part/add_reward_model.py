#!/usr/bin/env python3
"""
为数据集添加reward_model字段
用于VERL RL训练
"""

import pandas as pd
from pathlib import Path

def add_reward_model_field(input_file: str, output_file: str):
    """
    为parquet数据集添加reward_model字段

    Args:
        input_file: 输入parquet文件路径
        output_file: 输出parquet文件路径
    """
    print(f"读取数据: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"原始列: {df.columns.tolist()}")
    print(f"数据行数: {len(df)}")

    # 添加reward_model字段
    # ground_truth使用response字段（SFT的参考答案）
    # 虽然Judge API可能不使用它，但VERL需要这个字段用于日志
    df['reward_model'] = df.apply(lambda row: {
        'ground_truth': row['response']
    }, axis=1)

    print(f"添加reward_model字段后的列: {df.columns.tolist()}")

    # 保存
    print(f"保存到: {output_file}")
    df.to_parquet(output_file)
    print("✓ 完成")

if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    # 处理训练集
    train_input = base_dir / "./dataset/parquet/train_split.parquet"
    train_output = base_dir / "./dataset/parquet/train_with_reward.parquet"
    add_reward_model_field(str(train_input), str(train_output))

    print()

    # 处理验证集
    val_input = base_dir / "./dataset/parquet/val.parquet"
    val_output = base_dir / "./dataset/parquet/val_with_reward.parquet"
    add_reward_model_field(str(val_input), str(val_output))

    print("\n" + "="*50)
    print("数据转换完成！")
    print("="*50)
    print("\n请更新训练脚本中的数据路径：")
    print(f"  data.train_files={train_output}")
    print(f"  data.val_files={val_output}")