#!/usr/bin/env python3
"""
为数据集补齐 VERL RL 训练所需字段。

目标：
1. 为每条样本添加 reward_model.ground_truth
2. 将字符串 prompt 转成 VERL RL 需要的 chat message 格式
3. 补齐 data_source 与 extra_info.user_sleep_data，供自定义 reward 使用
"""

import pandas as pd
from pathlib import Path


def _to_chat_prompt(prompt):
    if isinstance(prompt, list):
        return prompt
    return [{"role": "user", "content": str(prompt)}]


def _extract_user_sleep_data(prompt):
    if isinstance(prompt, list):
        for msg in prompt:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        return ""
    return str(prompt)


def add_reward_model_field(input_file: str, output_file: str, data_source: str = "sleep_cot"):
    """
    为 parquet 数据集添加 RL 所需字段
    """
    print(f"读取数据: {input_file}")
    df = pd.read_parquet(input_file)

    print(f"原始列: {df.columns.tolist()}")
    print(f"数据行数: {len(df)}")

    df["prompt"] = df["prompt"].apply(_to_chat_prompt)

    df["reward_model"] = df.apply(
        lambda row: {
            "ground_truth": row["response"],
        },
        axis=1,
    )

    if "data_source" not in df.columns:
        df["data_source"] = data_source
    else:
        df["data_source"] = df["data_source"].fillna(data_source)

    def update_extra_info(row):
        extra_info = row.get("extra_info", {})
        if not isinstance(extra_info, dict):
            extra_info = {}
        extra_info["user_sleep_data"] = _extract_user_sleep_data(row["prompt"])
        if "case_study_id" in row and row["case_study_id"]:
            extra_info["case_study_id"] = row["case_study_id"]
        return extra_info

    df["extra_info"] = df.apply(update_extra_info, axis=1)

    print(f"添加reward_model字段后的列: {df.columns.tolist()}")
    print(f"保存到: {output_file}")
    df.to_parquet(output_file)
    print("✓ 完成")


if __name__ == "__main__":
    base_dir = Path(__file__).parent.parent

    input_dir = base_dir / "dataset/parquet/cot"
    output_dir = base_dir / "dataset/parquet/cot"

    train_input = input_dir / "train.parquet"
    train_output = output_dir / "train_rl.parquet"
    add_reward_model_field(str(train_input), str(train_output))

    print()

    val_input = input_dir / "val.parquet"
    val_output = output_dir / "val_rl.parquet"
    add_reward_model_field(str(val_input), str(val_output))

    print("\n" + "=" * 50)
    print("数据转换完成！")
    print("=" * 50)
    print("\n请更新训练脚本中的数据路径：")
    print(f"  data.train_files={train_output}")
    print(f"  data.val_files={val_output}")
