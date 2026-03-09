import json
import random
import os
import pandas as pd
import argparse
from tqdm import tqdm

SYSTEM_INSTRUCTION = (
    "You are a professional Fitness Coach. Your task is to analyze the user's "
    "demographic information, training load metrics, sleep data, health metrics, "
    "subjective readiness, and muscle soreness to provide professional, personalized "
    "fitness insights and recommendations."
)
SEED = 42
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15  # 剩余部分

def build_chat_messages(instruction, user_input):
    """构建chat格式的消息列表"""
    return [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]

def combine_fitness_inputs(item):
    """将fitness数据的多个输入字段合并为一个完整的用户输入"""
    sections = []

    # 1. Demographics
    if item.get("demographics_input"):
        sections.append("## Demographics\n" + item["demographics_input"])

    # 2. Training Load
    if item.get("training_load_input"):
        sections.append("## Training Load\n" + item["training_load_input"])

    # 3. Sleep Data
    if item.get("sleep_input"):
        sections.append("## Sleep Data\n" + item["sleep_input"])

    # 4. Health Metrics
    if item.get("health_metrics_input"):
        sections.append("## Health Metrics\n" + item["health_metrics_input"])

    # 5. Subjective Readiness
    if item.get("subjective_readiness_input"):
        sections.append("## Subjective Readiness\n" + item["subjective_readiness_input"])

    # 6. Muscle Soreness
    if item.get("muscle_soreness_input"):
        sections.append("## Muscle Soreness\n" + item["muscle_soreness_input"])

    return "\n\n".join(sections)

def combine_fitness_outputs(item):
    """将fitness数据的多个输出字段合并为一个完整的回答"""
    sections = []

    # 1. Demographics Analysis
    if item.get("demographics_output"):
        sections.append("## Demographics Analysis\n" + item["demographics_output"])

    # 2. Training Load Analysis
    if item.get("training_load_output"):
        sections.append("## Training Load Analysis\n" + item["training_load_output"])

    # 3. Sleep Analysis
    if item.get("sleep_output"):
        sections.append("## Sleep Analysis\n" + item["sleep_output"])

    # 4. Health Metrics Analysis
    if item.get("health_metrics_output"):
        sections.append("## Health Metrics Analysis\n" + item["health_metrics_output"])

    # 5. Readiness Assessment
    if item.get("readiness_assessment_output"):
        sections.append("## Readiness Assessment\n" + item["readiness_assessment_output"])

    return "\n\n".join(sections)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", default="dataset/raw/fitness_case_studies.all.v2.jsonl")
    parser.add_argument("--output_dir", default="dataset/parquet/fitness")
    args = parser.parse_args()

    if not os.path.exists(args.input_file):
        print(f"Error: {args.input_file} not found")
        return

    print("Step 1: Reading and cleaning raw data...")
    clean_data = []
    with open(args.input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                # 检查是否有必要的输入和输出字段
                has_input = any([
                    rec.get("demographics_input"),
                    rec.get("training_load_input"),
                    rec.get("sleep_input"),
                    rec.get("health_metrics_input")
                ])
                has_output = any([
                    rec.get("demographics_output"),
                    rec.get("training_load_output"),
                    rec.get("sleep_output"),
                    rec.get("health_metrics_output"),
                    rec.get("readiness_assessment_output")
                ])

                if has_input and has_output:
                    clean_data.append(rec)
            except Exception as e:
                print(f"Warning: Failed to parse line: {e}")
                continue

    print(f"Total valid records: {len(clean_data)}")

    # 设置随机种子以确保可复现
    random.seed(SEED)
    random.shuffle(clean_data)

    # 计算分割点
    total = len(clean_data)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    # 分割数据集
    splits = {
        "train": clean_data[:train_end],
        "val": clean_data[train_end:val_end],
        "test": clean_data[val_end:]
    }

    print(f"\nDataset split (seed={SEED}):")
    print(f"  Train: {len(splits['train'])} records ({TRAIN_RATIO*100:.0f}%)")
    print(f"  Val:   {len(splits['val'])} records ({VAL_RATIO*100:.0f}%)")
    print(f"  Test:  {len(splits['test'])} records ({TEST_RATIO*100:.0f}%)")
    print()

    print("Step 2: Converting to Parquet...")
    os.makedirs(args.output_dir, exist_ok=True)

    for split, data in splits.items():
        rows = []
        for item in tqdm(data, desc=f"Processing {split}"):
            # 1. 合并所有输入字段
            user_input = combine_fitness_inputs(item)

            # 2. 构建 List[Dict] 格式的 Prompt
            chat_msgs = build_chat_messages(SYSTEM_INSTRUCTION, user_input)

            # 3. 合并所有输出字段
            response_str = combine_fitness_outputs(item)

            rows.append({
                "prompt": chat_msgs,
                "response": response_str,
                "ability": "fitness_coach",
                "extra_info": {
                    "case_study_id": item.get("case_study_id", ""),
                    "user_id": item.get("user_id", ""),
                    "vertical": item.get("vertical", "fitness")
                }
            })

        # 保存为 parquet
        save_path = os.path.join(args.output_dir, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(save_path)
        print(f"✓ Saved {split} set to {save_path} ({len(rows)} records)")

if __name__ == "__main__":
    main()
