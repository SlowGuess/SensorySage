#!/usr/bin/env python3
"""
从全量数据集中提取 Test 集合的 Ground Truth，
并将其映射为 LLM Judge 标准评估格式。
"""

import json
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract 69 Test Cases Ground Truth (Updated Format)")
    parser.add_argument("--all-data", default="data/sleep_case_studies.all.jsonl", help="包含所有数据的源文件")
    parser.add_argument("--ref-preds", required=True, help="参考的预测文件 (用于提取 69 个对齐的 Test ID)")
    parser.add_argument("--output", default="results/groundtruth_predictions.jsonl", help="输出的 Ground Truth 格式化文件")
    
    args = parser.parse_args()

    # 1. 提取 69 个 Test Case 的 ID，确保严格对齐
    print(f"Loading reference IDs from {args.ref_preds}...")
    test_ids = set()
    with open(args.ref_preds, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            if "case_study_id" in item:
                test_ids.add(item["case_study_id"])
    
    print(f"Found {len(test_ids)} unique test cases in reference file.")

    # 2. 捞取 Ground Truth 并重新组装键值
    print(f"Extracting and formatting ground truth from {args.all_data}...")
    processed_data = []
    found_count = 0
    
    with open(args.all_data, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            case_id = item.get("case_study_id")
            
            # 只有当该 ID 在我们的测试集列表中时才处理
            if case_id in test_ids:
                found_count += 1
                
                # 提取原始 Prompt
                user_prompt = item.get("input", "No user data")
                
                # 直接映射原数据中的三个专门字段
                predictions = {
                    "insights": item.get("insight_output", "").strip(),
                    "etiology": item.get("etiology_output", "").strip(),
                    "recommendations": item.get("recommendation_output", "").strip()
                }
                
                # 组装为 LLM Judge 需要的格式
                formatted_item = {
                    "case_study_id": case_id,
                    "prompt": user_prompt,
                    "predictions": predictions
                }
                processed_data.append(formatted_item)

    # 3. 保存输出
    print(f"\nSaving to {args.output}...")
    with open(args.output, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"\n✅ Done!")
    print(f"   Target IDs expected: {len(test_ids)}")
    print(f"   Successfully formatted: {found_count}")

if __name__ == "__main__":
    main()