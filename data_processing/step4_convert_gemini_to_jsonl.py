#!/usr/bin/env python3
"""
Step 4: 将Gemini的完整三阶段结果转换为标准JSONL格式
输入：gemini_holdout_stage3_complete.json
输出：sleep_holdout_gemini_with_ids.jsonl (150条记录)
"""

import json
from pathlib import Path

def convert_to_standard_jsonl(stage3_file, output_file):
    """将stage3的JSON转换为标准JSONL格式"""

    # 加载stage3完整结果
    print(f"Loading stage3 results from {stage3_file}...")
    with open(stage3_file, 'r') as f:
        stage3_results = json.load(f)

    print(f"Loaded {len(stage3_results)} cases")

    # 转换为JSONL记录
    records = []

    for case_id in sorted(stage3_results.keys()):
        case_data = stage3_results[case_id]

        # 为每个task创建一条记录
        tasks = [
            {
                'task_type': 'insights',
                'prompt': case_data['insights_prompt'],
                'completion': case_data['insights_output']
            },
            {
                'task_type': 'etiology',
                'prompt': case_data['etiology_prompt'],
                'completion': case_data['etiology_output']
            },
            {
                'task_type': 'recommendations',
                'prompt': case_data['recommendations_prompt'],
                'completion': case_data['recommendations_output']
            }
        ]

        for task in tasks:
            record = {
                'case_study_id': case_id,
                'task_type': task['task_type'],
                'prompt': task['prompt'],
                'completion': task['completion']
            }
            records.append(record)

    # 保存为JSONL
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"\n完成！转换了 {len(records)} 条记录")
    print(f"保存到: {output_file}")

    # 统计验证
    print(f"\n统计验证:")
    print(f"  总记录数: {len(records)}")
    print(f"  Unique cases: {len(stage3_results)}")
    print(f"  每个case的tasks: {len(records) // len(stage3_results)}")

    task_counts = {}
    for record in records:
        task_type = record['task_type']
        task_counts[task_type] = task_counts.get(task_type, 0) + 1

    print(f"\n各task统计:")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count} 条")

    # 显示一个完整示例
    example_case_id = sorted(stage3_results.keys())[0]
    example_records = [r for r in records if r['case_study_id'] == example_case_id]

    print(f"\n示例 Case {example_case_id}:")
    for record in example_records:
        print(f"  {record['task_type']}:")
        print(f"    - Prompt length: {len(record['prompt'])} chars")
        print(f"    - Completion length: {len(record['completion'])} chars")

    return records

def validate_format(jsonl_file, reference_file):
    """验证生成的JSONL格式是否与参考文件一致"""

    print(f"\n验证格式一致性...")
    print(f"  生成文件: {jsonl_file}")
    print(f"  参考文件: {reference_file}")

    # 读取生成的文件
    with open(jsonl_file, 'r') as f:
        generated_records = [json.loads(line) for line in f if line.strip()]

    # 读取参考文件
    with open(reference_file, 'r') as f:
        reference_records = [json.loads(line) for line in f if line.strip()]

    print(f"\n记录数对比:")
    print(f"  生成文件: {len(generated_records)} 条")
    print(f"  参考文件: {len(reference_records)} 条")

    # 验证字段
    if generated_records:
        gen_keys = set(generated_records[0].keys())
        ref_keys = set(reference_records[0].keys())

        print(f"\n字段对比:")
        print(f"  生成文件字段: {sorted(gen_keys)}")
        print(f"  参考文件字段: {sorted(ref_keys)}")

        if gen_keys == ref_keys:
            print(f"  ✓ 字段完全一致")
        else:
            print(f"  ✗ 字段不一致")

    # 验证case_ids是否相同
    gen_case_ids = set(r['case_study_id'] for r in generated_records)
    ref_case_ids = set(r['case_study_id'] for r in reference_records)

    print(f"\nCase IDs对比:")
    print(f"  生成文件: {len(gen_case_ids)} 个unique cases")
    print(f"  参考文件: {len(ref_case_ids)} 个unique cases")

    if gen_case_ids == ref_case_ids:
        print(f"  ✓ Case IDs完全一致")
    else:
        print(f"  ✗ Case IDs不一致")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Convert Gemini results to standard JSONL format')
    parser.add_argument('--stage3_file', default='results/gemini_holdout_stage3_complete.json',
                        help='Input stage3 JSON file')
    parser.add_argument('--output_file', default='dataset/raw/sleep_holdout_gemini_with_ids.jsonl',
                        help='Output JSONL file')
    parser.add_argument('--validate', action='store_true',
                        help='Validate format against reference file')
    parser.add_argument('--reference_file', default='dataset/raw/sleep_holdout_with_ids.jsonl',
                        help='Reference JSONL file for validation')
    args = parser.parse_args()

    # 转换
    records = convert_to_standard_jsonl(
        stage3_file=args.stage3_file,
        output_file=args.output_file
    )

    # 可选：验证格式
    if args.validate:
        validate_format(args.output_file, args.reference_file)

    print("\n" + "="*80)
    print("Gemini响应生成完成！")
    print("="*80)
    print(f"\n生成的文件: {args.output_file}")
    print(f"  - 包含 {len(records)} 条记录")
    print(f"  - 对应 {len(records)//3} 个unique cases")
    print("\n现在你有4个来源的holdout数据:")
    print("  1. sleep_holdout_with_ids.jsonl (专家)")
    print("  2. sleep_holdout_model_with_ids.jsonl (PH-LLM)")
    print("  3. sleep_holdout_llama3_with_ids.jsonl (Llama3)")
    print("  4. sleep_holdout_gemini_with_ids.jsonl (Gemini)")
    print("\n下一步: 准备人工评分（Llama3和Gemini各150条 × 15个principles）")

if __name__ == '__main__':
    main()
