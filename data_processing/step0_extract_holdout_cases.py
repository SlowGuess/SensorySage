#!/usr/bin/env python3
"""
Step 0: 从holdout数据中提取50个unique cases（修正版）
将150条记录（50 cases × 3 tasks）按case_id聚合

关键修正：
- base_input对于同一个case的3个tasks是完全相同的
- 从insights task的prompt中提取base_input（去掉Instruction部分）
- 保存每个task的完整原始prompt（包含专家的前置生成内容）
"""

import json
from pathlib import Path
from collections import defaultdict

def extract_base_input_from_insights_prompt(insights_prompt):
    """
    从insights task的prompt中提取base_input

    insights_prompt格式：
    "You are a sleep medicine expert. ... [睡眠数据] ... Instruction: [task instruction]"

    返回：base_input（去掉Instruction部分）
    """
    if "Instruction:" in insights_prompt:
        base_input = insights_prompt.split("Instruction:")[0].strip()
    else:
        # 如果没有Instruction标记，整个prompt就是base_input
        base_input = insights_prompt.strip()

    return base_input

def extract_holdout_cases(input_file, output_file):
    """
    提取holdout数据，按case_id聚合

    关键点：
    1. base_input从insights task提取（因为它没有前置生成内容）
    2. 同一个case的base_input对所有tasks都相同
    3. 保存每个task的完整原始prompt
    """

    cases = defaultdict(dict)

    print(f"读取 {input_file}...")

    # 读取所有holdout数据
    with open(input_file, 'r') as f:
        for line in f:
            record = json.loads(line)
            case_id = record['case_study_id']
            task_type = record['task_type']

            # 保存每个task的数据
            cases[case_id][task_type] = {
                'full_prompt': record['prompt'],  # 完整原始prompt
                'expert_completion': record.get('completion', '')
            }

    # 为每个case提取base_input（从insights task）
    for case_id, tasks in cases.items():
        if 'insights' not in tasks:
            print(f"警告: Case {case_id} 缺少insights task，跳过")
            continue

        # 从insights task提取base_input
        base_input = extract_base_input_from_insights_prompt(tasks['insights']['full_prompt'])

        # 为所有tasks添加base_input（它们共享相同的base_input）
        for task_type in tasks:
            tasks[task_type]['base_input'] = base_input

    # 统计信息
    num_cases = len(cases)
    task_counts = defaultdict(int)
    for case_id, tasks in cases.items():
        for task_type in tasks.keys():
            task_counts[task_type] += 1

    print(f"\n提取统计:")
    print(f"  Unique cases: {num_cases}")
    for task_type, count in sorted(task_counts.items()):
        print(f"  {task_type}: {count} cases")

    # 验证：每个case都应该有3个tasks，且base_input相同
    incomplete_cases = []
    base_input_mismatch = []

    for case_id, tasks in cases.items():
        # 检查是否有3个tasks
        if len(tasks) != 3:
            incomplete_cases.append(case_id)

        # 验证所有tasks的base_input是否相同
        base_inputs = [tasks[t]['base_input'] for t in tasks if 'base_input' in tasks[t]]
        if len(set(base_inputs)) > 1:
            base_input_mismatch.append(case_id)

    if incomplete_cases:
        print(f"\n警告：以下cases缺少某些tasks: {incomplete_cases}")
    else:
        print(f"\n✓ 所有{num_cases}个cases都包含完整的3个tasks")

    if base_input_mismatch:
        print(f"\n警告：以下cases的base_input不一致: {base_input_mismatch}")
    else:
        print(f"✓ 所有cases的base_input在3个tasks间一致")

    # 保存为JSON
    output_data = {
        case_id: tasks
        for case_id, tasks in sorted(cases.items())
    }

    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n保存到: {output_file}")

    # 显示一个示例
    example_case_id = sorted(cases.keys())[0]
    print(f"\n示例 Case {example_case_id}:")
    print(f"  包含tasks: {sorted(cases[example_case_id].keys())}")

    # 验证base_input长度
    base_input = cases[example_case_id]['insights']['base_input']
    print(f"  Base input长度: {len(base_input)} chars")

    # 显示各task的prompt长度
    for task_type in ['insights', 'etiology', 'recommendations']:
        if task_type in cases[example_case_id]:
            full_prompt = cases[example_case_id][task_type]['full_prompt']
            print(f"  {task_type} full_prompt长度: {len(full_prompt)} chars")

    # 验证结构
    print(f"\n验证 etiology 和 recommendations prompt 包含前置内容:")
    etiology_prompt = cases[example_case_id]['etiology']['full_prompt']
    recommendations_prompt = cases[example_case_id]['recommendations']['full_prompt']

    if 'Based on the data, we can get the following insights:' in etiology_prompt:
        print(f"  ✓ etiology包含专家insights标记")
    else:
        print(f"  ✗ etiology不包含insights标记")

    if 'Causes:' in recommendations_prompt:
        print(f"  ✓ recommendations包含专家etiology标记")
    else:
        print(f"  ✗ recommendations不包含etiology标记")

    return output_data

def main():
    cases = extract_holdout_cases(
        input_file='dataset/raw/sleep_holdout_with_ids.jsonl',
        output_file='dataset/raw/sleep_holdout_cases.json'
    )

    print(f"\n完成！提取了 {len(cases)} 个cases")
    print("\n下一步: 运行 step1_generate_llama3_stage1.py 生成insights")

if __name__ == '__main__':
    main()
