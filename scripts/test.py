#!/usr/bin/env python3
"""
验证holdout数据结构的脚本
用于确认base_input的提取是否正确
"""

import json

def verify_structure():
    """验证提取的cases.json结构"""

    # 读取原始holdout数据
    print("读取原始holdout数据...")
    original_records = {}
    with open('dataset/raw/sleep_holdout_with_ids.jsonl', 'r') as f:
        for line in f:
            record = json.loads(line)
            case_id = record['case_study_id']
            if case_id not in original_records:
                original_records[case_id] = {}
            original_records[case_id][record['task_type']] = record

    # 读取提取的cases
    print("读取提取的cases.json...")
    with open('dataset/raw/sleep_holdout_cases.json', 'r') as f:
        extracted_cases = json.load(f)

    print(f"\n原始记录: {len(original_records)} cases")
    print(f"提取的cases: {len(extracted_cases)} cases")

    # 取一个case详细验证
    case_id = sorted(original_records.keys())[0]
    print(f"\n详细验证 Case {case_id}:")
    print("="*80)

    original = original_records[case_id]
    extracted = extracted_cases[case_id]

    # 验证base_input
    base_input = extracted['insights']['base_input']
    print(f"\nBase input长度: {len(base_input)} chars")

    # 验证insights
    print(f"\n1. INSIGHTS验证:")
    insights_original_prompt = original['insights']['prompt']
    insights_extracted_base = extracted['insights']['base_input']

    # base_input应该是prompt去掉Instruction之后的部分
    expected_base = insights_original_prompt.split('Instruction:')[0].strip()

    if insights_extracted_base == expected_base:
        print(f"   ✓ Insights base_input提取正确")
    else:
        print(f"   ✗ Insights base_input提取错误")
        print(f"     期望长度: {len(expected_base)}")
        print(f"     实际长度: {len(insights_extracted_base)}")

    # 验证etiology
    print(f"\n2. ETIOLOGY验证:")
    etiology_original_prompt = original['etiology']['prompt']
    etiology_extracted_base = extracted['etiology']['base_input']

    # etiology的base_input应该与insights的相同
    if etiology_extracted_base == base_input:
        print(f"   ✓ Etiology base_input与insights一致")
    else:
        print(f"   ✗ Etiology base_input与insights不一致")

    # 验证etiology prompt是否包含专家insights
    expert_insights = original['insights']['completion']
    if expert_insights[:100] in etiology_original_prompt:
        print(f"   ✓ 原始etiology prompt包含专家insights")

        # 检查位置
        marker = "Based on the data, we can get the following insights:"
        if marker in etiology_original_prompt:
            idx = etiology_original_prompt.find(marker)
            print(f"   ✓ 找到insights标记，位置: {idx}")
            print(f"   ✓ Base input长度: {len(base_input)}")

            if idx == len(base_input) + 1:  # +1是因为有个空格
                print(f"   ✓ 标记位置正确（紧跟base_input之后）")
    else:
        print(f"   ✗ 原始etiology prompt不包含专家insights")

    # 验证recommendations
    print(f"\n3. RECOMMENDATIONS验证:")
    recommendations_original_prompt = original['recommendations']['prompt']
    recommendations_extracted_base = extracted['recommendations']['base_input']

    # recommendations的base_input应该与insights的相同
    if recommendations_extracted_base == base_input:
        print(f"   ✓ Recommendations base_input与insights一致")
    else:
        print(f"   ✗ Recommendations base_input与insights不一致")

    # 验证recommendations prompt是否包含专家insights和etiology
    expert_etiology = original['etiology']['completion']

    if 'Based on the data, we can get the following insights:' in recommendations_original_prompt:
        print(f"   ✓ 原始recommendations prompt包含insights标记")

    if 'Causes:' in recommendations_original_prompt:
        print(f"   ✓ 原始recommendations prompt包含Causes标记")

        # 检查etiology内容是否在其中
        if expert_etiology[:100] in recommendations_original_prompt:
            print(f"   ✓ 原始recommendations prompt包含专家etiology内容")

    # 打印prompt长度对比
    print(f"\n4. PROMPT长度对比:")
    print(f"   Base input:            {len(base_input):6d} chars (所有tasks共享)")
    print(f"   Insights prompt:       {len(insights_original_prompt):6d} chars")
    print(f"   Etiology prompt:       {len(etiology_original_prompt):6d} chars (+{len(etiology_original_prompt)-len(insights_original_prompt)})")
    print(f"   Recommendations prompt:{len(recommendations_original_prompt):6d} chars (+{len(recommendations_original_prompt)-len(etiology_original_prompt)})")

    print("\n" + "="*80)
    print("验证完成！")

if __name__ == '__main__':
    verify_structure()
