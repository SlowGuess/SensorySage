#!/usr/bin/env python3
"""
使用思维链prompt调用教师模型（Claude）生成训练数据
用于后续蒸馏训练Llama3
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI  # 使用OpenAI客户端（兼容格式）
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from prompts.cot_sleep_coaching_prompt import format_prompt_with_data, parse_cot_response


def create_gemini_client(api_key=None, base_url=None):
    """
    创建 Gemini API 客户端（使用OpenAI兼容格式）
    """
    client = OpenAI(
        api_key=api_key or "sk-QHWpCYbhD79wjJPFc5U0VvJqYk5CYEMq32g5izTLzirGIQXQ",  # 请通过命令行参数--api_key传入真实key
        base_url=base_url or "https://once.novai.su/v1"
    )
    return client


def call_gemini_api(client, prompt, model="[次]gemini-3-pro-preview", max_tokens=16384):
    """
    调用Gemini API生成思维链响应（通过OpenAI兼容接口）
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,  # 贪婪解码，保证可复现
        )

        # OpenAI格式返回
        return response.choices[0].message.content

    except Exception as e:
        print(f"API调用错误: {e}")
        raise


def generate_cot_training_data(
    input_file,
    output_file,
    model="[次]gemini-3-pro-preview",
    max_samples=None,
    api_key=None,
    base_url=None,
    sft_output_file=None
):
    """
    为训练数据生成思维链响应

    Args:
        input_file: 输入JSONL文件（如sleep_train_with_ids.jsonl）
        output_file: 输出JSONL文件（包含思维链的训练数据）
        model: Gemini模型名称
        max_samples: 限制处理的样本数
        api_key: API密钥（可选，用于换API时指定）
        base_url: API基础URL（可选，用于换API时指定）
        sft_output_file: SFT输出文件（可选，用于断点续传和实时转换）
    """

    # 创建 Gemini 客户端
    client = create_gemini_client(api_key=api_key, base_url=base_url)

    # 读取输入数据并按case_id聚合
    print(f"Loading data from {input_file}...")
    cases = {}

    with open(input_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue

            record = json.loads(line)
            case_id = record['case_study_id']

            if case_id not in cases:
                cases[case_id] = {
                    'case_study_id': case_id,
                    'tasks': {}
                }

            task_type = record['task_type']
            cases[case_id]['tasks'][task_type] = {
                'prompt': record['prompt'],
                'expert_completion': record['completion']
            }

    print(f"Loaded {len(cases)} unique cases")

    # 为每个case提取base_input（从insights task，因为它没有前置内容）
    print("\n提取base_input（包含用户基本信息和睡眠数据）...")
    for case_id, case_data in cases.items():
        if 'insights' not in case_data['tasks']:
            print(f"Warning: Case {case_id} 缺少insights task，跳过")
            continue

        insights_prompt = case_data['tasks']['insights']['prompt']

        # 提取base_input（去掉Instruction部分）
        # base_input应该包含：专家前缀 + 用户信息 + 睡眠数据
        if 'Instruction:' in insights_prompt:
            base_input = insights_prompt.split('Instruction:')[0].strip()
            case_data['base_input'] = base_input

            # 验证是否包含用户信息
            if case_id == list(cases.keys())[0]:  # 只对第一个case打印验证
                print(f"\n示例 Case {case_id} 的base_input前300字符:")
                print(base_input[:300])
                print(f"...（总长度: {len(base_input)} chars）")

                # 检查是否包含关键信息
                has_expert_prefix = "sleep medicine expert" in base_input.lower()
                has_user_info = "years old" in base_input or "male" in base_input or "female" in base_input
                has_sleep_data = "Sleep logs:" in base_input or "Sleep Score" in base_input

                print(f"\n验证base_input完整性:")
                print(f"  ✓ 包含专家前缀: {has_expert_prefix}")
                print(f"  ✓ 包含用户信息: {has_user_info}")
                print(f"  ✓ 包含睡眠数据: {has_sleep_data}")

                if not (has_expert_prefix and has_user_info and has_sleep_data):
                    print(f"  警告：base_input可能不完整！")
        else:
            print(f"Warning: No 'Instruction:' found in {case_id}, using full prompt")
            case_data['base_input'] = insights_prompt

    # 限制样本数
    case_ids = list(cases.keys())
    if max_samples:
        case_ids = case_ids[:max_samples]
        print(f"\n处理前 {max_samples} 个cases")

    # 断点续传：基于SFT文件检查已经处理过的cases（如果指定了sft_output_file）
    processed_case_ids = set()

    # 优先检查SFT文件（最终输出），如果没有则检查CoT文件
    check_file = None
    if sft_output_file and Path(sft_output_file).exists():
        check_file = sft_output_file
        print(f"\n检测到已存在的SFT输出文件: {sft_output_file}")
    elif Path(output_file).exists():
        check_file = output_file
        print(f"\n检测到已存在的CoT输出文件: {output_file}")

    if check_file:
        with open(check_file, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        processed_case_ids.add(record['case_study_id'])
                    except:
                        continue
        print(f"  已在SFT文件中找到 {len(processed_case_ids)} 个cases")

        # 过滤掉已处理的cases
        remaining_case_ids = [cid for cid in case_ids if cid not in processed_case_ids]
        print(f"  剩余 {len(remaining_case_ids)} 个cases待处理")
        case_ids = remaining_case_ids

        if len(case_ids) == 0:
            print("\n所有cases都已处理完成！")
            return []

    # 生成思维链响应
    results = []

    print(f"\n开始生成思维链响应（使用模型: {model}）...")

    for case_id in tqdm(case_ids, desc="Generating CoT responses"):
        case_data = cases[case_id]

        if 'base_input' not in case_data:
            print(f"Skipping {case_id}: no base_input")
            continue

        print(f"\n处理 {case_id}...")

        # 构造完整prompt（将base_input插入CoT prompt模板）
        full_prompt = format_prompt_with_data(case_data['base_input'])

        print(f"  Base input长度: {len(case_data['base_input'])} chars")
        print(f"  完整prompt长度: {len(full_prompt)} chars")

        # 调用Gemini API
        try:
            response_text = call_gemini_api(client, full_prompt, model=model)
            print(f"  生成响应长度: {len(response_text)} chars")

            # 解析响应
            parsed = parse_cot_response(response_text)

            # 验证解析结果
            parsed_keys = list(parsed.keys())
            print(f"  解析到的部分: {parsed_keys}")

            if not all(key in parsed for key in ['thinking', 'insights', 'etiology', 'recommendations']):
                print(f"  警告：响应解析不完整！")

            # 保存结果
            result = {
                'case_study_id': case_id,
                'base_input': case_data['base_input'],
                'full_prompt': full_prompt,
                'raw_response': response_text,
                'parsed_response': parsed,
                'expert_ground_truth': {
                    task: data['expert_completion']
                    for task, data in case_data['tasks'].items()
                }
            }

            results.append(result)

            # 保存到CoT文件（覆盖模式：替换同case_id的旧记录）
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)

            # 读取现有的所有记录
            existing_records = {}
            if Path(output_file).exists():
                with open(output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                rec = json.loads(line)
                                existing_records[rec['case_study_id']] = rec
                            except:
                                continue

            # 用新记录覆盖（如果存在同case_id）或添加
            existing_records[case_id] = result

            # 重写整个文件
            with open(output_file, 'w') as f:
                for cid in sorted(existing_records.keys()):  # 排序保持顺序
                    f.write(json.dumps(existing_records[cid], ensure_ascii=False) + '\n')

            print(f"  ✓ 已更新CoT文件（覆盖模式）")

            # 如果指定了SFT输出文件，立即转换并追加写入
            if sft_output_file:
                import re
                sft_record = {
                    'case_study_id': case_id,
                    'prompt': case_data['base_input'],
                    'response': re.sub(r'\n?```$', '', re.sub(r'^```[\w]*\n?', '', response_text.strip()).strip())
                }

                Path(sft_output_file).parent.mkdir(parents=True, exist_ok=True)
                sft_mode = 'a' if Path(sft_output_file).exists() else 'w'
                with open(sft_output_file, sft_mode) as f:
                    f.write(json.dumps(sft_record, ensure_ascii=False) + '\n')

                print(f"  ✓ 已追加到SFT文件")

            # API限速（稍微等待一下，防止触发代理商的并发限制）
            time.sleep(1)

        except Exception as e:
            err_msg = str(e)
            print(f"  错误：处理 {case_id} 时出错: {err_msg}")

            if "429" in err_msg:
                print("  🚨 触发频率限制 (429)，强制暂停 15 秒...")
                time.sleep(15)
            else:
                # 其他错误（如暂时的网络波动）也稍作等待
                time.sleep(2)

            continue

    # 结果已在生成过程中实时保存，这里只做统计
    print(f"\n本次生成了 {len(results)} 个思维链响应")
    if len(results) > 0:
        print(f"已保存到: {output_file}")
        if sft_output_file:
            print(f"已同步追加到SFT文件: {sft_output_file}")

    # 统计所有已生成的数据（优先统计SFT文件）
    if sft_output_file and Path(sft_output_file).exists():
        print(f"\n读取SFT文件进行统计: {sft_output_file}")
        sft_records = []
        with open(sft_output_file, 'r') as f:
            for line in f:
                if line.strip():
                    sft_records.append(json.loads(line))

        print(f"\n总计SFT文件中有 {len(sft_records)} 个cases")

        if sft_records:
            prompt_lengths = [len(r['prompt']) for r in sft_records]
            response_lengths = [len(r['response']) for r in sft_records]

            print(f"\n统计:")
            print(f"  平均Prompt长度: {sum(prompt_lengths)/len(prompt_lengths):.0f} chars")
            print(f"  平均Response长度: {sum(response_lengths)/len(response_lengths):.0f} chars")
    else:
        # 如果没有SFT文件，统计CoT文件
        print("\n读取CoT文件进行统计...")
        all_results = []
        if Path(output_file).exists():
            with open(output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        all_results.append(json.loads(line))

        print(f"\n总计完成 {len(all_results)} 个cases的思维链生成")

        if all_results:
            thinking_lengths = [len(r['parsed_response'].get('thinking', '')) for r in all_results]
            insights_lengths = [len(r['parsed_response'].get('insights', '')) for r in all_results]
            etiology_lengths = [len(r['parsed_response'].get('etiology', '')) for r in all_results]
            recommendations_lengths = [len(r['parsed_response'].get('recommendations', '')) for r in all_results]
            total_lengths = [len(r['raw_response']) for r in all_results]

            print(f"\n统计:")
            print(f"  平均思维链长度: {sum(thinking_lengths)/len(thinking_lengths):.0f} chars")
            print(f"  平均insights长度: {sum(insights_lengths)/len(insights_lengths):.0f} chars")
            print(f"  平均etiology长度: {sum(etiology_lengths)/len(etiology_lengths):.0f} chars")
            print(f"  平均recommendations长度: {sum(recommendations_lengths)/len(recommendations_lengths):.0f} chars")
            print(f"  平均总响应长度: {sum(total_lengths)/len(total_lengths):.0f} chars")

    return results


def convert_to_sft_format(cot_data_file, output_file, append_mode=False):
    """
    将思维链数据转换为SFT训练格式

    Args:
        cot_data_file: CoT数据文件
        output_file: SFT输出文件
        append_mode: 是否追加模式（默认False，覆盖写入）
    """
    import re

    print(f"\n转换 {cot_data_file} 为SFT格式...")

    # 如果是追加模式，先读取已有的SFT case_ids
    existing_sft_ids = set()
    if append_mode and Path(output_file).exists():
        with open(output_file, 'r') as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    existing_sft_ids.add(record['case_study_id'])
        print(f"  SFT文件已有 {len(existing_sft_ids)} 个cases，将追加新的cases")

    with open(cot_data_file, 'r') as f:
        cot_data = [json.loads(line) for line in f if line.strip()]

    sft_records = []

    for record in cot_data:
        case_id = record['case_study_id']

        # 追加模式下跳过已存在的cases
        if append_mode and case_id in existing_sft_ids:
            continue

        # 使用base_input作为prompt（包含用户信息和睡眠数据）
        prompt = record['base_input']

        # 使用完整的raw_response作为response
        response = record['raw_response']

        # 清理Markdown代码块标记
        response = re.sub(r'^```[\w]*\n?', '', response.strip())
        response = re.sub(r'\n?```$', '', response.strip())

        sft_records.append({
            'case_study_id': case_id,
            'prompt': prompt,
            'response': response
        })

    # 保存
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    file_mode = 'a' if append_mode else 'w'
    with open(output_file, file_mode) as f:
        for record in sft_records:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f"保存了 {len(sft_records)} 条SFT记录到 {output_file}")

    # 显示一个示例
    if sft_records:
        example = sft_records[0]
        print(f"\n示例SFT记录:")
        print(f"  Case ID: {example['case_study_id']}")
        print(f"  Prompt长度: {len(example['prompt'])} chars")
        print(f"  Response长度: {len(example['response'])} chars")
        print(f"\n  Prompt前200字符:")
        print(f"  {example['prompt'][:200]}")
        print(f"\n  Response前300字符:")
        print(f"  {example['response'][:300]}")

        # 验证是否成功清理了代码块标记
        if example['response'].startswith('```'):
            print(f"\n  警告：Response仍包含代码块标记！")
        else:
            print(f"\n  ✓ Response已清理代码块标记")

    return sft_records


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate CoT training data using Gemini')
    parser.add_argument('--input_file', required=True,
                        help='Input JSONL file (e.g., sleep_train_with_ids.jsonl)')
    parser.add_argument('--output_file', required=True,
                        help='Output JSONL file for CoT responses')
    parser.add_argument('--model', default='[次]gemini-3-pro-preview',
                        help='Gemini model name (default: [次]gemini-3-pro-preview)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--api_key', default=None,
                        help='API key (optional, for changing API)')
    parser.add_argument('--base_url', default=None,
                        help='API base URL (optional, for changing API)')
    parser.add_argument('--convert_to_sft', action='store_true',
                        help='Also convert to SFT format')
    parser.add_argument('--sft_output_file', default=None,
                        help='Output file for SFT format')

    args = parser.parse_args()

    # 确定SFT输出文件
    sft_output = None
    if args.convert_to_sft:
        sft_output = args.sft_output_file or args.output_file.replace('.jsonl', '_sft.jsonl')

    # 生成思维链数据（如果指定了convert_to_sft，会实时转换并写入）
    results = generate_cot_training_data(
        input_file=args.input_file,
        output_file=args.output_file,
        model=args.model,
        max_samples=args.max_samples,
        api_key=args.api_key,
        base_url=args.base_url,
        sft_output_file=sft_output
    )

    # 如果需要转换SFT但没有在生成过程中完成（例如之前已经生成了CoT数据）
    if args.convert_to_sft and len(results) == 0:
        print(f"\n所有cases已处理完成，检查是否需要补充转换SFT格式...")
        if Path(args.output_file).exists():
            # 检查CoT文件和SFT文件的差异
            cot_case_ids = set()
            with open(args.output_file, 'r') as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        cot_case_ids.add(record['case_study_id'])

            sft_case_ids = set()
            if Path(sft_output).exists():
                with open(sft_output, 'r') as f:
                    for line in f:
                        if line.strip():
                            record = json.loads(line)
                            sft_case_ids.add(record['case_study_id'])

            missing = cot_case_ids - sft_case_ids
            if missing:
                print(f"发现 {len(missing)} 个cases在CoT文件中但不在SFT文件中，进行补充转换...")
                convert_to_sft_format(args.output_file, sft_output, append_mode=True)

    if args.convert_to_sft:
        print(f"\n下一步:")
        print(f"  1. 检查生成的思维链数据: {args.output_file}")
        print(f"  2. 查看SFT格式数据: {sft_output}")
        print(f"  3. 转换为Parquet格式用于训练")
        print(f"  4. 使用verl训练Llama3")


if __name__ == '__main__':
    main()
