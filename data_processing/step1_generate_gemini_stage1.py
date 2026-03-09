#!/usr/bin/env python3
"""
Step 1: 使用Gemini 3 Pro生成Stage 1 - Insights
输入：sleep_holdout_cases.json
输出：gemini_holdout_stage1.json
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Task instructions
INSIGHTS_INSTRUCTION = """Instruction: List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn't fit the above categories. - For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. - For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. - For Alertness, consider the number of naps and nap length. - For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. - For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. - For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Sleep insights report"""

def create_api_client(api_key=None, base_url=None):
    """创建OpenAI兼容的API客户端（Gemini 3 Pro）"""
    client = OpenAI(
        api_key=api_key or "sk-QHWpCYbhD79wjJPFc5U0VvJqYk5CYEMq32g5izTLzirGIQXQ",
        base_url=base_url or "https://once.novai.su/v1"
    )
    return client

def generate_response(client, prompt, model="[次]gemini-3-pro-preview", max_tokens=2048):
    """使用Gemini API生成回复"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.0,  # 贪婪解码，保证可复现
            stream=False  # 关闭流式传输
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API调用错误: {e}")
        raise

def generate_stage1_insights(cases_file, api_key, output_file, model="[次]gemini-3-pro-preview", base_url=None, max_samples=None):
    """生成所有cases的insights"""

    # 加载cases
    print(f"Loading cases from {cases_file}...")
    with open(cases_file, 'r') as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases")

    # 限制样本数（用于测试）
    if max_samples:
        case_ids = sorted(list(cases.keys()))[:max_samples]
        print(f"Testing with first {max_samples} case(s)")
    else:
        case_ids = sorted(cases.keys())

    # 创建API客户端
    print(f"Creating API client for model: {model}")
    client = create_api_client(api_key=api_key, base_url=base_url)

    # 生成insights
    results = {}

    for case_id in tqdm(case_ids, desc="Generating Insights"):
        case_data = cases[case_id]

        # 获取base_input（所有tasks共享相同的基础输入）
        base_input = case_data['insights']['base_input']

        # 构造insights prompt
        insights_prompt = f"{base_input} {INSIGHTS_INSTRUCTION}"

        print(f"\nProcessing {case_id}...")
        print(f"  Base input length: {len(base_input)} chars")
        print(f"  Prompt length: {len(insights_prompt)} chars")

        # 生成insights
        try:
            insights_output = generate_response(client, insights_prompt, model=model)
            print(f"  Generated insights length: {len(insights_output)} chars")
        except Exception as e:
            print(f"  Error generating insights for {case_id}: {e}")
            continue

        # 保存结果
        results[case_id] = {
            'base_input': base_input,
            'insights_prompt': insights_prompt,
            'insights_output': insights_output
        }

        # API限速（避免触发rate limit）
        time.sleep(1)

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的insights")
    print(f"保存到: {output_file}")

    # 统计
    if results:
        output_lengths = [len(r['insights_output']) for r in results.values()]
        avg_length = sum(output_lengths) / len(output_lengths)
        print(f"\n统计:")
        print(f"  平均输出长度: {avg_length:.0f} chars")
        print(f"  最短输出: {min(output_lengths)} chars")
        print(f"  最长输出: {max(output_lengths)} chars")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Stage 1 Insights with Gemini')
    parser.add_argument('--cases_file', default='dataset/raw/sleep_holdout_cases.json',
                        help='Input cases JSON file')
    parser.add_argument('--api_key', default=None,
                        help='Gemini API key (optional, has default)')
    parser.add_argument('--model', default='[次]gemini-3-pro-preview',
                        help='Model name (default: [次]gemini-3-pro-preview)')
    parser.add_argument('--base_url', default=None,
                        help='API base URL (optional)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process (for testing)')
    parser.add_argument('--output_file', default='results/gemini_holdout_stage1.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage1_insights(
        cases_file=args.cases_file,
        api_key=args.api_key,
        output_file=args.output_file,
        model=args.model,
        base_url=args.base_url,
        max_samples=args.max_samples
    )

    print("\n下一步: 运行 step2_generate_gemini_stage2.py 生成etiology")

if __name__ == '__main__':
    main()
