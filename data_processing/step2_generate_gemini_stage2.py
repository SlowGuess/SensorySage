#!/usr/bin/env python3
"""
Step 2: 使用Gemini 3 Pro生成Stage 2 - Etiology
输入：gemini_holdout_stage1.json (包含模型生成的insights)
输出：gemini_holdout_stage2.json
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Etiology instruction
ETIOLOGY_INSTRUCTION = """Instruction: What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, "consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive". Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Causes report"""

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
            temperature=0.0,
            stream=False
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"API调用错误: {e}")
        raise

def generate_stage2_etiology(stage1_file, api_key, output_file, model="[次]gemini-3-pro-preview", base_url=None, max_samples=None):
    """生成所有cases的etiology（基于模型生成的insights）"""

    # 加载stage1结果
    print(f"Loading stage1 results from {stage1_file}...")
    with open(stage1_file, 'r') as f:
        stage1_results = json.load(f)

    print(f"Loaded {len(stage1_results)} cases with insights")

    # 限制样本数（用于测试）
    if max_samples:
        case_ids = sorted(list(stage1_results.keys()))[:max_samples]
        print(f"Testing with first {max_samples} case(s)")
    else:
        case_ids = sorted(stage1_results.keys())

    # 创建API客户端
    print(f"Creating API client for model: {model}")
    client = create_api_client(api_key=api_key, base_url=base_url)

    # 生成etiology
    results = {}

    for case_id in tqdm(case_ids, desc="Generating Etiology"):
        stage1_data = stage1_results[case_id]

        base_input = stage1_data['base_input']
        model_insights = stage1_data['insights_output']  # 使用模型生成的insights

        # 构造etiology prompt（拼接模型insights）
        etiology_prompt = f"{base_input} Based on the data, we can get the following insights: {model_insights} {ETIOLOGY_INSTRUCTION}"

        print(f"\nProcessing {case_id}...")
        print(f"  Insights length: {len(model_insights)} chars")
        print(f"  Prompt length: {len(etiology_prompt)} chars")

        # 生成etiology
        try:
            etiology_output = generate_response(client, etiology_prompt, model=model)
            print(f"  Generated etiology length: {len(etiology_output)} chars")
        except Exception as e:
            print(f"  Error generating etiology for {case_id}: {e}")
            continue

        # 保存结果（累积前面的信息）
        results[case_id] = {
            'base_input': base_input,
            'insights_prompt': stage1_data['insights_prompt'],
            'insights_output': model_insights,
            'etiology_prompt': etiology_prompt,
            'etiology_output': etiology_output
        }

        # API限速
        time.sleep(1)

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的etiology")
    print(f"保存到: {output_file}")

    # 统计
    if results:
        output_lengths = [len(r['etiology_output']) for r in results.values()]
        avg_length = sum(output_lengths) / len(output_lengths)
        print(f"\n统计:")
        print(f"  平均输出长度: {avg_length:.0f} chars")
        print(f"  最短输出: {min(output_lengths)} chars")
        print(f"  最长输出: {max(output_lengths)} chars")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Stage 2 Etiology with Gemini')
    parser.add_argument('--stage1_file', default='results/gemini_holdout_stage1.json',
                        help='Input stage1 JSON file')
    parser.add_argument('--api_key', default=None,
                        help='Gemini API key (optional, has default)')
    parser.add_argument('--model', default='[次]gemini-3-pro-preview',
                        help='Model name (default: [次]gemini-3-pro-preview)')
    parser.add_argument('--base_url', default=None,
                        help='API base URL (optional)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process (for testing)')
    parser.add_argument('--output_file', default='results/gemini_holdout_stage2.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage2_etiology(
        stage1_file=args.stage1_file,
        api_key=args.api_key,
        output_file=args.output_file,
        model=args.model,
        base_url=args.base_url,
        max_samples=args.max_samples
    )

    print("\n下一步: 运行 step3_generate_gemini_stage3.py 生成recommendations")

if __name__ == '__main__':
    main()
