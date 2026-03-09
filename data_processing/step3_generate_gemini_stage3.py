#!/usr/bin/env python3
"""
Step 3: 使用Gemini 3 Pro生成Stage 3 - Recommendations
输入：gemini_holdout_stage2.json (包含模型生成的insights和etiology)
输出：gemini_holdout_stage3_complete.json
"""

import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import OpenAI

# Recommendations instruction
RECOMMENDATIONS_INSTRUCTION = """Instruction: What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, "Recommendations to address Circadian rhythm". Tie recommendations to user's sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee's lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Recommendations report"""

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

def generate_stage3_recommendations(stage2_file, api_key, output_file, model="[次]gemini-3-pro-preview", base_url=None, max_samples=None):
    """生成所有cases的recommendations（基于模型生成的insights和etiology）"""

    # 加载stage2结果
    print(f"Loading stage2 results from {stage2_file}...")
    with open(stage2_file, 'r') as f:
        stage2_results = json.load(f)

    print(f"Loaded {len(stage2_results)} cases with insights and etiology")

    # 限制样本数（用于测试）
    if max_samples:
        case_ids = sorted(list(stage2_results.keys()))[:max_samples]
        print(f"Testing with first {max_samples} case(s)")
    else:
        case_ids = sorted(stage2_results.keys())

    # 创建API客户端
    print(f"Creating API client for model: {model}")
    client = create_api_client(api_key=api_key, base_url=base_url)

    # 生成recommendations
    results = {}

    for case_id in tqdm(case_ids, desc="Generating Recommendations"):
        stage2_data = stage2_results[case_id]

        base_input = stage2_data['base_input']
        model_insights = stage2_data['insights_output']    # 模型生成的insights
        model_etiology = stage2_data['etiology_output']    # 模型生成的etiology

        # 构造recommendations prompt（拼接模型insights + etiology）
        recommendations_prompt = f"{base_input} Based on the data, we can get the following insights: {model_insights} Causes: {model_etiology} {RECOMMENDATIONS_INSTRUCTION}"

        print(f"\nProcessing {case_id}...")
        print(f"  Insights length: {len(model_insights)} chars")
        print(f"  Etiology length: {len(model_etiology)} chars")
        print(f"  Prompt length: {len(recommendations_prompt)} chars")

        # 生成recommendations
        try:
            recommendations_output = generate_response(client, recommendations_prompt, model=model)
            print(f"  Generated recommendations length: {len(recommendations_output)} chars")
        except Exception as e:
            print(f"  Error generating recommendations for {case_id}: {e}")
            continue

        # 保存完整结果（包含所有3个stages）
        results[case_id] = {
            'base_input': base_input,
            'insights_prompt': stage2_data['insights_prompt'],
            'insights_output': model_insights,
            'etiology_prompt': stage2_data['etiology_prompt'],
            'etiology_output': model_etiology,
            'recommendations_prompt': recommendations_prompt,
            'recommendations_output': recommendations_output
        }

        # API限速
        time.sleep(1)

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的完整三阶段输出")
    print(f"保存到: {output_file}")

    # 统计
    if results:
        print(f"\n统计:")
        for stage in ['insights', 'etiology', 'recommendations']:
            output_key = f'{stage}_output'
            lengths = [len(r[output_key]) for r in results.values()]
            avg_length = sum(lengths) / len(lengths)
            print(f"  {stage.capitalize()}:")
            print(f"    平均长度: {avg_length:.0f} chars")
            print(f"    最短: {min(lengths)} chars")
            print(f"    最长: {max(lengths)} chars")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Stage 3 Recommendations with Gemini')
    parser.add_argument('--stage2_file', default='results/gemini_holdout_stage2.json',
                        help='Input stage2 JSON file')
    parser.add_argument('--api_key', default=None,
                        help='Gemini API key (optional, has default)')
    parser.add_argument('--model', default='[次]gemini-3-pro-preview',
                        help='Model name (default: [次]gemini-3-pro-preview)')
    parser.add_argument('--base_url', default=None,
                        help='API base URL (optional)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max number of samples to process (for testing)')
    parser.add_argument('--output_file', default='results/gemini_holdout_stage3_complete.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage3_recommendations(
        stage2_file=args.stage2_file,
        api_key=args.api_key,
        output_file=args.output_file,
        model=args.model,
        base_url=args.base_url,
        max_samples=args.max_samples
    )

    print("\n下一步: 运行 step4_convert_gemini_to_jsonl.py 转换为标准格式")

if __name__ == '__main__':
    main()
