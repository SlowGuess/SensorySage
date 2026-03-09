#!/usr/bin/env python3
"""
Step 1: 使用Llama3生成Stage 1 - Insights
输入：sleep_holdout_cases.json
输出：llama3_holdout_stage1.json
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Task instructions (从原始数据处理脚本中提取)
INSIGHTS_INSTRUCTION = """Instruction: List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn't fit the above categories. - For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. - For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. - For Alertness, consider the number of naps and nap length. - For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. - For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. - For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Sleep insights report"""

def load_model_and_tokenizer(checkpoint_path):
    """加载转换后的HuggingFace格式模型"""
    print(f"Loading model from {checkpoint_path}...")

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=2048):
    """生成回复（使用token-level extraction避免截断）"""
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # Token count

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # 贪婪解码，保证可复现
            pad_token_id=tokenizer.eos_token_id,
        )

    # Token-level extraction
    output_tokens = outputs[0][input_length:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    return response

def generate_stage1_insights(cases_file, checkpoint_path, output_file):
    """生成所有cases的insights"""

    # 加载cases
    print(f"Loading cases from {cases_file}...")
    with open(cases_file, 'r') as f:
        cases = json.load(f)

    print(f"Loaded {len(cases)} cases")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # 生成insights
    results = {}

    for case_id in tqdm(sorted(cases.keys()), desc="Generating Insights"):
        case_data = cases[case_id]

        # 获取base_input（所有tasks共享相同的基础输入）
        base_input = case_data['insights']['base_input']

        # 构造insights prompt
        insights_prompt = f"{base_input} {INSIGHTS_INSTRUCTION}"

        print(f"\nProcessing {case_id}...")
        print(f"  Base input length: {len(base_input)} chars")
        print(f"  Prompt length: {len(insights_prompt)} chars")

        # 生成insights
        insights_output = generate_response(model, tokenizer, insights_prompt)

        print(f"  Generated insights length: {len(insights_output)} chars")

        # 保存结果
        results[case_id] = {
            'base_input': base_input,
            'insights_prompt': insights_prompt,
            'insights_output': insights_output
        }

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的insights")
    print(f"保存到: {output_file}")

    # 统计
    output_lengths = [len(r['insights_output']) for r in results.values()]
    avg_length = sum(output_lengths) / len(output_lengths)
    print(f"\n统计:")
    print(f"  平均输出长度: {avg_length:.0f} chars")
    print(f"  最短输出: {min(output_lengths)} chars")
    print(f"  最长输出: {max(output_lengths)} chars")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Stage 1 Insights')
    parser.add_argument('--cases_file', default='dataset/raw/sleep_holdout_cases.json',
                        help='Input cases JSON file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to converted HuggingFace checkpoint')
    parser.add_argument('--output_file', default='results/llama3_holdout_stage1.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage1_insights(
        cases_file=args.cases_file,
        checkpoint_path=args.checkpoint,
        output_file=args.output_file
    )

    print("\n下一步: 运行 step2_generate_llama3_stage2.py 生成etiology")

if __name__ == '__main__':
    main()
