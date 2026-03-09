#!/usr/bin/env python3
"""
Step 2: 使用Llama3生成Stage 2 - Etiology
输入：llama3_holdout_stage1.json (包含模型生成的insights)
输出：llama3_holdout_stage2.json
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Etiology instruction
ETIOLOGY_INSTRUCTION = """Instruction: What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, "consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive". Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Causes report"""

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
            do_sample=False,  # 贪婪解码
            pad_token_id=tokenizer.eos_token_id,
        )

    # Token-level extraction
    output_tokens = outputs[0][input_length:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    return response

def generate_stage2_etiology(stage1_file, checkpoint_path, output_file):
    """生成所有cases的etiology（基于模型生成的insights）"""

    # 加载stage1结果
    print(f"Loading stage1 results from {stage1_file}...")
    with open(stage1_file, 'r') as f:
        stage1_results = json.load(f)

    print(f"Loaded {len(stage1_results)} cases with insights")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # 生成etiology
    results = {}

    for case_id in tqdm(sorted(stage1_results.keys()), desc="Generating Etiology"):
        stage1_data = stage1_results[case_id]

        base_input = stage1_data['base_input']
        model_insights = stage1_data['insights_output']  # 使用模型生成的insights

        # 构造etiology prompt（拼接模型insights）
        etiology_prompt = f"{base_input} Based on the data, we can get the following insights: {model_insights} {ETIOLOGY_INSTRUCTION}"

        print(f"\nProcessing {case_id}...")
        print(f"  Insights length: {len(model_insights)} chars")
        print(f"  Prompt length: {len(etiology_prompt)} chars")

        # 生成etiology
        etiology_output = generate_response(model, tokenizer, etiology_prompt)

        print(f"  Generated etiology length: {len(etiology_output)} chars")

        # 保存结果（累积前面的信息）
        results[case_id] = {
            'base_input': base_input,
            'insights_prompt': stage1_data['insights_prompt'],
            'insights_output': model_insights,
            'etiology_prompt': etiology_prompt,
            'etiology_output': etiology_output
        }

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的etiology")
    print(f"保存到: {output_file}")

    # 统计
    output_lengths = [len(r['etiology_output']) for r in results.values()]
    avg_length = sum(output_lengths) / len(output_lengths)
    print(f"\n统计:")
    print(f"  平均输出长度: {avg_length:.0f} chars")
    print(f"  最短输出: {min(output_lengths)} chars")
    print(f"  最长输出: {max(output_lengths)} chars")

    return results

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Generate Stage 2 Etiology')
    parser.add_argument('--stage1_file', default='results/llama3_holdout_stage1.json',
                        help='Input stage1 JSON file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to converted HuggingFace checkpoint')
    parser.add_argument('--output_file', default='results/llama3_holdout_stage2.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage2_etiology(
        stage1_file=args.stage1_file,
        checkpoint_path=args.checkpoint,
        output_file=args.output_file
    )

    print("\n下一步: 运行 step3_generate_llama3_stage3.py 生成recommendations")

if __name__ == '__main__':
    main()
