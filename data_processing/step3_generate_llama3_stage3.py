#!/usr/bin/env python3
"""
Step 3: 使用Llama3生成Stage 3 - Recommendations
输入：llama3_holdout_stage2.json (包含模型生成的insights和etiology)
输出：llama3_holdout_stage3_complete.json
"""

import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Recommendations instruction
RECOMMENDATIONS_INSTRUCTION = """Instruction: What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, "Recommendations to address Circadian rhythm". Tie recommendations to user's sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee's lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Recommendations report"""

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

def generate_stage3_recommendations(stage2_file, checkpoint_path, output_file):
    """生成所有cases的recommendations（基于模型生成的insights和etiology）"""

    # 加载stage2结果
    print(f"Loading stage2 results from {stage2_file}...")
    with open(stage2_file, 'r') as f:
        stage2_results = json.load(f)

    print(f"Loaded {len(stage2_results)} cases with insights and etiology")

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)

    # 生成recommendations
    results = {}

    for case_id in tqdm(sorted(stage2_results.keys()), desc="Generating Recommendations"):
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
        recommendations_output = generate_response(model, tokenizer, recommendations_prompt)

        print(f"  Generated recommendations length: {len(recommendations_output)} chars")

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

    # 保存结果
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n完成！生成了 {len(results)} 个cases的完整三阶段输出")
    print(f"保存到: {output_file}")

    # 统计
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

    parser = argparse.ArgumentParser(description='Generate Stage 3 Recommendations')
    parser.add_argument('--stage2_file', default='results/llama3_holdout_stage2.json',
                        help='Input stage2 JSON file')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to converted HuggingFace checkpoint')
    parser.add_argument('--output_file', default='results/llama3_holdout_stage3_complete.json',
                        help='Output JSON file')
    args = parser.parse_args()

    results = generate_stage3_recommendations(
        stage2_file=args.stage2_file,
        checkpoint_path=args.checkpoint,
        output_file=args.output_file
    )

    print("\n下一步: 运行 step4_convert_to_standard_jsonl.py 转换为标准格式")

if __name__ == '__main__':
    main()
