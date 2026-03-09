#!/usr/bin/env python3
"""
级联推理脚本 - 修复response提取问题
使用token-level截取而不是字符串长度
"""

import json
import argparse
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(checkpoint_path):
    """加载模型"""
    print(f"Loading tokenizer from {checkpoint_path}...")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, legacy=False)

    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    print("Model loaded successfully!")
    return model, tokenizer


def generate_response(model, tokenizer, prompt, max_new_tokens=2048):
    """生成回复 - 修复版：使用token-level截取"""
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    input_length = inputs['input_ids'].shape[1]  # 记录输入token数量

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 修复：使用token数量来截取，只decode生成的部分
    output_tokens = outputs[0][input_length:]
    response = tokenizer.decode(output_tokens, skip_special_tokens=True).strip()

    return response


def extract_base_input_from_prompt(prompt):
    """从prompt中提取原始传感器数据"""
    instruction_pos = prompt.find("Instruction:")

    if instruction_pos == -1:
        print(f"Warning: Could not find 'Instruction:' in prompt")
        return prompt

    base_input = prompt[:instruction_pos].strip()
    return base_input


def cascade_inference(model, tokenizer, base_input, task_instructions):
    """级联推理：Insights → Etiology → Recommendations"""
    results = {}

    # Step 1: 生成 Insights
    print("  Generating Insights...")
    insights_prompt = f"{base_input} {task_instructions['insights']}"
    insights_output = generate_response(model, tokenizer, insights_prompt)
    results['insights'] = insights_output

    # Step 2: 生成 Etiology（使用生成的Insights）
    print("  Generating Etiology...")
    etiology_prompt = f"{base_input} Based on the data, we can get the following insights: {insights_output} {task_instructions['etiology']}"
    etiology_output = generate_response(model, tokenizer, etiology_prompt)
    results['etiology'] = etiology_output

    # Step 3: 生成 Recommendations（使用生成的Insights和Etiology）
    print("  Generating Recommendations...")
    recommendations_prompt = f"{base_input} Based on the data, we can get the following insights: {insights_output} Causes: {etiology_output} {task_instructions['recommendations']}"
    recommendations_output = generate_response(model, tokenizer, recommendations_prompt)
    results['recommendations'] = recommendations_output

    return results


def main():
    parser = argparse.ArgumentParser(description='Cascade inference for sleep case studies')
    parser.add_argument('--checkpoint', required=True, help='Path to converted HuggingFace checkpoint')
    parser.add_argument('--test_file', required=True, help='Path to test JSONL file')
    parser.add_argument('--output_file', required=True, help='Path to save predictions')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # 任务指令
    EXPERT_PREFIX = "You are a sleep medicine expert. You are given the following sleep data."

    task_instructions = {
        'insights': """Instruction: List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn't fit the above categories. - For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. - For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. - For Alertness, consider the number of naps and nap length. - For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. - For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. - For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Sleep insights report""",

        'etiology': """Instruction: What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, "consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive". Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Causes report""",

        'recommendations': """Instruction: What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, "Recommendations to address Circadian rhythm". Tie recommendations to user's sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee's lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don't mention "the user". Talk like you're speaking directly to someone. Be concise. # Recommendations report"""
    }

    # 读取测试数据
    print(f"Loading test data from {args.test_file}...")
    test_cases = {}

    with open(args.test_file, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            case_id = record['case_study_id']

            if case_id not in test_cases:
                test_cases[case_id] = {'base_input': None, 'ground_truth': {}}

            if test_cases[case_id]['base_input'] is None:
                prompt = record['prompt']
                base_input = extract_base_input_from_prompt(prompt)
                test_cases[case_id]['base_input'] = base_input

            task_type = record['task_type']
            test_cases[case_id]['ground_truth'][task_type] = record['completion']

    print(f"Loaded {len(test_cases)} unique test cases")

    # 推理
    predictions = []
    case_ids = list(test_cases.keys())

    if args.max_samples:
        case_ids = case_ids[:args.max_samples]
        print(f"Limiting to first {args.max_samples} cases for testing")

    print(f"\nStarting cascade inference on {len(case_ids)} cases...")

    for case_id in tqdm(case_ids, desc="Inference"):
        case_data = test_cases[case_id]
        print(f"\nProcessing {case_id}...")

        results = cascade_inference(
            model,
            tokenizer,
            case_data['base_input'],
            task_instructions
        )

        predictions.append({
            'case_study_id': case_id,
            'predictions': results,
            'ground_truth': case_data['ground_truth']
        })

    # 保存结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving predictions to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')

    print("\n" + "="*80)
    print("Inference completed!")
    print("="*80)
    print(f"Total cases processed: {len(predictions)}")
    print(f"Output file: {args.output_file}")


if __name__ == '__main__':
    main()
