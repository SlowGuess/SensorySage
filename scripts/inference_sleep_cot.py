#!/usr/bin/env python3
"""
CoT一体化推理脚本 - 使用transformers原生API（无需vLLM）
一次性生成完整的 <THINKING> + <INSIGHTS> + <ETIOLOGY> + <RECOMMENDATIONS>
"""

import json
import argparse
import re
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(checkpoint_path):
    """加载模型"""
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


def generate_cot_response(model, tokenizer, prompt, max_new_tokens=8192):
    """
    一体化生成CoT响应
    """
    # 应用chat模板
    messages = [{"role": "user", "content": prompt}]

    input_text = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False
    )

    # 【修复 1】：获取包含 attention_mask 的完整输入
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
    
    # 获取 input_ids 的长度，用于后续剥离 prompt
    input_length = inputs.input_ids.shape[1]

    # 生成
    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,  # 【修复 2】：传入 attention_mask 消除警告并防死循环
            max_new_tokens=2048,                   # CoT 推荐 2048
            do_sample=True,                        # 开启采样
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id
        )

    # 【修复 3】：只解码新生成的 token（跳过 prompt 长度），这比字符串切片安全得多
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    return response


def parse_cot_response(response: str) -> dict:
    """
    解析CoT响应为三个section

    预期格式：
    <THINKING>...</THINKING>
    <INSIGHTS>...</INSIGHTS>
    <ETIOLOGY>...</ETIOLOGY>
    <RECOMMENDATIONS>...</RECOMMENDATIONS>
    """
    result = {
        "insights": "",
        "etiology": "",
        "recommendations": "",
        "thinking": ""  # 可选，用于调试
    }

    # 提取thinking
    thinking_match = re.search(r'<THINKING>(.*?)</THINKING>', response, re.DOTALL | re.IGNORECASE)
    if thinking_match:
        result["thinking"] = thinking_match.group(1).strip()

    # 提取insights
    insights_match = re.search(r'<INSIGHTS>(.*?)</INSIGHTS>', response, re.DOTALL | re.IGNORECASE)
    if insights_match:
        result["insights"] = insights_match.group(1).strip()
    else:
        # Fallback: 寻找</THINKING>和<ETIOLOGY>之间的内容
        fallback = re.search(r'</THINKING>\s*(.*?)<ETIOLOGY>', response, re.DOTALL | re.IGNORECASE)
        if fallback:
            result["insights"] = fallback.group(1).strip()

    # 提取etiology
    etiology_match = re.search(r'<ETIOLOGY>(.*?)</ETIOLOGY>', response, re.DOTALL | re.IGNORECASE)
    if etiology_match:
        result["etiology"] = etiology_match.group(1).strip()
    else:
        fallback = re.search(r'</INSIGHTS>\s*(.*?)<RECOMMENDATIONS>', response, re.DOTALL | re.IGNORECASE)
        if fallback:
            result["etiology"] = fallback.group(1).strip()

    # 提取recommendations
    rec_match = re.search(r'<RECOMMENDATIONS>(.*?)</RECOMMENDATIONS>', response, re.DOTALL | re.IGNORECASE)
    if rec_match:
        result["recommendations"] = rec_match.group(1).strip()
    else:
        fallback = re.search(r'</ETIOLOGY>\s*(.*?)$', response, re.DOTALL | re.IGNORECASE)
        if fallback:
            result["recommendations"] = fallback.group(1).strip()

    # 清理残留的XML标签
    for key in ["insights", "etiology", "recommendations"]:
        if result[key]:
            result[key] = re.sub(r'</?[A-Z]+>', '', result[key]).strip()

    return result


def main():
    parser = argparse.ArgumentParser(description='CoT integrated inference')
    parser.add_argument('--checkpoint', required=True, help='Path to CoT checkpoint')
    parser.add_argument('--test_file', required=True, help='Path to test JSONL file')
    parser.add_argument('--output_file', required=True, help='Path to save predictions')
    parser.add_argument('--source_data', default='data/sleep_case_studies.all.jsonl',
                       help='Source data for ground truth (optional)')
    parser.add_argument('--max_samples', type=int, default=None, help='Max samples to process')
    parser.add_argument('--max_tokens', type=int, default=8192, help='Max new tokens')
    args = parser.parse_args()

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(args.checkpoint)

    # 加载测试数据
    print(f"Loading test data from {args.test_file}...")
    test_data = []
    with open(args.test_file, 'r') as f:
        for line in f:
            test_data.append(json.loads(line))

    if args.max_samples:
        test_data = test_data[:args.max_samples]

    print(f"Loaded {len(test_data)} test cases")

    # 加载ground truth（如果有）
    ground_truth_map = {}
    if Path(args.source_data).exists():
        print(f"Loading ground truth from {args.source_data}...")
        with open(args.source_data, 'r') as f:
            for line in f:
                item = json.loads(line)
                case_id = item.get('case_study_id')
                if case_id:
                    ground_truth_map[case_id] = {
                        "insights": item.get('insight_output', ''),
                        "etiology": item.get('etiology_output', ''),
                        "recommendations": item.get('recommendation_output', '')
                    }

    # 推理
    results = []
    failed_cases = []

    for case in tqdm(test_data, desc="Generating predictions"):
        case_id = case.get('case_study_id', 'unknown')
        user_input = case.get('input', '')

        print(f"\nProcessing {case_id}...")

        try:
            # 生成完整响应
            full_response = generate_cot_response(model, tokenizer, user_input, args.max_tokens)

            # 解析为三个section
            parsed = parse_cot_response(full_response)

            # 检查解析是否成功
            if not any([parsed['insights'], parsed['etiology'], parsed['recommendations']]):
                print(f"⚠️  Failed to parse {case_id}")
                failed_cases.append(case_id)
                # 保存原始响应
                parsed = {
                    "insights": full_response[:1000],  # 截取前1000字符
                    "etiology": "Parsing failed - see insights for raw output",
                    "recommendations": "Parsing failed"
                }

            # 构建输出
            output = {
                "case_study_id": case_id,
                "predictions": {
                    "insights": parsed['insights'],
                    "etiology": parsed['etiology'],
                    "recommendations": parsed['recommendations']
                },
                "prompt": user_input
            }

            # 添加ground truth
            if case_id in ground_truth_map:
                output["ground_truth"] = ground_truth_map[case_id]

            results.append(output)

        except Exception as e:
            print(f"❌ Error processing {case_id}: {e}")
            failed_cases.append(case_id)
            continue

    # 保存结果
    print(f"\nSaving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    # 统计
    print("\n" + "="*60)
    print("Summary:")
    print("="*60)
    print(f"Total cases: {len(test_data)}")
    print(f"Successfully processed: {len(results) - len(failed_cases)}")
    print(f"Failed to parse: {len(failed_cases)}")
    if failed_cases:
        print(f"Failed IDs: {failed_cases[:10]}...")
    print("="*60)

    # 显示示例
    if results:
        print("\nSample output:")
        sample = results[0]
        print(f"Case ID: {sample['case_study_id']}")
        print(f"Insights (first 200 chars): {sample['predictions']['insights'][:200]}...")
        print(f"Etiology (first 200 chars): {sample['predictions']['etiology'][:200]}...")
        print(f"Recommendations (first 200 chars): {sample['predictions']['recommendations'][:200]}...")

    print("\n✅ Done! Now you can evaluate:")
    print(f"   python3 scripts/llm_judge_evaluation_v2.py \\")
    print(f"       --input {args.output_file} \\")
    print(f"       --output results/eval_cot.json \\")
    print(f"       --model claude-opus-4-6")


if __name__ == "__main__":
    main()
