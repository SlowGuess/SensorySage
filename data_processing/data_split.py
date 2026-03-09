import json
import random
import os

def create_dataset_for_finetuning_and_eval(input_file):
    # ==============================================================================
    # 1. 定义精确的 Prompt 模板 (来自原文)
    # ==============================================================================
    EXPERT_PREFIX = "You are a sleep medicine expert. You are given the following sleep data."

    INSIGHTS_INSTRUCTION = """Instruction: List the most important insights. Identify all of the patterns of data that are likely out of the preferred range. Make sure to consider various sleep health dimensions: Routine, Sleep Quality, Alertness, Timing, Efficiency, and Duration. Add a heading for each dimension. Optionally (only do this if extremely important) add a heading called Other for anything else that doesn’t fit the above categories. - For Routine, consider the average bedtime, wake time, midsleep point and standard deviations of these, focus on the consistency of the routine, not timing. - For Sleep Quality, consider light sleep duration, deep sleep duration, REM sleep duration, sleep score, restlessness score, time to quality sleep, and wake time after sleep onset. - For Alertness, consider the number of naps and nap length. - For Timing, consider midsleep point, bedtime, wake time, make any comments on weekend vs. workday. - For Efficiency, consider sleep efficiency, wake time after sleep onset, and time to quality sleep, describe how they compare to similar users. - For Duration, consider average sleep duration, weekend vs. workday sleep durations and standard deviations, describe how they compare to similar users. When determining whether a metric is normal or abnormal, always provide the corresponding percentile. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Sleep insights report"""

    ETIOLOGY_INSTRUCTION = """Instruction: What are the underlying causes? Make sure to consider the following causes: Circadian rhythm, Homeostatic drive, Psychophysiologic hyperarousal, and Extrinsic factors. Order the causes from most to least relevant. Identify the likelihood of the causes (e.g. unlikely, possible, very likely). Cite relevant data and insights, for example, “consistently low sleep efficiency despite normal sleep durations suggests low homeostatic drive”. Avoid diagnosing health conditions. Avoid providing recommendations. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Causes report"""

    RECOMMENDATION_INSTRUCTION = """Instruction: What recommendation(s) can you provide to help this user improve their sleep? Tie recommendations to the very likely and possible causes, for example, “Recommendations to address Circadian rhythm”. Tie recommendations to user’s sleep data such as average bedtime, average wake time, and number of naps, and recommend a goal bedtime and wake time based on their data. The recommendations should be time-bound, for example for the next week or the next month. Write one short question to ask the user in order to better understand their sleep. Avoid assumptions regarding the trainee’s lifestyle or behavioral choices. Avoid generic statements. Avoid incorrect knowledge, inconsistencies and contradictions. Don’t mention “the user”. Talk like you’re speaking directly to someone. Be concise. # Recommendations report"""

    # ==============================================================================
    # 2. 读取与处理
    # ==============================================================================
    data_buffer = {}

    try:
        print(f"Reading file: {input_file} ...")
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError: continue

                # 提取元数据
                split = record.get('split', 'unknown')
                case_id = record.get('case_study_id', 'unknown_id') # 获取 ID
                
                # 提取内容
                raw_input = record.get('input', '')
                gt_insights = record.get('insight_output', '')
                gt_etiology = record.get('etiology_output', '')
                gt_recommendations = record.get('recommendation_output', '')

                if split not in data_buffer:
                    data_buffer[split] = []

                # --- Task 1: Insights ---
                prompt_1 = f"{EXPERT_PREFIX} {raw_input} {INSIGHTS_INSTRUCTION}"
                data_buffer[split].append({
                    "case_study_id": case_id,       # 保留 ID
                    "task_type": "insights",        # 标记任务类型
                    "prompt": prompt_1,
                    "completion": gt_insights
                })

                # --- Task 2: Etiology ---
                prompt_2 = f"{EXPERT_PREFIX} {raw_input} Based on the data, we can get the following insights: {gt_insights} {ETIOLOGY_INSTRUCTION}"
                data_buffer[split].append({
                    "case_study_id": case_id,
                    "task_type": "etiology",
                    "prompt": prompt_2,
                    "completion": gt_etiology
                })

                # --- Task 3: Recommendations ---
                prompt_3 = f"{EXPERT_PREFIX} {raw_input} Based on the data, we can get the following insights: {gt_insights} Causes: {gt_etiology} {RECOMMENDATION_INSTRUCTION}"
                data_buffer[split].append({
                    "case_study_id": case_id,
                    "task_type": "recommendations",
                    "prompt": prompt_3,
                    "completion": gt_recommendations
                })

    except FileNotFoundError:
        print(f"Error: File {input_file} not found.")
        return

    # ==============================================================================
    # 3. 打乱并写入
    # ==============================================================================
    print("Shuffling and writing data...")
    random.seed(42)

    for split_name, samples in data_buffer.items():
        random.shuffle(samples)
        
        output_filename = f"sleep_{split_name}_with_ids.jsonl"
        with open(output_filename, 'w', encoding='utf-8') as out_f:
            for s in samples:
                out_f.write(json.dumps(s, ensure_ascii=False) + '\n')
        
        print(f"  -> Generated {output_filename}: {len(samples)} samples (Shuffled)")

# 执行脚本
input_filename = 'dataset/raw/sleep_case_studies.all.jsonl'
create_dataset_for_finetuning_and_eval(input_filename)