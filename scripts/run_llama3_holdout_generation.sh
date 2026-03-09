#!/bin/bash
#
# 一键执行脚本：生成Llama3 Holdout数据
# 用法: ./run_llama3_holdout_generation.sh <checkpoint_path>
#

set -e  # 遇到错误立即退出

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <checkpoint_path>"
    echo "示例: $0 checkpoints/sleep_mixed_sft_llama3_8b/final_checkpoint"
    exit 1
fi

CHECKPOINT=$1

# 检查checkpoint是否存在
if [ ! -d "$CHECKPOINT" ]; then
    echo "错误: Checkpoint目录不存在: $CHECKPOINT"
    exit 1
fi

echo "=========================================="
echo "Llama3 Holdout数据生成"
echo "=========================================="
echo "Checkpoint: $CHECKPOINT"
echo "=========================================="

# Step 0
echo ""
echo "Step 0: 提取holdout cases..."
python scripts/step0_extract_holdout_cases.py

if [ $? -ne 0 ]; then
    echo "错误: Step 0 失败"
    exit 1
fi

# Step 1
echo ""
echo "Step 1: 生成Insights..."
python scripts/step1_generate_llama3_stage1.py \
    --cases_file dataset/raw/sleep_holdout_cases.json \
    --checkpoint $CHECKPOINT \
    --output_file results/llama3_holdout_stage1.json

if [ $? -ne 0 ]; then
    echo "错误: Step 1 失败"
    exit 1
fi

# Step 2
echo ""
echo "Step 2: 生成Etiology..."
python scripts/step2_generate_llama3_stage2.py \
    --stage1_file results/llama3_holdout_stage1.json \
    --checkpoint $CHECKPOINT \
    --output_file results/llama3_holdout_stage2.json

if [ $? -ne 0 ]; then
    echo "错误: Step 2 失败"
    exit 1
fi

# Step 3
echo ""
echo "Step 3: 生成Recommendations..."
python scripts/step3_generate_llama3_stage3.py \
    --stage2_file results/llama3_holdout_stage2.json \
    --checkpoint $CHECKPOINT \
    --output_file results/llama3_holdout_stage3_complete.json

if [ $? -ne 0 ]; then
    echo "错误: Step 3 失败"
    exit 1
fi

# Step 4
echo ""
echo "Step 4: 转换为标准JSONL格式..."
python scripts/step4_convert_to_standard_jsonl.py \
    --stage3_file results/llama3_holdout_stage3_complete.json \
    --output_file dataset/raw/sleep_holdout_llama3_with_ids.jsonl \
    --validate \
    --reference_file dataset/raw/sleep_holdout_with_ids.jsonl

if [ $? -ne 0 ]; then
    echo "错误: Step 4 失败"
    exit 1
fi

echo ""
echo "=========================================="
echo "完成！"
echo "=========================================="
echo "生成的文件: dataset/raw/sleep_holdout_llama3_with_ids.jsonl"
echo ""
echo "验证输出:"
echo "  wc -l dataset/raw/sleep_holdout_llama3_with_ids.jsonl"
echo "  head -1 dataset/raw/sleep_holdout_llama3_with_ids.jsonl | python3 -m json.tool"
echo ""
echo "下一步:"
echo "  1. 检查生成的JSONL文件"
echo "  2. 准备人工评分（150条 × 15个principles）"
echo "  3. 生成Gemini响应（使用类似流程）"
