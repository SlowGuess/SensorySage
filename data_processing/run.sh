#!/bin/bash
#
# 自动生成Gemini三阶段holdout数据
# Step 1: Insights -> Step 2: Etiology -> Step 3: Recommendations
#

set -e  # 遇到错误立即退出

echo "======================================================"
echo "Gemini Holdout Data Generation - Three Stages"
echo "======================================================"
echo ""

# 定义文件路径
CASES_FILE="dataset/raw/sleep_holdout_cases.json"
STAGE1_OUTPUT="results/gemini_holdout_stage1.json"
STAGE2_OUTPUT="results/gemini_holdout_stage2.json"
STAGE3_OUTPUT="results/gemini_holdout_stage3_complete.json"

# 创建输出目录
mkdir -p results

echo "检查输入文件..."
if [ ! -f "$CASES_FILE" ]; then
    echo "错误: $CASES_FILE 不存在！"
    exit 1
fi

echo "输入文件: $CASES_FILE"
echo ""

# Step 1: 生成Insights
echo "======================================================"
echo "Step 1: Generating Insights"
echo "======================================================"
python data_processing/step1_generate_gemini_stage1.py \
    --cases_file "$CASES_FILE" \
    --output_file "$STAGE1_OUTPUT"

if [ $? -ne 0 ]; then
    echo "错误: Step 1 失败！"
    exit 1
fi

echo ""
echo "✓ Step 1 完成: $STAGE1_OUTPUT"
echo ""

# Step 2: 生成Etiology
echo "======================================================"
echo "Step 2: Generating Etiology"
echo "======================================================"
python data_processing/step2_generate_gemini_stage2.py \
    --stage1_file "$STAGE1_OUTPUT" \
    --output_file "$STAGE2_OUTPUT"

if [ $? -ne 0 ]; then
    echo "错误: Step 2 失败！"
    exit 1
fi

echo ""
echo "✓ Step 2 完成: $STAGE2_OUTPUT"
echo ""

# Step 3: 生成Recommendations
echo "======================================================"
echo "Step 3: Generating Recommendations"
echo "======================================================"
python data_processing/step3_generate_gemini_stage3.py \
    --stage2_file "$STAGE2_OUTPUT" \
    --output_file "$STAGE3_OUTPUT"

if [ $? -ne 0 ]; then
    echo "错误: Step 3 失败！"
    exit 1
fi

echo ""
echo "✓ Step 3 完成: $STAGE3_OUTPUT"
echo ""

# 完成
echo "======================================================"
echo "所有步骤完成！"
echo "======================================================"
echo ""
echo "生成的文件："
ls -lh "$STAGE1_OUTPUT" "$STAGE2_OUTPUT" "$STAGE3_OUTPUT"
echo ""
echo "下一步: 运行 step4 转换为标准格式用于训练"