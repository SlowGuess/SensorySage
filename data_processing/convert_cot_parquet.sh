#!/bin/bash
#
# 批量转换CoT训练数据为Parquet格式
#

echo "转换CoT数据为Parquet格式"
echo "================================"

# 定义文件映射
declare -A FILE_MAP=(
    ["train"]="data/cot_training_full_sft.jsonl"
    ["val"]="data/cot_validation_sft.jsonl"
    ["test"]="data/cot_test_sft.jsonl"
)

OUTPUT_DIR="dataset/parquet/cot"
mkdir -p $OUTPUT_DIR

for split in train val test; do
    input_file="${FILE_MAP[$split]}"

    if [ ! -f "$input_file" ]; then
        echo "警告: $input_file 不存在，跳过 $split"
        continue
    fi

    echo ""
    echo "转换 $split 数据..."
    echo "  输入: $input_file"

    python data_processing/convert_cot_to_parquet.py \
        --input_file "$input_file" \
        --output_dir "$OUTPUT_DIR"

    # 重命名为对应的split名称
    mv "$OUTPUT_DIR/full.parquet" "$OUTPUT_DIR/${split}.parquet"

    echo "  输出: $OUTPUT_DIR/${split}.parquet"
done

echo ""
echo "================================"
echo "转换完成！"
echo ""
echo "生成的文件:"
ls -lh "$OUTPUT_DIR"/*.parquet

echo ""
echo "下一步: bash scripts/run_cot_sft.sh"
