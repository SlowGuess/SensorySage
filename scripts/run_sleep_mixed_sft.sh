#!/bin/bash
#
# 优化的混合训练脚本 - 同时训练 Insights, Etiology, Recommendations
# 基于论文的Mixture Training方法
#
# 优化点:
# 1. 调整checkpoint保存策略（节省磁盘空间）
# 2. 添加验证频率控制
# 3. 优化epoch数量
# 4. 添加学习率调度器配置
#

export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC_PER_NODE=4
SAVE_PATH="checkpoints/sleep_mixed_sft_llama3_8b"

BASE_DIR=$(pwd)
TRAIN_DATA="$BASE_DIR/dataset/parquet/sleep_mixed/train.parquet"
VAL_DATA="$BASE_DIR/dataset/parquet/sleep_mixed/val.parquet"
MODEL_PATH="$BASE_DIR/verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct"

# 创建保存目录
mkdir -p $SAVE_PATH

echo "=========================================="
echo "Sleep Mixed SFT Training (Optimized)"
echo "=========================================="
echo "Training Data: $TRAIN_DATA"
echo "Validation Data: $VAL_DATA"
echo "Model Path: $MODEL_PATH"
echo "Save Path: $SAVE_PATH"
echo "GPUs: $NPROC_PER_NODE"
echo "=========================================="

# 检查数据文件是否存在
if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please run: python data_processing/convert_mixed_sleep_to_parquet.py"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data not found at $VAL_DATA"
    echo "Please run: python data_processing/convert_mixed_sleep_to_parquet.py"
    exit 1
fi

# 启动训练
torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=32 \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bfloat16 \
    model.strategy=fsdp2 \
    model.trust_remote_code=True \
    optim.lr=2e-5 \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.1 \
    optim.clip_grad=1.0 \
    optim.lr_scheduler=cosine \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=PH-LLM-Reproduction \
    trainer.experiment_name=sleep_mixed_sft_llama3_8b \
    trainer.total_epochs=5 \
    trainer.save_freq=-1 \
    +trainer.save_steps=100 \
    trainer.test_freq=1 \
    trainer.max_ckpt_to_keep=5 \
    trainer.logger='["console","wandb"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.device=cuda

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $SAVE_PATH"
echo "=========================================="
