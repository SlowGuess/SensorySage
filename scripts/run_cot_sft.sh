#!/bin/bash
#
# CoT (Chain-of-Thought) 蒸馏训练脚本
# 训练Llama3学习思维链推理模式
#

export CUDA_VISIBLE_DEVICES=0,1,2,3

NPROC_PER_NODE=4
SAVE_PATH="checkpoints/sleep_coach_cot_sft"

BASE_DIR=$(pwd)
TRAIN_DATA="$BASE_DIR/dataset/parquet/cot/train.parquet"
VAL_DATA="$BASE_DIR/dataset/parquet/cot/val.parquet"
MODEL_PATH="$BASE_DIR/verl/models/Llama-3-8B-Instruct/LLM-Research/Meta-Llama-3-8B-Instruct"

mkdir -p $SAVE_PATH

echo "================================================"
echo "CoT Distillation Training"
echo "================================================"
echo "Training Data: $TRAIN_DATA"
echo "Validation Data: $VAL_DATA"
echo "Model Path: $MODEL_PATH"
echo "Save Path: $SAVE_PATH"
echo "GPUs: $NPROC_PER_NODE"
echo "================================================"

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROC_PER_NODE \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    data.train_batch_size=16 \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain=$MODEL_PATH \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bfloat16 \
    model.strategy=fsdp2 \
    model.trust_remote_code=True \
    optim.lr=1e-5 \
    optim.weight_decay=0.01 \
    optim.clip_grad=1.0 \
    trainer.default_local_dir=$SAVE_PATH \
    trainer.project_name=Health-LLM-verl \
    trainer.experiment_name=llama3_cot_sft \
    trainer.total_epochs=5 \
    trainer.save_freq=1 \
    +trainer.save_steps=-1 \
    +trainer.max_ckpt_to_keep=3 \
    trainer.test_freq=1 \
    trainer.logger='["console","wandb"]' \
    trainer.nnodes=1 \
    trainer.n_gpus_per_node=$NPROC_PER_NODE \
    trainer.device=cuda

echo ""
echo "Training completed!"
echo "Checkpoints saved to: $SAVE_PATH"
