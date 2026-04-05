#!/bin/bash
set -euo pipefail

echo "=========================================="
echo "Sleep Coaching RL Training with LLM Judge"
echo "=========================================="

export RAY_LOG_TO_STDERR=0
export RAY_DEDUP_LOGS=1
export PYTHONWARNINGS="ignore"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1,3}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

NPROC_PER_NODE=${NPROC_PER_NODE:-2}
SAVE_PATH="${SAVE_PATH:-checkpoints/sleep_coach_rl}"

BASE_DIR=$(pwd)
TRAIN_DATA="${TRAIN_DATA:-$BASE_DIR/dataset/parquet/cot/train_rl.parquet}"
VAL_DATA="${VAL_DATA:-$BASE_DIR/dataset/parquet/cot/val_rl.parquet}"
SFT_MODEL="${SFT_MODEL:-$BASE_DIR/checkpoints/sleep_coach_cot_sft/global_step_95_hf}"
REWARD_API_URL="${REWARD_API_URL:-http://127.0.0.1:6009/get_reward2}"
export REWARD_API_URL

mkdir -p "$SAVE_PATH"

if [ ! -f "$TRAIN_DATA" ]; then
    echo "Error: Training data not found at $TRAIN_DATA"
    echo "Please run: python3 reward_part/add_reward_model.py"
    exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
    echo "Error: Validation data not found at $VAL_DATA"
    echo "Please run: python3 reward_part/add_reward_model.py"
    exit 1
fi

echo ""
echo "配置信息:"
echo "  CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "  GPU数量: $NPROC_PER_NODE"
echo "  Train data: $TRAIN_DATA"
echo "  Val data: $VAL_DATA"
echo "  SFT模型: $SFT_MODEL"
echo "  Reward API: $REWARD_API_URL"
echo "  保存路径: $SAVE_PATH"
echo ""
echo "开始训练..."
echo "=========================================="
echo ""

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$TRAIN_DATA" \
    data.val_files="$VAL_DATA" \
    data.train_batch_size=3 \
    data.max_prompt_length=6144 \
    data.max_response_length=2048 \
    data.prompt_key=prompt \
    actor_rollout_ref.model.path="$SFT_MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.prompt_length=6144 \
    actor_rollout_ref.rollout.response_length=2048 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    reward_model.model.input_tokenizer="$SFT_MODEL" \
    ++custom_reward_function.path=reward_part/sleep_reward_function.py \
    ++custom_reward_function.name=sleep_coaching_reward \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=Health-LLM-verl \
    trainer.experiment_name=llama3_8b_grpo_judge \
    trainer.n_gpus_per_node="$NPROC_PER_NODE" \
    trainer.nnodes=1 \
    trainer.total_epochs=1 \
    trainer.save_freq=150 \
    trainer.test_freq=100 \
    trainer.default_local_dir="$SAVE_PATH"

echo ""
echo "=========================================="
echo "训练完成"
echo "=========================================="