#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-0.6B}
TRAIN_FILE=${TRAIN_FILE:-${ROOT}/data/memory_builder/train.parquet}
VAL_FILE=${VAL_FILE:-${ROOT}/data/memory_builder/val.parquet}
NGPUS=${NGPUS:-1}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-32}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-8}
PPO_MICRO_BATCH_SIZE=${PPO_MICRO_BATCH_SIZE:-4}
ROLLOUT_N=${ROLLOUT_N:-4}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${ROOT}/checkpoints/memory_builder}
ROLLOUT_DATA_DIR=${ROLLOUT_DATA_DIR:-${CHECKPOINT_DIR}/rollouts}
VERL_HOME=${VERL_HOME:-${ROOT}/.verl-cu124-src}
PYTHON_BIN=${PYTHON_BIN:-${ROOT}/.venv-cu124/bin/python}

export PYTHONPATH="${VERL_HOME}:${ROOT}:${PYTHONPATH:-}"
cd "${VERL_HOME}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.max_prompt_length=4096 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=batch \
    custom_reward_function.path="${ROOT}/defender/verl_reward.py" \
    custom_reward_function.name=compute_scores \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name=adv_mem \
    trainer.experiment_name=memory_builder_qwen3_0.6b_grpo \
    trainer.default_local_dir="${CHECKPOINT_DIR}" \
    trainer.rollout_data_dir="${ROLLOUT_DATA_DIR}" \
    trainer.n_gpus_per_node="${NGPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    "$@"
