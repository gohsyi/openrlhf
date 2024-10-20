#!/bin/bash

set -xe

for STEP in $(seq 1 100);
do
    deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo \
        --prompt_data $NAS/data/ultrafeedback-prompt \
        --prompt_split test \
        --eval_split test \
        --input_key messages \
        --load_checkpoint global_step${STEP} \
        --apply_chat_template \
        --n_samples_per_prompt 1 \
        --reward_pretrain $NAS/models/Meta-Llama-3.1-8B-Instruct-rm-ultrafeedback \
        --pretrain $NAS/models/Meta-Llama-3.1-8B-Instruct \
        --save_path $NAS/models/Meta-Llama-3.1-8B-Instruct-ppo4-rwt-ultrafeedback-v0.1-step${STEP} \
        --ckpt_path $NAS/models/Meta-Llama-3.1-8B-Instruct-ppo4-rwt-ultrafeedback-v0.1-ckpt \
        --logging_steps 1 \
        --save_steps 1 \
        --eval_steps -1 \
        --micro_train_batch_size 1 \
        --train_batch_size 512 \
        --micro_rollout_batch_size 2 \
        --rollout_batch_size 512 \
        --max_epochs 1 \
        --prompt_max_len 2048 \
        --generate_max_len 2048 \
        --zero_stage 2 \
        --bf16 \
        --actor_learning_rate 5e-7 \
        --critic_learning_rate 9e-6 \
        --init_kl_coef 0.01 \
        --init_kl_coef2 1.0 \
        --actor_init_on_gpu \
        --adam_offload \
        --gradient_checkpointing \
        --use_wandb $WANDB_API_KEY \
        --wandb_run_name Meta-Llama-3.1-8B-Instruct-ppo4-rwt-ultrafeedback-v0.1-step${STEP}
done