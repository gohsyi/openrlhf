#!/bin/bash

set -xe

for STEP in 1 4;
do
    deepspeed --include localhost:0,2 --master_port 29505 --module openrlhf.cli.train_ppo \
        --prompt_data gohsyi/ultrafeedback-prompt \
        --prompt_split test \
        --input_key messages \
        --load_checkpoint global_step${STEP} \
        --apply_chat_template \
        --n_samples_per_prompt 1 \
        --reward_pretrain gohsyi/Meta-Llama-3.1-8B-Instruct-rm-ultrafeedback \
        --pretrain meta-llama/Meta-Llama-3.1-8B-Instruct \
        --save_path ../models/Meta-Llama-3.1-8B-Instruct-ppo4-ultrafeedback-v0.1-step${STEP} \
        --ckpt_path ../models/Meta-Llama-3.1-8B-Instruct-ppo4-ultrafeedback-v0.1-ckpt \
        --logging_steps 1 \
        --save_steps 1 \
        --eval_steps -1 \
        --micro_train_batch_size 1 \
        --train_batch_size 16 \
        --micro_rollout_batch_size 2 \
        --rollout_batch_size 16 \
        --max_epochs 1 \
        --prompt_max_len 2048 \
        --generate_max_len 2048 \
        --max_samples 16 \
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
        --wandb_run_name Meta-Llama-3.1-8B-Instruct-ppo4-ultrafeedback-v0.1-step${STEP}
done