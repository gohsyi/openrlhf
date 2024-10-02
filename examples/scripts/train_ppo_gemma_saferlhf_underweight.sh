#!/bin/bash

set -x 

deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo \
    --prompt_data $NAS/data/saferlhf-1k-underweight-gemma \
    --prompt_split train \
    --eval_split test \
    --input_key prompt \
    --apply_chat_template \
    --reward_pretrain $NAS/models/gemma-2-2b-rm-saferlhf \
    --pretrain $NAS/models/gemma-2-2b-sft \
    --save_path $NAS/models/gemma-2-2b-ppo-saferlhf-1k-underweight \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps 1 \
    --micro_train_batch_size 2 \
    --train_batch_size 128 \
    --micro_rollout_batch_size 4 \
    --rollout_batch_size 1024 \
    --micro_eval_batch_size 16 \
    --max_samples_eval 1000 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --actor_init_on_gpu \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-ppo-saferlhf-1k-underweight
