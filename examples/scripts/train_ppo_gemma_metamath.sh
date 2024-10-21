#!/bin/bash

set -xe

deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo \
    --prompt_data $NAS/data/metamath-prompt \
    --prompt_split train \
    --input_key messages \
    --solution_key solution \
    --apply_chat_template \
    --n_samples_per_prompt 8 \
    --critic_pretrain $NAS/models/gemma-2-2b-ppo8-metamath-v0.3_critic \
    --pretrain $NAS/models/gemma-2-2b-ppo8-metamath-v0.3 \
    --save_path $NAS/models/gemma-2-2b-ppo8-metamath-v0.3-epoch2 \
    --ckpt_path $NAS/models/gemma-2-2b-ppo8-metamath-v0.3-epoch2-ckpt \
    --save_value_network \
    --max_ckpt_num 100 \
    --logging_steps 1 \
    --save_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size 256 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --init_kl_coef2 0.0 \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-ppo8-metamath-v0.3-epoch2
