#!/bin/bash

set -xe

deepspeed --master_port $PORT1 --module openrlhf.cli.train_rloo \
    --prompt_data /mnt/bn/hongyi-yg/data/saferlhf_prompt \
    --input_key prompt \
    --apply_chat_template \
    --reward_pretrain /mnt/bn/hongyi-yg/models/gemma-2-2b-it-rm-saferlhf \
    --pretrain /mnt/bn/hongyi-yg/models/gemma-2-2b-it \
    --save_path /mnt/bn/hongyi-yg/models/gemma-2-2b-it-saferlhf-rloo8-reweight1.0 \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 2 \
    --train_batch_size 1024 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --init_kl_coef 0.01 \
    --init_kl_coef2 0.0 \
    --max_samples 1000000 \
    --actor_init_on_gpu \
    --adam_offload \
    --flash_attn \
    --gradient_checkpointing \
    --rloo_k 8 \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-it-saferlhf-rloo8-reweight1.0
