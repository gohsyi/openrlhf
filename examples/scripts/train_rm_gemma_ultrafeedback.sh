#!/bin/bash

set -xe

deepspeed --master_port $PORT1 --module openrlhf.cli.train_rm \
   --save_path /mnt/bn/hongyi-yg/models/gemma-2-2b-rm-ultrafeedback \
   --logging_steps 1 \
   --eval_steps 1000 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain /mnt/bn/hongyi-yg/models/gemma-2-2b-sft \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --dataset /mnt/bn/hongyi-yg/data/ultrafeedback_binarized \
   --train_split train_prefs \
   --eval_split test_prefs \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --adam_offload \
   --gradient_checkpointing \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_run_name gemma-2-2b-rm-ultrafeedback
