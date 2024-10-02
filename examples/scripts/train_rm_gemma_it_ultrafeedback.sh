#!/bin/bash

set -xe

deepspeed --master_port $PORT1 --module openrlhf.cli.train_rm \
   --save_path $NAS/models/gemma-2-2b-it-rm-ultrafeedback \
   --logging_steps 1 \
   --eval_steps 1000 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain $NAS/models/gemma-2-2b-it \
   --bf16 \
   --max_epochs 1 \
   --max_len 8192 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --dataset $NAS/data/ultrafeedback-binarized \
   --train_split train_prefs \
   --eval_split test_prefs \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --adam_offload \
   --gradient_checkpointing \
   --use_wandb ${WANDB_API_KEY} \
   --wandb_run_name gemma-2-2b-it-rm-ultrafeedback
