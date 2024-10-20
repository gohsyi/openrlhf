#!/bin/bash

set -xe

deepspeed --include localhost:2,3,4,5 --master_port 29500 --module openrlhf.cli.train_rm \
   --save_path ../models/Llama-3.2-3B-Instruct-rm-ultrafeedback \
   --logging_steps 1 \
   --eval_steps 1000 \
   --train_batch_size 256 \
   --micro_train_batch_size 8 \
   --pretrain meta-llama/Llama-3.2-3B-Instruct \
   --bf16 \
   --max_epochs 1 \
   --max_len 4096 \
   --zero_stage 2 \
   --learning_rate 9e-6 \
   --dataset HuggingFaceH4/ultrafeedback_binarized \
   --train_split train_prefs \
   --eval_split test_prefs \
   --apply_chat_template \
   --chosen_key chosen \
   --rejected_key rejected \
   --adam_offload \
   --gradient_checkpointing \
   --use_wandb $WANDB_API_KEY \
   --flash_attn \
   --wandb_run_name Llama-3.2-3B-Instruct-rm-ultrafeedback

huggingface-cli upload gohsyi/Llama-3.2-3B-Instruct-rm-ultrafeedback
