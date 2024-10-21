#!/bin/bash

set -xe

MODEL=$1

# deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
#     --eval_task generate \
#     --pretrain $NAS/models/$MODEL \
#     --dataset $NAS/data/saferlhf-prompt \
#     --dataset_split test \
#     --max_samples 1000 \
#     --input_key prompt \
#     --micro_batch_size 32 \
#     --max_new_tokens 2048 \
#     --prompt_max_len 2048 \
#     --apply_chat_template \
#     --output_path $NAS/data/$MODEL-saferlhf-test.jsonl \
#     --temperature 0.7 \
#     --bf16

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task llamaguard \
    --pretrain $NAS/models/llama-guard-3-8b \
    --dataset $NAS/data/$MODEL-saferlhf-test.jsonl \
    --input_key messages \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-saferlhf-llamaguard.jsonl

# deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
#     --eval_task shieldgemma \
#     --pretrain $NAS/models/shieldgemma-2b \
#     --dataset $NAS/data/$MODEL-saferlhf-test.jsonl \
#     --input_key messages \
#     --apply_chat_template \
#     --output_path $NAS/data/$MODEL-saferlhf-shieldgemma.jsonl
