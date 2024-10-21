#!/bin/bash

set -xe

MODEL=$1

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain $NAS/models/$MODEL \
    --dataset $NAS/data/gsm8k \
    --dataset_split test \
    --solution_key solution \
    --input_key messages \
    --micro_batch_size 16 \
    --max_new_tokens 2048 \
    --prompt_max_len 2048 \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-gsm8k-test.jsonl \
    --bf16 \
    --greedy_sampling
