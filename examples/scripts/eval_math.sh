#!/bin/bash

set -xe

MODEL=$1

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain $NAS/models/$MODEL \
    --dataset $NAS/data/math \
    --dataset_split test \
    --solution_key solution \
    --input_key problem \
    --micro_batch_size 8 \
    --max_new_tokens 2048 \
    --prompt_max_len 2048 \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-math-test.jsonl \
    --greedy_sampling \
    --bf16
