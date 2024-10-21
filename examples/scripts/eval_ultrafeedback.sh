#!/bin/bash

set -xe

MODEL=$1

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task generate \
    --pretrain $NAS/models/$MODEL \
    --dataset $NAS/data/ultrafeedback-prompt \
    --dataset_split test \
    --input_key messages \
    --micro_batch_size 16 \
    --max_new_tokens 2048 \
    --prompt_max_len 2048 \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-test.jsonl \
    --temperature 0.5 \
    --bf16

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task rm \
    --pretrain $NAS/models/gemma-2-2b-rm-ultrafeedback \
    --dataset $NAS/data/$MODEL-test.jsonl \
    --input_key messages \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-test-rm.jsonl

deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
    --eval_task armo \
    --pretrain $NAS/models/ArmoRM-Llama3-8B-v0.1 \
    --dataset $NAS/data/$MODEL-test.jsonl \
    --input_key messages \
    --apply_chat_template \
    --output_path $NAS/data/$MODEL-test-armo.jsonl
