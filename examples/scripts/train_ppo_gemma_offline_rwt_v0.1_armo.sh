#!/bin/bash

set -xe

ITER=1
DATASET=ultrafeedback
SFT_MODEL=$NAS/models/gemma-2-2b-it
ROLLOUT_BATCH_SIZE=2048
N_SAMPLES_PER_PROMPT=4

while getopts i:d:m:b:n: FLAG
do
    case "${FLAG}" in
        i) ITER=${OPTARG};;
        d) DATASET=${OPTARG};;
        m) SFT_MODEL=${OPTARG};;
        b) ROLLOUT_BATCH_SIZE=${OPTARG};;
        n) N_SAMPLES_PER_PROMPT=${OPTARG};;
    esac
done

MODEL=$SFT_MODEL
REWARD_MODEL=$NAS/models/gemma-2-2b-it
CRITIC_MODEL=$REWARD_MODEL
BETA1=0.01

for i in {1..80}; do
    if [ $i -ge $ITER ]; then
        deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
            --eval_task generate \
            --pretrain $MODEL \
            --dataset $NAS/data/${DATASET}-prompt \
            --dataset_split train \
            --input_key messages \
            --apply_chat_template \
            --iter $i \
            --rollout_batch_size $ROLLOUT_BATCH_SIZE \
            --micro_batch_size 8 \
            --max_new_tokens 2048 \
            --prompt_max_len 2048 \
            --output_path $NAS/data/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2.jsonl \
            --best_of_n $N_SAMPLES_PER_PROMPT \
            --bf16

        deepspeed --master_port $PORT1 --module openrlhf.cli.batch_inference \
            --eval_task armo \
            --pretrain $NAS/models/ArmoRM-Llama3-8B-v0.1 \
            --dataset $NAS/data/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2.jsonl \
            --input_key messages \
            --max_len 4096 \
            --prompt_max_len 2048 \
            --apply_chat_template \
            --output_path $NAS/data/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2-rm.jsonl \
            --post_processor reweight \
            --beta $BETA1

        deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo_offline \
            --prompt_data $NAS/data/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2-rm.jsonl \
            --input_key messages \
            --reward_key reward \
            --weight_key weight \
            --apply_chat_template \
            --critic_pretrain $CRITIC_MODEL \
            --pretrain $MODEL \
            --initial $SFT_MODEL \
            --save_path $NAS/models/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2 \
            --save_value_network \
            --save_steps -1 \
            --logging_steps 1 \
            --eval_steps -1 \
            --micro_train_batch_size 2 \
            --train_batch_size $(( N_SAMPLES_PER_PROMPT*512 )) \
            --micro_rollout_batch_size 4 \
            --rollout_batch_size $(( N_SAMPLES_PER_PROMPT*ROLLOUT_BATCH_SIZE )) \
            --max_epochs 1 \
            --prompt_max_len 2048 \
            --generate_max_len 2048 \
            --zero_stage 2 \
            --bf16 \
            --actor_learning_rate 5e-7 \
            --critic_learning_rate 9e-6 \
            --init_kl_coef $BETA1 \
            --actor_init_on_gpu \
            --adam_offload \
            --gradient_checkpointing \
            --use_wandb $WANDB_API_KEY \
            --wandb_run_name gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2
    fi
    MODEL=$NAS/models/gemma-2-2b-ppo${N_SAMPLES_PER_PROMPT}-${DATASET}-${ROLLOUT_BATCH_SIZE:0:1}k-iter$i-armo-v0.2
    CRITIC_MODEL=$MODEL-critic
done
