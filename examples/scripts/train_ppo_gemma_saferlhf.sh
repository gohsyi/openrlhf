set -x 

deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo \
    --prompt_data $NAS/data/gemma-2-2b-sft-saferlhf-iter1-annotation.jsonl \
    --input_key messages \
    --apply_chat_template \
    --reward_pretrain $NAS/models/gemma-2-2b-rm-saferlhf \
    --pretrain $NAS/models/gemma-2-2b-sft \
    --save_path $NAS/models/gemma-2-2b-ppo-saferlhf-iter1 \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size 64 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 512 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --max_samples 1000000 \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-ppo-saferlhf-iter1


# No flash_attn for gemma
