set -x

deepspeed --master_port $PORT1 --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset /mnt/bn/hongyi-yg/data/openhermes \
    --train_split train \
    --input_key filtered \
    --apply_chat_template \
    --train_batch_size 256 \
    --micro_train_batch_size 2 \
    --pretrain /mnt/bn/hongyi-yg/models/gemma-2-2b \
    --save_path /mnt/bn/hongyi-yg/models/gemma-2-2b-sft-mixture \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --adam_offload \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-sft-mixture

# No flash-attn for gemma 2
