set -x

deepspeed --master_port $PORT1 --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset $NAS/data/ultrafeedback_binarized \
    --train_split train_sft \
    --eval_split test_sft \
    --input_key messages \
    --apply_chat_template \
    --train_batch_size 256 \
    --micro_train_batch_size 2 \
    --pretrain $NAS/models/gemma-2-2b-it \
    --save_path $NAS/models/gemma-2-2b-it-sft-ultrafeedback \
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
    --wandb_run_name gemma-2-2b-it-sft-ultrafeedback

# No flash-attn for gemma 2
