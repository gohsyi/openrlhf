set -x

deepspeed --master_port $PORT1 --module openrlhf.cli.train_dpo \
    --save_path $NAS/models/gemma-2-2b-it-dpo-ultrafeedback \
    --save_steps -1 \
    --logging_steps 1 \
    --eval_steps -1 \
    --train_batch_size 256 \
    --micro_train_batch_size 1 \
    --pretrain $NAS/models/gemma-2-2b-it \
    --bf16 \
    --max_epochs 1 \
    --max_len 8192 \
    --zero_stage 2 \
    --learning_rate 5e-7 \
    --beta 0.1 \
    --dataset $NAS/data/ultrafeedback_binarized \
    --train_split train_prefs \
    --apply_chat_template \
    --chosen_key chosen \
    --rejected_key rejected \
    --load_checkpoint \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-it-dpo-ultrafeedback
