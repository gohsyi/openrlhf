set -x

deepspeed --master_port $PORT1 --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset $NAS/data/metamath-sft \
    --train_split train \
    --input_key messages \
    --apply_chat_template \
    --tokenizer_chat_template "{{ bos_token }}{% if messages[0]['role'] == 'system' %}{{ raise_exception('System role not supported') }}{% endif %}{% for message in messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}" \
    --train_batch_size 256 \
    --micro_train_batch_size 2 \
    --pretrain $NAS/models/gemma-2-2b \
    --save_path $NAS/models/gemma-2-2b-sft-metamath \
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
    --wandb_run_name gemma-2-2b-sft-metamath

# No flash-attn for gemma 2
