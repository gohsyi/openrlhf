# Collect
git clone -b checkpoint https://github.com/gohsyi/openrlhf.git
cd openrlhf

while getopts ":m:n:" arg; do
    case $arg in
        m)
            MODEL=$OPTARG
            ;;
        n)
            NAS=$OPTARG
            ;;
    esac
done

for STEP in $(seq 1 100);
do
    if [ -d "$NAS/models/$MODEL-ckpt/_actor/global_step$STEP" ]; then
        deepspeed --master_port $PORT1 --module openrlhf.cli.train_ppo \
            --prompt_data gohsyi/ultrafeedback-prompt \
            --prompt_split test \
            --input_key messages \
            --load_checkpoint global_step${STEP} \
            --apply_chat_template \
            --n_samples_per_prompt 1 \
            --pretrain $NAS/models/$MODEL \
            --save_path $NAS/models/$MODEL-earlystop \
            --ckpt_path $NAS/models/$MODEL-ckpt \
            --logging_steps 1 \
            --save_steps 1 \
            --eval_steps -1 \
            --micro_train_batch_size 1 \
            --train_batch_size 16 \
            --micro_rollout_batch_size 2 \
            --rollout_batch_size 16 \
            --max_epochs 1 \
            --prompt_max_len 2048 \
            --generate_max_len 2048 \
            --max_samples 16 \
            --zero_stage 2 \
            --bf16 \
            --actor_learning_rate 5e-7 \
            --critic_learning_rate 9e-6 \
            --init_kl_coef 0.01 \
            --init_kl_coef2 1.0 \
            --actor_init_on_gpu \
            --adam_offload \
            --gradient_checkpointing \
            --use_wandb $WANDB_API_KEY \
            --wandb_run_name $MODEL-step${STEP}

        if [ -z "$( ls $NAS/models/$MODEL-earlystop )" ]; then
            huggingface-cli upload gohsyi/$MODEL-earlystop $NAS/models/$MODEL-earlystop --token=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
        fi
    fi
done
