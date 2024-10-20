set -x 

ray job submit --address="http://127.0.0.1:8265" \
    -- python3 -m openrlhf.cli.train_ppo_ray \
    --ref_num_nodes 1 \
    --ref_num_gpus_per_node 2 \
    --reward_num_nodes 1 \
    --reward_num_gpus_per_node 2 \
    --critic_num_nodes 1 \
    --critic_num_gpus_per_node 2 \
    --actor_num_nodes 1 \
    --actor_num_gpus_per_node 2 \
    --colocate_critic_reward \
    --colocate_actor_ref \
    --prompt_data $NAS/data/ultrafeedback_binarized \
    --prompt_split train_gen \
    --input_key messages \
    --apply_chat_template \
    --n_samples_per_prompt 8 \
    --reward_pretrain $NAS/models/gemma-2-2b-rm-ultrafeedback \
    --pretrain $NAS/models/gemma-2-2b-sft \
    --save_path $NAS/models/gemma-2-2b-ppo-ultrafeedback-v0.1 \
    --ckpt_path $NAS/models/gemma-2-2b-ppo-ultrafeedback-v0.1/ckpt \
    --max_ckpt_num 100 \
    --logging_steps 1 \
    --save_steps 1 \
    --eval_steps -1 \
    --micro_train_batch_size 1 \
    --train_batch_size 1024 \
    --micro_rollout_batch_size 2 \
    --rollout_batch_size 1024 \
    --max_epochs 1 \
    --prompt_max_len 2048 \
    --generate_max_len 2048 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 9e-6 \
    --init_kl_coef 0.01 \
    --init_kl_coef2 0.0 \
    --actor_init_on_gpu \
    --adam_offload \
    --gradient_checkpointing \
    --use_wandb $WANDB_API_KEY \
    --wandb_run_name gemma-2-2b-ppo-saferlhf-v0.1