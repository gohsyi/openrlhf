set -xe

ray job submit --address="http://127.0.0.1:8265" \
   -- python3 -m openrlhf.cli.train_ppo_ray \
   --ref_num_nodes 1 \
   --ref_num_gpus_per_node 1 \
   --reward_num_nodes 1 \
   --reward_num_gpus_per_node 1 \
   --critic_num_nodes 1 \
   --critic_num_gpus_per_node 2 \
   --actor_num_nodes 1 \
   --actor_num_gpus_per_node 2 \
   --n_samples_per_prompt 4 \
   --pretrain meta-llama/Llama-3.2-1B-Instruct \
   --reward_pretrain gohsyi/Llama-3.2-1B-Instruct-rm-ultrafeedback \
   --save_path ../models/Llama-3.2-1B-Instruct-ppo4-ultrafeedback-v0.1 \
   --micro_train_batch_size 2 \
   --train_batch_size 128 \
   --micro_rollout_batch_size 4 \
   --rollout_batch_size 1024 \
   --max_epochs 1 \
   --prompt_max_len 1024 \
   --generate_max_len 1024 \
   --zero_stage 2 \
   --bf16 \
   --actor_learning_rate 5e-7 \
   --critic_learning_rate 9e-6 \
   --init_kl_coef 0.01 \
   --prompt_data gohsyi/ultrafeedback-prompt \
   --input_key messages \
   --apply_chat_template \
   --normalize_reward \
   --adam_offload \
   --flash_attn \
   --gradient_checkpointing \
   --load_checkpoint \
   --use_wandb $WANDB_API_KEY \
   --wandb_run_name Llama-3.2-1B-Instruct-ppo4-ultrafeedback-v0.1
