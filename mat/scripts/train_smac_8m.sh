#!/bin/sh
env="StarCraft2"
map="8m"
algo="mat"
exp="8m"
gpu_id="${1}"       # 第1个参数：GPU ID
seed="${2:-1}"      # 第2个参数：seed，默认为1（兼容旧用法）
cms_mid_chunk_size="${3:-4}"  # 第3个参数：CMS中间层chunk_size，默认为4

echo "env is ${env}, map is ${map}, algo is ${algo}, exp is ${exp}, seed is ${seed}, cms_mid_chunk_size is ${cms_mid_chunk_size}"
CUDA_VISIBLE_DEVICES=${gpu_id} python train/train_smac.py --env_name ${env} --algorithm_name ${algo} --experiment_name ${exp} --map_name ${map} --seed ${seed} --n_training_threads 16 --n_rollout_threads 32 --num_mini_batch 1 --episode_length 100 --num_env_steps 5000000 --lr 5e-4 --ppo_epoch 15 --clip_param 0.2 --save_interval 100000 --use_value_active_masks --use_eval --cms_mid_chunk_size ${cms_mid_chunk_size}
