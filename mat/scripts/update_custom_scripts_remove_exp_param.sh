#!/bin/bash
# 批量更新所有 custom 脚本，取消 $4 参数（exp_name），完全自动生成实验名称

# 定义地图列表和对应的训练参数
declare -A map_configs=(
    ["1c3s5z"]="5000000:5e-4:10:0.2"
    ["3s5z"]="10000000:5e-4:15:0.05"
    ["5m_vs_6m"]="5000000:5e-4:15:0.05"
    ["6h_vs_8z"]="10000000:5e-4:15:0.05"
    ["8m_vs_9m"]="5000000:5e-4:15:0.05"
    ["10m_vs_11m"]="5000000:5e-4:15:0.05"
    ["3s5z_vs_3s6z"]="20000000:5e-4:15:0.05"
    ["27m_vs_30m"]="10000000:5e-4:15:0.05"
    ["MMM2"]="10000000:5e-4:15:0.05"
)

# 为每个地图生成 custom 脚本
for map in "${!map_configs[@]}"; do
    IFS=':' read -r num_steps lr ppo_epoch clip_param <<< "${map_configs[$map]}"
    
    # 生成脚本文件
    cat > "train_smac_${map}_custom.sh" << 'EOF'
#!/bin/sh
# MAP_NAME 地图专用训练脚本，支持自定义 n_levels、mid_chunk_size 和 seed
# 实验名称自动生成，格式：mscm_{map}_adaptive_seed{seed} 或 mscm_{map}_L{n_levels}_c{mid_chunk_size}_seed{seed}
#
# 使用方法：
# 方法1（使用地图自适应配置，默认 seed=1）：
#   bash train_smac_MAP_NAME_custom.sh 0
#   参数说明：$1=gpu_id
#   实验名称：mscm_MAP_NAME_adaptive_seed1
#
# 方法2（使用地图自适应配置，指定 seed）：
#   bash train_smac_MAP_NAME_custom.sh 0 42
#   参数说明：$1=gpu_id, $2=seed
#   实验名称：mscm_MAP_NAME_adaptive_seed42
#
# 方法3（手动指定 n_levels 和 mid_chunk_size，默认 seed=1）：
#   bash train_smac_MAP_NAME_custom.sh 0 3 4
#   参数说明：$1=gpu_id, $2=n_levels, $3=mid_chunk_size
#   实验名称：mscm_MAP_NAME_L3_c4_seed1
#
# 方法4（手动指定 n_levels、mid_chunk_size 和 seed）：
#   bash train_smac_MAP_NAME_custom.sh 0 3 4 42
#   参数说明：$1=gpu_id, $2=n_levels, $3=mid_chunk_size, $4=seed
#   实验名称：mscm_MAP_NAME_L3_c4_seed42

env="StarCraft2"
map="MAP_NAME"  # 固定为 MAP_NAME 地图
gpu_id="$1"
algo="mat"

# 智能识别参数：如果 $2 是纯数字且 $3 为空，则 $2 是 seed
if [ -n "$2" ] && [ -z "$3" ] && [ "$2" -eq "$2" ] 2>/dev/null; then
    # 方法2：只指定 gpu_id 和 seed
    seed="$2"
    use_map_adaptive="true"
    n_levels=""
    mid_chunk_size=""
    exp="mscm_MAP_NAME_adaptive_seed${seed}"
    echo "=================================="
    echo "Using MAP-ADAPTIVE configuration"
    echo "=================================="
elif [ -z "$2" ] || [ -z "$3" ]; then
    # 方法1：只指定 gpu_id，使用默认 seed=1
    seed="${4:-1}"
    use_map_adaptive="true"
    n_levels=""
    mid_chunk_size=""
    exp="mscm_MAP_NAME_adaptive_seed${seed}"
    echo "=================================="
    echo "Using MAP-ADAPTIVE configuration"
    echo "=================================="
else
    # 方法3/4：手动指定 n_levels 和 mid_chunk_size
    seed="${4:-1}"
    use_map_adaptive="false"
    n_levels="$2"
    mid_chunk_size="$3"
    exp="mscm_MAP_NAME_L${n_levels}_c${mid_chunk_size}_seed${seed}"
    echo "=================================="
    echo "Using MANUAL configuration"
    echo "=================================="
    echo "n_levels:          ${n_levels}"
    echo "mid_chunk_size:    ${mid_chunk_size}"
fi

echo "env:               ${env}"
echo "map:               ${map}"
echo "algo:              ${algo}"
echo "exp:               ${exp}"
echo "seed:              ${seed}"
echo "gpu_id:            ${gpu_id}"
echo "use_map_adaptive:  ${use_map_adaptive}"
echo "=================================="

# 构建命令
cmd="CUDA_VISIBLE_DEVICES=${gpu_id} python train/train_smac.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --map_name ${map} \
    --seed ${seed} \
    --n_training_threads 16 \
    --n_rollout_threads 32 \
    --num_mini_batch 1 \
    --episode_length 100 \
    --num_env_steps NUM_STEPS \
    --lr LR \
    --ppo_epoch PPO_EPOCH \
    --clip_param CLIP_PARAM \
    --save_interval 100000 \
    --use_value_active_masks \
    --use_eval"

# 如果使用手动配置，添加相应参数
if [ "$use_map_adaptive" = "false" ]; then
    cmd="${cmd} --n_cms_levels ${n_levels} --cms_mid_chunk_size ${mid_chunk_size} --no-use_map_adaptive"
fi

# 执行命令
eval $cmd
EOF

    # 替换地图名称和训练参数
    sed -i "s/MAP_NAME/${map}/g" "train_smac_${map}_custom.sh"
    sed -i "s/NUM_STEPS/${num_steps}/g" "train_smac_${map}_custom.sh"
    sed -i "s/LR/${lr}/g" "train_smac_${map}_custom.sh"
    sed -i "s/PPO_EPOCH/${ppo_epoch}/g" "train_smac_${map}_custom.sh"
    sed -i "s/CLIP_PARAM/${clip_param}/g" "train_smac_${map}_custom.sh"
    
    chmod +x "train_smac_${map}_custom.sh"
    echo "Updated: train_smac_${map}_custom.sh (removed exp_name parameter)"
done

echo ""
echo "All custom scripts updated - exp_name parameter removed!"
