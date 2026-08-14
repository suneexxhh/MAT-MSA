#!/bin/bash
# 批量生成所有地图的 custom 训练脚本

# 定义地图列表和对应的训练参数
declare -A map_configs=(
    ["3s5z"]="10000000:5e-4:15:0.05"
    ["5m_vs_6m"]="5000000:5e-4:15:0.05"
    ["6h_vs_8z"]="10000000:5e-4:15:0.05"
    ["8m_vs_9m"]="5000000:5e-4:15:0.05"
    ["10m_vs_11m"]="5000000:5e-4:15:0.05"
    ["3s5z_vs_3s6z"]="20000000:5e-4:15:0.05"
    ["MMM2"]="10000000:5e-4:15:0.05"
)

# 为每个地图生成 custom 脚本
for map in "${!map_configs[@]}"; do
    IFS=':' read -r num_steps lr ppo_epoch clip_param <<< "${map_configs[$map]}"
    
    # 生成脚本文件
    cat > "train_smac_${map}_custom.sh" << EOF
#!/bin/sh
# ${map} 地图专用训练脚本，支持自定义 n_levels 和 mid_chunk_size
#
# 使用方法：
# 方法1（使用地图自适应配置）：
#   bash train_smac_${map}_custom.sh 0
#   参数说明：\$1=gpu_id
#
# 方法2（手动指定参数）：
#   bash train_smac_${map}_custom.sh 0 3 4
#   参数说明：\$1=gpu_id, \$2=n_levels, \$3=mid_chunk_size
#
# 方法3（手动指定参数 + 实验名称）：
#   bash train_smac_${map}_custom.sh 0 3 4 my_experiment
#   参数说明：\$1=gpu_id, \$2=n_levels, \$3=mid_chunk_size, \$4=exp_name

env="StarCraft2"
map="${map}"  # 固定为 ${map} 地图
gpu_id="\$1"
algo="mat"
seed=1

# 检查是否提供了自定义参数
if [ -z "\$2" ] || [ -z "\$3" ]; then
    # 没有提供 n_levels 和 mid_chunk_size，使用地图自适应配置
    use_map_adaptive="true"
    n_levels=""
    mid_chunk_size=""
    exp="\${4:-map_adaptive_${map}}"
    echo "=================================="
    echo "Using MAP-ADAPTIVE configuration"
    echo "=================================="
else
    # 提供了 n_levels 和 mid_chunk_size，使用手动配置
    use_map_adaptive="false"
    n_levels="\$2"
    mid_chunk_size="\$3"
    exp="\${4:-${map}_manual_\${n_levels}levels_c\${mid_chunk_size}}"
    echo "=================================="
    echo "Using MANUAL configuration"
    echo "=================================="
    echo "n_levels:          \${n_levels}"
    echo "mid_chunk_size:    \${mid_chunk_size}"
fi

echo "env:               \${env}"
echo "map:               \${map}"
echo "algo:              \${algo}"
echo "exp:               \${exp}"
echo "seed:              \${seed}"
echo "gpu_id:            \${gpu_id}"
echo "use_map_adaptive:  \${use_map_adaptive}"
echo "=================================="

# 构建命令
cmd="CUDA_VISIBLE_DEVICES=\${gpu_id} python train/train_smac.py \\
    --env_name \${env} \\
    --algorithm_name \${algo} \\
    --experiment_name \${exp} \\
    --map_name \${map} \\
    --seed \${seed} \\
    --n_training_threads 16 \\
    --n_rollout_threads 32 \\
    --num_mini_batch 1 \\
    --episode_length 100 \\
    --num_env_steps ${num_steps} \\
    --lr ${lr} \\
    --ppo_epoch ${ppo_epoch} \\
    --clip_param ${clip_param} \\
    --save_interval 100000 \\
    --use_value_active_masks \\
    --use_eval"

# 如果使用手动配置，添加相应参数
if [ "\$use_map_adaptive" = "false" ]; then
    cmd="\${cmd} --n_cms_levels \${n_levels} --cms_mid_chunk_size \${mid_chunk_size} --no-use_map_adaptive"
fi

# 执行命令
eval \$cmd
EOF

    chmod +x "train_smac_${map}_custom.sh"
    echo "Created: train_smac_${map}_custom.sh"
done

echo ""
echo "All custom scripts generated successfully!"
