#!/bin/sh
# 3s5z 地图专用训练脚本，支持自定义 n_levels、mid_chunk_size 和 seed
# 实验名称自动生成，格式：mat_msa_{map}_adaptive_seed{seed} 或 mat_msa_{map}_L{n_levels}_c{mid_chunk_size}_seed{seed}
# 日志自动保存到 logs/ 目录，文件名与实验名称一致
#
# 使用方法：
# 方法1（使用地图自适应配置，默认 seed=1）：
#   bash train_smac_3s5z_custom.sh 0
#   参数说明：$1=gpu_id
#   实验名称：mat_msa_3s5z_adaptive_seed1
#   日志文件：logs/mat_msa_3s5z_adaptive_seed1.log
#
# 方法2（使用地图自适应配置，指定 seed）：
#   bash train_smac_3s5z_custom.sh 0 42
#   参数说明：$1=gpu_id, $2=seed
#   实验名称：mat_msa_3s5z_adaptive_seed42
#   日志文件：logs/mat_msa_3s5z_adaptive_seed42.log
#
# 方法3（手动指定 n_levels 和 mid_chunk_size，默认 seed=1）：
#   bash train_smac_3s5z_custom.sh 0 3 4
#   参数说明：$1=gpu_id, $2=n_levels, $3=mid_chunk_size
#   实验名称：mat_msa_3s5z_L3_c4_seed1
#   日志文件：logs/mat_msa_3s5z_L3_c4_seed1.log
#
# 方法4（手动指定 n_levels、mid_chunk_size 和 seed）：
#   bash train_smac_3s5z_custom.sh 0 3 4 42
#   参数说明：$1=gpu_id, $2=n_levels, $3=mid_chunk_size, $4=seed
#   实验名称：mat_msa_3s5z_L3_c4_seed42
#   日志文件：logs/mat_msa_3s5z_L3_c4_seed42.log

env="StarCraft2"
map="3s5z"  # 固定为 3s5z 地图
gpu_id="$1"
algo="mat"

# 智能识别参数：如果 $2 是纯数字且 $3 为空，则 $2 是 seed
if [ -n "$2" ] && [ -z "$3" ] && [ "$2" -eq "$2" ] 2>/dev/null; then
    # 方法2：只指定 gpu_id 和 seed
    seed="$2"
    use_map_adaptive="true"
    n_levels=""
    mid_chunk_size=""
    exp="mat_msa_3s5z_adaptive_seed${seed}"
    echo "=================================="
    echo "Using MAP-ADAPTIVE configuration"
    echo "=================================="
elif [ -z "$2" ] || [ -z "$3" ]; then
    # 方法1：只指定 gpu_id，使用默认 seed=1
    seed="1"
    use_map_adaptive="true"
    n_levels=""
    mid_chunk_size=""
    exp="mat_msa_3s5z_adaptive_seed${seed}"
    echo "=================================="
    echo "Using MAP-ADAPTIVE configuration"
    echo "=================================="
else
    # 方法3：手动指定 n_levels 和 mid_chunk_size，seed可选（默认为1）
    seed="${4:-1}"
    use_map_adaptive="false"
    n_levels="$2"
    mid_chunk_size="$3"
    exp="mat_msa_3s5z_L${n_levels}_c${mid_chunk_size}_seed${seed}"
    echo "=================================="
    echo "Using MANUAL configuration"
    echo "=================================="
    echo "n_levels:          ${n_levels}"
    echo "mid_chunk_size:    ${mid_chunk_size}"
fi

# 创建日志目录
log_dir="logs"
mkdir -p ${log_dir}

# 日志文件名：实验名称 + 时间戳
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="${log_dir}/${exp}_${timestamp}.log"

echo "env:               ${env}"
echo "map:               ${map}"
echo "algo:              ${algo}"
echo "exp:               ${exp}"
echo "seed:              ${seed}"
echo "gpu_id:            ${gpu_id}"
echo "use_map_adaptive:  ${use_map_adaptive}"
echo "log_file:          ${log_file}"
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
    --num_env_steps 10000000 \
    --lr 5e-4 \
    --ppo_epoch 15 \
    --clip_param 0.05 \
    --save_interval 100000 \
    --use_value_active_masks \
    --use_eval"

# 如果使用手动配置，添加相应参数
if [ "$use_map_adaptive" = "false" ]; then
    cmd="${cmd} --n_cms_levels ${n_levels} --cms_mid_chunk_size ${mid_chunk_size} --no-use_map_adaptive"
fi

# 执行命令并保存日志（同时输出到终端和日志文件）
echo "Starting training... Output will be saved to ${log_file}"
echo "You can monitor the log in real-time with: tail -f ${log_file}"
echo ""

eval $cmd 2>&1 | tee ${log_file}

echo ""
echo "Training completed! Log saved to: ${log_file}"
