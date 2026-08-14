#!/bin/bash

# 定义要训练的地图列表
maps=("MMM2" "3s5z" "5m_vs_6m" "10m_vs_11m" "3s5z_vs_3s6z" "corridor" "1c3s5z" "27m_vs_30m" "6h_vs_8z" "MMM" "3s_vs_5z" "8m_vs_9m" "25m")
#maps=()
# 定义 GPU 数量
num_gpus=3  # 修改为你实际的 GPU 数量
gpu_id=0  # 初始 GPU 设备编号

# 创建 logs 目录（如果不存在）
mkdir -p logs

# 遍历地图列表
for map in "${maps[@]}"; do
    echo "Starting training on map: $map using GPU $gpu_id"

    # 使用 `nohup` 在后台运行，每个任务独占一个 GPU，并将日志保存到 logs/ 目录
    nohup bash train_smac.sh "$map" "$gpu_id" > "logs/${map}.log" 2>&1

    # 切换到下一个 GPU
    gpu_id=$(( (gpu_id + 1) % num_gpus ))

    # 避免多个任务同时启动导致资源争抢，间隔 5 秒启动下一个任务
    sleep 60
done

echo "All training tasks started!"
