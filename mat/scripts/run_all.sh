#!/bin/bash
# =============================================================
# 自动化批量训练启动器
# 用法:
#   ./run_all.sh [--gpu <gpu_ids>] [--parallel <n>]
#
# 示例:
#   ./run_all.sh --gpu 0              # 单 GPU
#   ./run_all.sh --gpu 0,1            # 多 GPU，任务轮询分配
#   nohup./run_all.sh --gpu 0, -p 1    # 3 GPU，同时跑 3 个任务
#
# 停止机制:
#   touch STOP                        # 停止派发新任务（已运行任务继续）
#   rm STOP                           # 恢复正常运行
#
# 说明:
#   - 脚本会在每次启动新任务前实时检查日志是否已存在，避免重复训练
#   - 脚本结束后会自动检查遗漏的任务并启动运行
#   - 已运行的任务不会被中断
#   - 可以安全地多次运行此脚本，已完成的任务会被自动跳过
# =============================================================

# ---------- 默认值 ----------
GPU_ARG="0"
MAX_PARALLEL=""   # 默认等于 GPU 数量

MAPS=(
    # "6h_vs_8z"
    # "3s5z_vs_3s6z"
    # "10m_vs_11m"
    # "1c3s5z"
    # "5m_vs_6m"
    # "8m_vs_9m"
    # "3s5z"
    # "MMM2"
    "27m_vs_30m"
)
SEEDS=(1 2 3 4 5)

# ---------- 解析参数 ----------
while [ $# -gt 0 ]; do
    case "$1" in
        --gpu|-g)
            GPU_ARG="$2"; shift 2 ;;
        --parallel|-p)
            MAX_PARALLEL="$2"; shift 2 ;;
        --)
            shift; break ;;
        -*)
            echo "[ERROR] 未知参数: $1"
            echo "用法: $0 [--gpu <gpu_ids>] [--parallel <n>]"
            exit 1 ;;
        *)
            break ;;
    esac
done

# 将逗号分隔的 GPU 列表转为数组，例如 "0,1,2" -> (0 1 2)
IFS=',' read -ra GPU_LIST <<< "$GPU_ARG"
GPU_COUNT=${#GPU_LIST[@]}

# 默认并发数 = GPU 数量
[ -z "$MAX_PARALLEL" ] && MAX_PARALLEL=$GPU_COUNT

echo "[INFO] GPU 列表: ${GPU_LIST[*]}  并发数: ${MAX_PARALLEL}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# ---------- 检查某个地图+种子是否已有日志 ----------
already_done() {
    local map="$1"
    local seed="$2"
    ls "${LOG_DIR}/${map}_seed${seed}_"*.log 2>/dev/null | grep -q .
}

# ---------- 启动单个任务 ----------
# 全局轮询计数器，用于分配 GPU
task_counter=0

launch_task() {
    local map="$1"
    local seed="$2"

    # 轮询选 GPU
    local gpu_idx=$(( task_counter % GPU_COUNT ))
    local gpu="${GPU_LIST[$gpu_idx]}"
    task_counter=$(( task_counter + 1 ))

    local map_script="${SCRIPT_DIR}/train_smac_${map}.sh"
    if [ ! -f "$map_script" ]; then
        echo "[WARN] 未找到脚本 ${map_script}，跳过 ${map} seed${seed}" >&2
        echo ""
        return
    fi
    [ ! -x "$map_script" ] && chmod +x "$map_script"

    local timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
    local log_file="${LOG_DIR}/${map}_seed${seed}_gpu${gpu}_${timestamp}.log"
    local pid_file="${LOG_DIR}/${map}_seed${seed}_gpu${gpu}_${timestamp}.pid"

    nohup sh "$map_script" "$gpu" "$seed" > "$log_file" 2>&1 &
    local pid=$!
    echo "$pid" > "$pid_file"
    echo "[START] map=${map} seed=${seed} gpu=${gpu} PID=${pid} log=${log_file}" >&2
    echo "$pid"
}

# ---------- 等待某个 PID 结束，然后再等 10s ----------
wait_and_delay() {
    local pid="$1"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        wait "$pid"
    fi
    echo "[DONE ] PID=${pid} 已结束，等待 60s 后启动下一个任务..."
    sleep 60
}

# ---------- 主循环 ----------
running_pids=()

for map in "${MAPS[@]}"; do
    echo "========================================"
    echo "[MAP ] 开始处理地图: ${map}"
    echo "========================================"

    for seed in "${SEEDS[@]}"; do
        if [ -f "${SCRIPT_DIR}/STOP" ]; then
            echo "[STOP] 检测到 STOP 文件，停止派发新任务"
            break 2
        fi

        if already_done "$map" "$seed"; then
            echo "[SKIP] ${map} seed${seed} 已有日志，跳过"
            continue
        fi

        # 并发数达到上限时，轮询等待任意一个任务结束
        while [ ${#running_pids[@]} -ge "$MAX_PARALLEL" ]; do
            new_pids=()
            found=0
            for pid in "${running_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    echo "[DONE ] PID=${pid} 已结束，等待 60s 后启动下一个任务..."
                    sleep 60
                    found=1
                fi
            done
            running_pids=("${new_pids[@]}")
            [ $found -eq 0 ] && sleep 3
        done

        # 启动前再次实时检查，防止其他进程已经完成了这个任务
        if already_done "$map" "$seed"; then
            echo "[SKIP] ${map} seed${seed} 在启动前检测到已有日志，跳过"
            continue
        fi

        new_pid=$(launch_task "$map" "$seed")
        if [ -n "$new_pid" ]; then
            running_pids+=("$new_pid")
            sleep 60
        fi
    done

done

# 等待所有剩余任务完成
echo "[WAIT] 等待所有剩余任务完成..."
for pid in "${running_pids[@]}"; do
    wait_and_delay "$pid"
done

echo "========================================"
echo "[ALL ] 所有任务已完成"
echo "========================================"

# 最终检查：查看是否有遗漏的任务并启动
echo ""
echo "========================================"
echo "[CHECK] 最终检查遗漏的任务..."
echo "========================================"
missing_tasks=()
for map in "${MAPS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        if ! already_done "$map" "$seed"; then
            echo "[MISSING] ${map} seed${seed} 没有日志，准备启动"
            missing_tasks+=("$map:$seed")
        fi
    done
done

if [ ${#missing_tasks[@]} -eq 0 ]; then
    echo "[SUCCESS] 所有任务都已完成，没有遗漏！"
else
    echo "[RETRY] 发现 ${#missing_tasks[@]} 个遗漏任务，开始运行..."
    running_pids=()
    for task in "${missing_tasks[@]}"; do
        map="${task%%:*}"
        seed="${task##*:}"

        # 并发控制
        while [ ${#running_pids[@]} -ge "$MAX_PARALLEL" ]; do
            new_pids=()
            found=0
            for pid in "${running_pids[@]}"; do
                if kill -0 "$pid" 2>/dev/null; then
                    new_pids+=("$pid")
                else
                    echo "[DONE ] PID=${pid} 已结束，等待 60s 后启动下一个任务..."
                    sleep 60
                    found=1
                fi
            done
            running_pids=("${new_pids[@]}")
            [ $found -eq 0 ] && sleep 3
        done

        new_pid=$(launch_task "$map" "$seed")
        if [ -n "$new_pid" ]; then
            running_pids+=("$new_pid")
            sleep 60
        fi
    done

    # 等待补充任务完成
    echo "[WAIT] 等待补充任务完成..."
    for pid in "${running_pids[@]}"; do
        wait_and_delay "$pid"
    done
    echo "[DONE] 所有补充任务已完成"
fi
echo "========================================"
