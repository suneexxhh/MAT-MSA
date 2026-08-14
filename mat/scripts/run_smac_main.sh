#!/bin/sh
# =============================================================
# 多地图 MAT 训练启动器
# 用法:
#   ./run_smac.sh --map <map_name> [--seed <seed>] [--gpu <gpu_id>] [--cms_mid_chunk_size <size>]
#   ./run_smac.sh -m <map_name> [-s <seed>] [-g <gpu_id>] [-c <size>]
#
# 示例:
#   ./run_smac.sh --map 6h_vs_8z --seed 1 --gpu 0
#   ./run_smac_main.sh -m 27m_vs_30m -s 1 -g 1 --cms_mid_chunk_size 9
# =============================================================

# ---------- 默认值 ----------
MAP=""
SEED=1
GPU=0
CMS_MID_CHUNK_SIZE=4

# ---------- 解析参数 ----------
while [ $# -gt 0 ]; do
    case "$1" in
        --map|-m)
            MAP="$2"; shift 2 ;;
        --seed|-s)
            SEED="$2"; shift 2 ;;
        --gpu|-g)
            GPU="$2"; shift 2 ;;
        --cms_mid_chunk_size|-c)
            CMS_MID_CHUNK_SIZE="$2"; shift 2 ;;
        --)
            shift; break ;;
        -*)
            echo "[ERROR] 未知参数: $1"
            echo "用法: $0 --map <map_name> [--seed <seed>] [--gpu <gpu_id>] [--cms_mid_chunk_size <size>]"
            exit 1 ;;
        *)
            break ;;
    esac
done

# ---------- 校验必填参数 ----------
if [ -z "$MAP" ]; then
    echo "[ERROR] 必须指定地图名称，例如: --map 6h_vs_8z"
    echo "用法: $0 --map <map_name> [--seed <seed>] [--gpu <gpu_id>] [--cms_mid_chunk_size <size>]"
    exit 1
fi

# ---------- 定位对应地图的 .sh 文件 ----------
# 脚本文件与本启动器放在同一目录，可按需修改 SCRIPT_DIR
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MAP_SCRIPT="${SCRIPT_DIR}/train_smac_${MAP}.sh"

if [ ! -f "$MAP_SCRIPT" ]; then
    echo "[ERROR] 未找到地图脚本: ${MAP_SCRIPT}"
    echo "请确认 train_smac_${MAP}.sh 与本脚本位于同一目录: ${SCRIPT_DIR}"
    exit 1
fi

if [ ! -x "$MAP_SCRIPT" ]; then
    echo "[INFO] 正在为 ${MAP_SCRIPT} 添加可执行权限..."
    chmod +x "$MAP_SCRIPT"
fi

# ---------- 生成日志文件名 ----------
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$LOG_DIR"

# 如果 chunk size 不是默认值 4，则在日志名中添加标识
if [ "$CMS_MID_CHUNK_SIZE" != "4" ]; then
    CHUNK_SUFFIX="_chunk${CMS_MID_CHUNK_SIZE}"
else
    CHUNK_SUFFIX=""
fi

LOG_FILE="${LOG_DIR}/${MAP}_seed${SEED}_gpu${GPU}${CHUNK_SUFFIX}_${TIMESTAMP}.log"

# ---------- 启动训练 ----------
echo "[${MAP}] GPU=${GPU} Seed=${SEED} CMS_MID_CHUNK_SIZE=${CMS_MID_CHUNK_SIZE}"

nohup sh "$MAP_SCRIPT" "$GPU" "$SEED" "$CMS_MID_CHUNK_SIZE" > "$LOG_FILE" 2>&1 &

TRAIN_PID=$!
PID_FILE="${LOG_DIR}/${MAP}_seed${SEED}_gpu${GPU}${CHUNK_SUFFIX}_${TIMESTAMP}.pid"
echo "$TRAIN_PID" > "$PID_FILE"
echo "Started PID=${TRAIN_PID} | Log: tail -f ${LOG_FILE}"