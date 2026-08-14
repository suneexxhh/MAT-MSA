# CMS 中间层 chunk_size 参数使用说明

## 概述
已成功添加 `--cms_mid_chunk_size` 参数，用于在运行时自定义 CMS (Continuum Memory System) 中间层的 chunk_size。默认值为 4。

## 修改的文件

### 1. 核心代码文件
- `mat/algorithms/mat/algorithm/ma_transformer.py`
  - 修改 `ContinuumMemorySystem.__init__()` 添加 `mid_chunk_size` 参数
  - 修改 `EncodeBlock.__init__()` 添加 `mid_chunk_size` 参数
  - 修改 `Encoder.__init__()` 添加 `mid_chunk_size` 参数
  - 修改 `MultiAgentTransformer.__init__()` 添加 `mid_chunk_size` 参数

- `mat/algorithms/mat/algorithm/transformer_policy.py`
  - 修改 `TransformerPolicy.__init__()` 传递 `mid_chunk_size` 参数

### 2. 训练脚本 (Python)
所有环境的训练脚本都已添加 `--cms_mid_chunk_size` 参数：
- `mat/scripts/train/train_smac.py` - SMAC 环境
- `mat/scripts/train/train_smac_multi.py` - SMAC 多地图环境
- `mat/scripts/train/train_football.py` - Google Research Football 环境
- `mat/scripts/train/train_mpe.py` - Multi-Agent Particle Environment
- `mat/scripts/train/train_hands.py` - Dexterous Hands 环境
- `mat/scripts/train/train_mujoco.py` - MuJoCo 环境

### 3. Shell 运行脚本
**SMAC 相关：**
- `mat/scripts/run_smac_main.sh` - 主启动器
- 所有单独的地图训练脚本（15个）：
  - `train_smac_3s5z.sh`
  - `train_smac_6h_vs_8z.sh`
  - `train_smac_3m.sh`
  - `train_smac_8m.sh`
  - `train_smac_10m_vs_11m.sh`
  - `train_smac_27m_vs_30m.sh`
  - `train_smac_MMM.sh`
  - `train_smac_MMM2.sh`
  - `train_smac_1c3s5z.sh`
  - `train_smac_3s5z_vs_3s6z.sh`
  - `train_smac_3s_vs_5z.sh`
  - `train_smac_5m_vs_6m.sh`
  - `train_smac_8m_vs_9m.sh`
  - `train_smac_multi.sh`
  - `train_smac_few_shot.sh`

**其他环境：**
- `mat/scripts/train_football.sh` - Football 环境
- `mat/scripts/train_mpe.sh` - MPE 环境
- `mat/scripts/train_hands.sh` - Hands 环境
- `mat/scripts/train_mujoco.sh` - MuJoCo 环境

## 使用方法

### SMAC 环境

#### 方法 1: 使用主启动器 (推荐)
```bash
# 使用默认值 (chunk_size=4)
./run_smac_main.sh --map 3s5z --seed 1 --gpu 0

# 自定义 chunk_size 为 8
./run_smac_main.sh --map 3s5z --seed 1 --gpu 0 --cms_mid_chunk_size 8

# 使用短参数
./run_smac_main.sh -m 6h_vs_8z -s 2 -g 1 -c 16
```

#### 方法 2: 直接调用单独的地图脚本
```bash
# 使用默认值 (chunk_size=4)
./train_smac_3s5z.sh 0 1

# 自定义 chunk_size 为 8
./train_smac_3s5z.sh 0 1 8

# 参数说明: GPU_ID SEED CMS_MID_CHUNK_SIZE
./train_smac_6h_vs_8z.sh 0 1 16
```

#### 方法 3: 直接使用 Python 命令
```bash
cd mat/scripts
CUDA_VISIBLE_DEVICES=0 python train/train_smac.py \
  --env_name StarCraft2 \
  --algorithm_name mat \
  --experiment_name test \
  --map_name 3s5z \
  --seed 1 \
  --cms_mid_chunk_size 8 \
  --n_training_threads 16 \
  --n_rollout_threads 32 \
  --num_mini_batch 1 \
  --episode_length 100 \
  --num_env_steps 5000000 \
  --lr 5e-4 \
  --ppo_epoch 10 \
  --clip_param 0.05 \
  --save_interval 100000 \
  --use_value_active_masks \
  --use_eval
```

### Football 环境
```bash
# 修改 train_football.sh 中的 cms_mid_chunk_size 变量
# 或直接使用 Python 命令
cd mat/scripts
CUDA_VISIBLE_DEVICES=0 python train/train_football.py \
  --env_name football \
  --algorithm_name mat \
  --scenario academy_counterattack_easy \
  --n_agent 4 \
  --cms_mid_chunk_size 8 \
  --seed 1 \
  --lr 5e-4 \
  --use_eval
```

### MPE 环境
```bash
# 修改 train_mpe.sh 中的 cms_mid_chunk_size 变量
# 或直接使用 Python 命令
cd mat/scripts
CUDA_VISIBLE_DEVICES=0 python train/train_mpe.py \
  --env_name MPE \
  --algorithm_name mat \
  --scenario_name simple_spread \
  --num_agents 3 \
  --cms_mid_chunk_size 8 \
  --seed 1 \
  --use_eval
```

### Hands 环境
```bash
# 修改 train_hands.sh 中的 cms_mid_chunk_size 变量
# 或直接使用 Python 命令
cd mat/scripts
CUDA_VISIBLE_DEVICES=0 python train/train_hands.py \
  --env_name hands \
  --algorithm_name mat \
  --task ShadowHandCatchOver2Underarm \
  --cms_mid_chunk_size 8 \
  --seed 1
```

### MuJoCo 环境
```bash
# 修改 train_mujoco.sh 中的 cms_mid_chunk_size 变量
# 或直接使用 Python 命令
cd mat/scripts
CUDA_VISIBLE_DEVICES=0 python train/train_mujoco.py \
  --env_name mujoco \
  --algorithm_name mat \
  --scenario HalfCheetah-v2 \
  --agent_conf 6x1 \
  --cms_mid_chunk_size 8 \
  --seed 1 \
  --use_eval
```

## 参数说明

- **参数名**: `--cms_mid_chunk_size` 或 `-c` (在 run_smac_main.sh 中)
- **类型**: 整数 (int)
- **默认值**: 4
- **作用**: 控制 CMS 中间层的时间尺度
  - 值越小，记忆范围越局部
  - 值越大，记忆范围越全局
  - 建议范围: 2-16

## CMS 层级结构

当 `n_levels=3` 时，三个层级的 chunk_size 分别为：
- **Level 0 (快记忆)**: chunk_size = 1 (固定)
- **Level 1 (中记忆)**: chunk_size = `cms_mid_chunk_size` (可自定义，默认 4)
- **Level 2 (慢记忆)**: chunk_size = n_agents (全局，固定)

如果有更多层级，中间层会按指数增长：
- Level 1: `cms_mid_chunk_size`
- Level 2: `cms_mid_chunk_size^2`
- Level 3: `cms_mid_chunk_size^3`
- ...
- Level n-1: n_agents (全局)

## 示例

### 实验不同的 chunk_size 值
```bash
# 测试 chunk_size = 2 (更局部的记忆)
./run_smac_main.sh -m 3s5z -s 1 -g 0 -c 2

# 测试 chunk_size = 8 (更全局的记忆)
./run_smac_main.sh -m 3s5z -s 1 -g 0 -c 8

# 测试 chunk_size = 16 (非常全局的记忆)
./run_smac_main.sh -m 3s5z -s 1 -g 0 -c 16
```

## 注意事项

1. 所有脚本都已更新，向后兼容旧的调用方式（不传第三个参数时使用默认值 4）
2. 修改后的代码保持了原有的功能，只是增加了可配置性
3. 建议根据具体任务和智能体数量调整该参数以获得最佳性能
