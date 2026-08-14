# MSCM-attn-nlevel

## 项目说明

基于 MSCM-attn 项目，实现按地图名称自适应配置 chunk size 和 n_levels。

## 主要修改

### 1. 按地图名称配置 chunk size

在 `ContinuumMemorySystem` 类中添加 `MAP_NAME_TO_CHUNK_SIZE` 字典：

```python
MAP_NAME_TO_CHUNK_SIZE = {
    # 对数尺度表现优异的地图
    "5m_vs_6m": 2,        # [1, 2, 5] - 77.6%
    "1c3s5z": 3,          # [1, 3, 9] - 99.3% ✓ 最佳
    "10m_vs_11m": 4,      # [1, 4, 10]
    "MMM2": 3,            # [1, 3, 10] - 91.8% ✓ 最佳
    
    # 默认值表现更好的地图
    "6h_vs_8z": 4,        # [1, 4, 6] - 98.3% ✓ 最佳
    "3s5z": 4,            # [1, 4, 8]
    "8m_vs_9m": 4,        # [1, 4, 8]
    "3s5z_vs_3s6z": 3,    # [1, 3, 8]
    
    # 大规模地图需要更多层级
    "27m_vs_30m": 3,      # [1, 3, 9, 27] (4层)
}
```

### 2. 按地图名称配置 n_levels

针对大规模地图 27m_vs_30m 使用 4 层结构：

```python
MAP_NAME_TO_LEVELS = {
    "27m_vs_30m": 4,      # 大规模地图使用 4 层
}
```

### 3. 参数传递链

- `MultiAgentTransformer.__init__()` 添加 `map_name` 参数
- `Encoder.__init__()` 添加 `map_name` 参数
- `EncodeBlock.__init__()` 添加 `map_name` 参数
- `ContinuumMemorySystem.__init__()` 添加 `map_name` 参数并实现自适应配置

### 4. 配置示例

**27m_vs_30m (4层结构)**:
- n_levels = 4
- mid_chunk_size = 3
- chunk_sizes = [1, 3, 9, 27]

**其他地图 (3层结构)**:
- n_levels = 3
- mid_chunk_size = 根据地图名称查表
- chunk_sizes = [1, mid_chunk_size, N]

## 使用方法

在训练脚本中传入 `map_name` 参数：

```python
model = MultiAgentTransformer(
    state_dim=state_dim,
    obs_dim=obs_dim,
    action_dim=action_dim,
    n_agent=n_agent,
    n_block=n_block,
    n_embd=n_embd,
    n_head=n_head,
    map_name="27m_vs_30m"  # 传入地图名称
)
```

## 实验依据

配置基于以下实验结果：

| 地图 | 最佳配置 | 胜率 |
|------|---------|------|
| 1c3s5z | chunk=3 | 99.3±1.0 |
| MMM2 | chunk=3 | 91.8±5.2 |
| 6h_vs_8z | chunk=4 | 98.3±1.3 |
| 27m_vs_30m | 4层 | 待验证 |

## 待完成

- [ ] 在 `transformer_policy.py` 中传入 `map_name` 参数
- [ ] 从环境配置中获取地图名称
- [ ] 验证 27m_vs_30m 的 4 层配置效果
