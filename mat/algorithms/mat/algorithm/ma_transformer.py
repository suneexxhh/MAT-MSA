import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torch.distributions import Categorical
from mat.algorithms.utils.util import check, init
from mat.algorithms.utils.transformer_act import discrete_autoregreesive_act
from mat.algorithms.utils.transformer_act import discrete_parallel_act
from mat.algorithms.utils.transformer_act import continuous_autoregreesive_act
from mat.algorithms.utils.transformer_act import continuous_parallel_act
from mat.algorithms.mat.algorithm.scale_config import (
    TABLE1_SCALE_CONFIG as TABLE1_SCALE_CONFIG_MAP,
    get_chunk_sizes,
)

def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)


class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0
        self.masked = masked
        self.n_head = n_head
        # key, query, value projections for all heads
        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))
        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))
        # if self.masked:
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(n_agent + 1, n_agent + 1))
                             .view(1, 1, n_agent + 1, n_agent + 1))

        self.att_bp = None

    def forward(self, key, value, query):
        B, L, D = query.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)  # (B, nh, L, hs)

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        # self.att_bp = F.softmax(att, dim=-1)

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent, mid_chunk_size=4, map_name=None, n_levels=3, use_map_adaptive=True):
        super(EncodeBlock, self).__init__()

        # ========== 打印参数传递信息 ==========
        print("\n" + "="*80)
        print("[4/5] EncodeBlock: Received parameters")
        print("="*80)
        print(f"  map_name:          {map_name}")
        print(f"  mid_chunk_size:    {mid_chunk_size}")
        print(f"  n_levels:          {n_levels}")
        print(f"  n_agent:           {n_agent}")
        print(f"  use_map_adaptive:  {use_map_adaptive}")
        print("="*80 + "\n")

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # 用CMS替换Self-Attention
        self.cms = ContinuumMemorySystem(
            n_embd=n_embd,
            n_levels=n_levels,
            n_agents=n_agent,
            dropout=0.0,
            mid_chunk_size=mid_chunk_size,
            map_name=map_name,
            use_map_adaptive=use_map_adaptive
        )
        # 恢复MLP
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.cms(x))  # CMS替代Self-Attention
        x = self.ln2(x + self.mlp(x))  # MLP做特征变换
        return x
class CMSLevel(nn.Module):
    """
    CMS 单个记忆层级 (One Level of Continuum Memory System)
    ─────────────────────────────────────────────────────────
    对应论文 Section 4 / Fig.3 中的 "Fast Memory" 组件。

    操作流程:
      1. 读操作 (Read):  q = W_q(x)
                          k = W_k(context)
                          v = W_v(context)
                          read = softmax(q·kᵀ/√d) · v
      2. 写操作 (Write): 更新内部 key-value 记忆
                         (此处为 in-context stateless 模式，
                          对应论文的 "boundary-target online" 的简化版)
      3. 输出投影:       out = W_out(read)

    Args:
        n_embd   (int): 输入/输出维度 D
        head_dim (int): 记忆槽内部维度 d_m
        chunk_size (int): 该层级的时间尺度 (影响 mask 范围)
        dropout  (float): dropout 概率
    """

    def __init__(self, n_embd: int, head_dim: int, chunk_size: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        self.head_dim = head_dim
        self.chunk_size = chunk_size
        self.scale = head_dim ** -0.5

        # Read: query projection
        self.W_q = nn.Linear(n_embd, head_dim, bias=False)
        # Write: key & value projections
        self.W_k = nn.Linear(n_embd, head_dim, bias=False)
        self.W_v = nn.Linear(n_embd, head_dim, bias=False)
        # Output projection back to n_embd
        self.W_out = nn.Linear(head_dim, n_embd, bias=False)

        self.dropout = nn.Dropout(dropout)

        # 初始化 (paper 使用 Xavier / 正交)
        for w in [self.W_q, self.W_k, self.W_v, self.W_out]:
            nn.init.xavier_uniform_(w.weight)

    def _chunk_mask(self, N: int, device) -> torch.Tensor:
        """
        生成 chunk-local 的 attention mask。
        当 chunk_size >= N 时为全局 attention（等价于无掩码）。
        当 chunk_size < N 时，每个位置只看本 chunk 内的 token。

        返回 (N, N) bool 张量，True = 允许 attend。
        """
        if self.chunk_size >= N:
            return None  # 全局 attention，无限制
        mask = torch.zeros(N, N, dtype=torch.bool, device=device)
        for i in range(N):
            start = (i // self.chunk_size) * self.chunk_size
            end = min(start + self.chunk_size, N)
            mask[i, start:end] = True
        return mask  # (N, N)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x       : (B, N, D) 当前输入
        Returns:
            out     : (B, N, D)
        """
        B, N, D = x.shape

        q = self.W_q(x)  # (B, N, d_m)
        k = self.W_k(x)  # (B, N, d_m)
        v = self.W_v(x)  # (B, N, d_m)

        # Scaled dot-product attention (联想记忆读取)
        attn_logits = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B, N, N)

        # 应用 chunk mask（若有）
        chunk_mask = self._chunk_mask(N, x.device)
        if chunk_mask is not None:
            # mask: True=允许, False=禁止
            attn_logits = attn_logits.masked_fill(~chunk_mask.unsqueeze(0), -1e9)

        attn = F.softmax(attn_logits, dim=-1)  # (B, N, N)
        attn = self.dropout(attn)

        read = torch.bmm(attn, v)  # (B, N, d_m)
        out = self.W_out(read)  # (B, N, D)
        return out

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)
        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Encoder(nn.Module):

    def __init__(self, state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, mid_chunk_size=4, map_name=None, n_levels=3, use_map_adaptive=True):
        super(Encoder, self).__init__()

        # ========== 打印参数传递信息 ==========
        print("\n" + "="*80)
        print("[3/5] Encoder: Received parameters")
        print("="*80)
        print(f"  map_name:          {map_name}")
        print(f"  mid_chunk_size:    {mid_chunk_size}")
        print(f"  n_levels:          {n_levels}")
        print(f"  n_agent:           {n_agent}")
        print(f"  use_map_adaptive:  {use_map_adaptive}")
        print("="*80 + "\n")

        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent
        self.encode_state = encode_state
        # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))

        self.state_encoder = nn.Sequential(nn.LayerNorm(state_dim),
                                           init_(nn.Linear(state_dim, n_embd), activate=True), nn.GELU())
        self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent, mid_chunk_size, map_name, n_levels, use_map_adaptive) for _ in range(n_block)])
        self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                  init_(nn.Linear(n_embd, 1)))

    def forward(self, state, obs):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        if self.encode_state:
            state_embeddings = self.state_encoder(state)
            x = state_embeddings
        else:
            obs_embeddings = self.obs_encoder(obs)
            x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return v_loc, rep


class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.dec_actor = dec_actor
        self.share_actor = share_actor
        self.action_type = action_type

        if action_type != 'Discrete':
            log_std = torch.ones(action_dim)
            # log_std = torch.zeros(action_dim)
            self.log_std = torch.nn.Parameter(log_std)
            # self.log_std = torch.nn.Parameter(torch.zeros(action_dim))

        if self.dec_actor:
            if self.share_actor:
                print("mac_dec!!!!!")
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # self.agent_id_emb = nn.Parameter(torch.zeros(1, n_agent, n_embd))
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                                                    nn.GELU())
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())
            self.obs_encoder = nn.Sequential(nn.LayerNorm(obs_dim),
                                             init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU())
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))

    def zero_std(self, device):
        if self.action_type != 'Discrete':
            log_std = torch.zeros(self.action_dim).to(device)
            self.log_std.data = log_std

    # state, action, and return
    def forward(self, action, obs_rep, obs):
        # action: (batch, n_agent, action_dim), one-hot/logits?
        # obs_rep: (batch, n_agent, n_embd)
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit

class ContinuumMemorySystem(nn.Module):
    """
    CMS (Continuum Memory System) — MLP 的替代模块
    ──────────────────────────────────────────────────────────────
    论文核心创新：用多时间尺度的联想记忆替换 Transformer 中的 FFN/MLP。

    架构 (n_levels=3 为例):

      x ──┬─► Level-0 (chunk_size=1, 快记忆) ──► out_0 ─────────────────────┐
          │                                          │                        │
          ├─► Level-1 (chunk_size=mid, 中记忆) ─► consolidate(out_0, out_1)   │
          │   context=out_0 (跨级别整合)            │                        │
          │                                          ▼                        │
          └─► Level-2 (chunk_size=N, 慢记忆) ─► consolidate(out_1, out_2)   │
              context=out_1                                                   │
                                                                              ▼
                          all level outputs ─► concat ─► gate ─► LayerNorm ─► output (D)

    跨级别整合公式 (对应论文 "memory consolidation"):
        out_l = sigmoid(γ_l) * out_{l-1} + (1 - sigmoid(γ_l)) * level_l(x, ctx=out_{l-1})
        其中 γ_l 是可学习的衰减门控参数。

    按地图名称自适应配置 (Map-adaptive Configuration):
        根据实验结果为每个地图选择最佳 mid_chunk_size 和 n_levels

    Args:
        n_embd    (int)  : 输入输出维度
        n_levels  (int)  : 记忆层级数 (默认 3, 27m_vs_30m 使用 4)
        n_agents  (int)  : 智能体数量 (用于最慢层的 chunk_size, 必须传入实际值)
        dropout   (float): dropout 概率
        map_name  (str)  : 地图名称 (用于自适应配置)
    """

    # Backward-compatible access to the Table 1 map configuration.
    TABLE1_SCALE_CONFIG = TABLE1_SCALE_CONFIG_MAP

    def __init__(self, n_embd: int, n_levels: int = 3,
                 n_agents: int = None, dropout: float = 0.0,
                 mid_chunk_size: int = 4, map_name: str = None,
                 use_map_adaptive: bool = True):
        super().__init__()
        assert n_agents is not None, "n_agents must be specified (actual number of agents in the environment)"

        # 保存原始参数用于打印
        original_mid_chunk = mid_chunk_size
        original_n_levels = n_levels

        n_levels, mid_chunk_size, chunk_sizes = get_chunk_sizes(
            map_name=map_name,
            n_agents=n_agents,
            n_levels=n_levels,
            chunk_size=mid_chunk_size,
            use_map_adaptive=use_map_adaptive,
        )

        self.n_levels = n_levels
        self.n_embd = n_embd

        # 每个级别的 head_dim（可以不同）
        # 这里取 n_embd//2，与原 MLP 的 4*n_embd 中间维度保持合理比较
        head_dim = max(n_embd // 2, 64)

        # ========== 打印实际配置参数（只打印一次）==========
        if not hasattr(ContinuumMemorySystem, '_config_printed'):
            print("\n" + "="*80)
            print("ContinuumMemorySystem Configuration (ACTUAL RUNTIME PARAMETERS)")
            print("="*80)
            print(f"  Map Name:           {map_name if map_name else 'N/A (using defaults)'}")
            print(f"  Num Agents:         {n_agents}")

            # 显示配置来源
            if map_name and map_name in TABLE1_SCALE_CONFIG_MAP:
                print(f"  Config Source:      MAP-ADAPTIVE ✓")
            else:
                print(f"  Config Source:      DEFAULT")

            print(f"\n  n_levels:           {n_levels}")
            print(f"  mid_chunk_size:     {mid_chunk_size}")
            print(f"  dropout:            {dropout}")
            print(f"  Chunk Sizes:        {chunk_sizes}")

            # 如果有覆盖，显示对比
            if original_n_levels != n_levels or original_mid_chunk != mid_chunk_size:
                print(f"\n  [Auto-Override Applied]")
                if original_n_levels != n_levels:
                    print(f"    n_levels:       {original_n_levels} → {n_levels}")
                if original_mid_chunk != mid_chunk_size:
                    print(f"    mid_chunk_size: {original_mid_chunk} → {mid_chunk_size}")

            print("="*80 + "\n")
            ContinuumMemorySystem._config_printed = True

        self.levels = nn.ModuleList([
            CMSLevel(n_embd, head_dim, chunk_size=chunk_sizes[l], dropout=dropout)
            for l in range(n_levels)
        ])

        # 可学习的跨级别衰减门控 γ_l  (论文中的 decay/consolidation gate)
        # 初始化为 0 → sigmoid(0) = 0.5，平衡快慢记忆
        self.decay_gates = nn.Parameter(torch.zeros(n_levels))

        # 多级输出的门控融合: n_levels * D → D
        self.output_proj = nn.Linear(n_embd * n_levels, n_embd)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        # 输出 LayerNorm（对应论文的 post-norm）
        self.output_norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, D)  B=batch, N=n_agents, D=n_embd
        Returns:
            out: (B, N, D)
        """
        level_outputs = []

        for l, level in enumerate(self.levels):
            # 每层都直接处理原始输入x，使用不同的chunk_size
            out = level(x)  # (B, N, D)
            level_outputs.append(out)

        # 所有层级输出拼接后融合
        combined = torch.cat(level_outputs, dim=-1)  # (B, N, D*n_levels)
        fused = self.output_proj(combined)  # (B, N, D)
        return self.output_norm(fused)
class MultiAgentTransformer(nn.Module):

    def __init__(self, state_dim, obs_dim, action_dim, n_agent,
                 n_block, n_embd, n_head, encode_state=False, device=torch.device("cpu"),
                 action_type='Discrete', dec_actor=False, share_actor=False, mid_chunk_size=4, map_name=None, n_levels=3, use_map_adaptive=True):
        super(MultiAgentTransformer, self).__init__()

        # ========== 打印参数传递信息 ==========
        print("\n" + "="*80)
        print("[2/5] MultiAgentTransformer: Received parameters")
        print("="*80)
        print(f"  map_name:          {map_name}")
        print(f"  mid_chunk_size:    {mid_chunk_size}")
        print(f"  n_levels:          {n_levels}")
        print(f"  n_agent:           {n_agent}")
        print(f"  use_map_adaptive:  {use_map_adaptive}")
        print("="*80 + "\n")

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.action_type = action_type
        self.device = device

        # state unused
        state_dim = 37

        self.encoder = Encoder(state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state, mid_chunk_size, map_name, n_levels, use_map_adaptive)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def zero_std(self):
        if self.action_type != 'Discrete':
            self.decoder.zero_std(self.device)

    def forward(self, state, obs, action, available_actions=None):
        # state: (batch, n_agent, state_dim)
        # obs: (batch, n_agent, obs_dim)
        # action: (batch, n_agent, 1)
        # available_actions: (batch, n_agent, act_dim)

        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(state)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, state, obs, available_actions=None, deterministic=False):
        # state unused
        ori_shape = np.shape(obs)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(state, obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, state, obs, available_actions=None):
        # state unused
        ori_shape = np.shape(state)
        state = np.zeros((*ori_shape[:-1], 37), dtype=np.float32)

        state = check(state).to(**self.tpdv)
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(state, obs)
        return v_tot

