# LLM 算法实习面试准备指南

> 基于 Vime / Slime / vLLM 项目的深度分析，面向 LLM & Agent 应用/后训练方向的面试准备

---

## 目录

1. [项目全景与架构概览](#1-项目全景与架构概览)
2. [LLM Post-Training 核心概念](#2-llm-post-training-核心概念)
3. [强化学习算法深度解析](#3-强化学习算法深度解析)
4. [推理引擎核心技术](#4-推理引擎核心技术)
5. [分布式训练与并行策略](#5-分布式训练与并行策略)
6. [Agent 训练关键技术](#6-agent-训练关键技术)
7. [面试高频考点与深挖点](#7-面试高频考点与深挖点)
8. [代码级面试准备](#8-代码级面试准备)
9. [系统设计面试准备](#9-系统设计面试准备)
10. [实战项目介绍模板](#10-实战项目介绍模板)

---

## 1. 项目全景与架构概览

### 1.1 三大项目定位

```
┌─────────────────────────────────────────────────────────────────┐
│                        vLLM (推理引擎)                          │
│  - PagedAttention, Continuous Batching, Tensor Parallelism      │
│  - 高性能 LLM 推理与服务                                         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ 推理后端集成
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Vime (Post-Training框架)                 │
│  - 基于 slime，使用 vLLM 作为 rollout 后端                       │
│  - Megatron 训练 + vLLM 推理的无缝桥接                           │
│  - 支持 GRPO/PPO/REINFORCE++ 等多种 RL 算法                     │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ 衍生自
                              │
┌─────────────────────────────────────────────────────────────────┐
│                        Slime (Post-Training框架)                │
│  - 清华 THUDM 团队开发                                          │
│  - Megatron + SGLang 深度集成                                   │
│  - GLM 系列模型背后的 RL 训练框架                                │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Vime 核心架构

```python
# Vime 三大核心模块
┌───────────────────────────────────────────────────────────────┐
│                    Training (Megatron)                        │
│  - MegatronTrainRayActor: 训练核心 Ray Actor                  │
│  - 权重快照管理: TensorBackuper 支持多版本模型切换              │
│  - 支持 Offload/Colocate 模式                                 │
└───────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Weight Sync     │
                    │  (NCCL/Disk/      │
                    │   Delta/Tensor)   │
                    └─────────┬─────────┘
                              │
┌───────────────────────────────────────────────────────────────┐
│                    Rollout (vLLM + Router)                    │
│  - RolloutManager: 管理 vLLM 引擎组                          │
│  - ServerGroup: 支持 PD Disaggregation                       │
│  - 异步数据生成 + Reward 计算                                  │
└───────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │   Data Buffer     │
                    │  (DataSource)     │
                    └─────────┬─────────┘
                              │
┌───────────────────────────────────────────────────────────────┐
│                    Data Buffer                                │
│  - RolloutDataSourceWithBuffer: 支持 Partial Rollout          │
│  - 动态采样过滤 (DAPO-style)                                  │
│  - 数据平衡与调度                                              │
└───────────────────────────────────────────────────────────────┘
```

### 1.3 关键代码路径速查

| 模块 | 关键文件 | 面试考察点 |
|------|---------|-----------|
| RL 算法实现 | `vime/utils/ppo_utils.py` | KL 散度估计、GAE、Policy Loss |
| 损失函数 | `vime/backends/megatron_utils/loss.py` | PPO/GSPO/GRPO 损失计算 |
| Agent 轨迹管理 | `vime/agent/trajectory.py` | 多轮对话、Token 漂移处理 |
| 权重同步 | `vime/backends/megatron_utils/update_weight/` | NCCL/Delta 同步策略 |
| vLLM 集成 | `vime/backends/vllm_utils/` | 引擎管理、参数透传 |
| 数据源 | `vime/rollout/data_source.py` | Buffer 机制、Partial Rollout |

---

## 2. LLM Post-Training 核心概念

### 2.1 Post-Training Pipeline

```
Pre-trained LLM → SFT → RLHF/DPO → Alignment → Deployed Model
       ↑              ↑           ↑
       │              │           │
    基座模型      监督微调     强化学习对齐
```

### 2.2 RLHF vs DPO vs GRPO

| 方法 | 核心思想 | 优点 | 缺点 |
|------|---------|------|------|
| **RLHF** | 训练 Reward Model + PPO 优化 | 效果好，灵活 | 需要 RM，训练复杂 |
| **DPO** | 直接用偏好数据优化 | 无需 RM，简单 | 依赖偏好数据质量 |
| **GRPO** | Group Relative Policy Optimization | 无需 Critic，高效 | 需要多次采样 |

### 2.3 GRPO 算法详解（Vime 默认）

**核心思想**：通过组内相对优势估计替代 Critic 网络

```python
# GRPO 的核心计算
def get_grpo_returns(rewards, kl):
    """
    rewards: 每个样本的 reward 值
    kl: 每个 token 的 KL 散度
    """
    returns = []
    for i in range(len(rewards)):
        # GRPO: 直接用 reward 作为 returns（组内相对优势）
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**GRPO vs PPO 的关键区别**：
- PPO: 需要 Critic 网络估计 Value，计算 GAE
- GRPO: 通过组内采样（n_samples_per_prompt）计算相对优势，无需 Critic

### 2.4 KL 散度约束

**为什么需要 KL 约束？**
- 防止策略偏离参考模型太远
- 保持生成质量和多样性
- 避免 reward hacking

**KL 散度估计方法**：

```python
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type):
    """
    支持多种 KL 估计器：
    - k1: log_ratio (简单但可能为负)
    - k2: log_ratio^2 / 2 (Taylor 近似)
    - k3/low_var_kl: exp(-log_ratio) - 1 + log_ratio (非负、无偏、低方差)
    """
    log_ratio = log_probs - log_probs_base
    
    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        kl = log_ratio.exp() - 1 - log_ratio
    
    return kl
```

**面试深挖点**：
1. 为什么 low_var_kl 是更好的估计器？
2. KL 散度在 RLHF 中的作用是什么？
3. 如何选择 KL 系数（kl_coef）？

---

## 3. 强化学习算法深度解析

### 3.1 PPO (Proximal Policy Optimization)

**核心公式**：

```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    """
    PPO Clipped Objective:
    L = -min(ratio * A, clip(ratio, 1-ε, 1+ε) * A)
    
    其中 ratio = π_θ(a|s) / π_θ_old(a|s)
    """
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    
    # Dual-clip PPO (来自 https://arxiv.org/pdf/1912.09729)
    if eps_clip_c is not None:
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1
    
    return pg_losses, clipfrac
```

**面试深挖点**：
1. PPO 中的 clipping 机制有什么作用？
2. Dual-clip PPO 解决了什么问题？
3. 如何理解 importance sampling？

### 3.2 GAE (Generalized Advantage Estimation)

**标准 GAE 实现**：

```python
def vanilla_gae(rewards, values, gamma, lambd):
    """
    GAE 公式：
    δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
    Â_t = δ_t + γλ * Â_{t+1}
    
    其中：
    - gamma: 折扣因子
    - lambd: GAE lambda 参数
    """
    B, T = rewards.shape
    lastgaelam = torch.zeros(B, device=device)
    adv_rev = []
    
    for t in reversed(range(T)):
        next_value = values[:, t + 1] if t < T - 1 else 0.0
        delta = rewards[:, t] + gamma * next_value - values[:, t]
        lastgaelam = delta + gamma * lambd * lastgaelam
        adv_rev.append(lastgaelam)
    
    full_advantages = torch.stack(adv_rev[::-1], dim=1)
    full_returns = full_advantages + values
    return full_advantages, full_returns
```

**Chunked GAE（Vime 创新）**：

```python
def chunked_gae(rewards, values, gamma, lambd, chunk_size=128):
    """
    FlashLinearAttention 启发的并行前缀扫描 GAE：
    - 将序列切分为 chunk_size 的块
    - 块内使用矩阵乘法并行计算（O(C^2)）
    - 块间通过递归状态传播
    - 将 O(T) 顺序依赖降为 O(T/chunk_size)
    """
    # 构建 δ_t = r_t + γ * V_{t+1} - V_t
    next_values = torch.cat([values[:, 1:], torch.zeros(B, 1, ...)], dim=1)
    deltas = rewards + gamma * next_values - values
    
    # 反向序列上的前缀扫描
    w = gamma * lambd
    deltas_rev = torch.flip(deltas, dims=[1])
    
    # 块内并行计算
    deltas_chunks = deltas_rev.view(B, n_chunks, chunk_size)
    S_local_flat = deltas_flat @ M  # M 是块内转移矩阵
    S_local_chunks = S_local_flat.view(B, n_chunks, chunk_size)
    
    # 块间递归传播
    for c in range(n_chunks):
        S_global = S_local + s_prev * pow_vec
        s_prev = S_global[:, -1]
    
    advantages = torch.flip(S_rev, dims=[1])
    returns = advantages + values
    return advantages, returns
```

**面试深挖点**：
1. GAE 中的 lambda 参数有什么作用？
2. Chunked GAE 如何实现并行化？
3. 为什么需要 Advantage Normalization？

### 3.3 REINFORCE++ 算法

```python
def get_reinforce_plus_plus_returns(rewards, kl, loss_masks, response_lengths, 
                                     total_lengths, kl_coef, gamma):
    """
    REINFORCE++ (https://arxiv.org/pdf/2501.03262)
    
    核心改进：
    1. Token-level discounted returns
    2. 使用 KL 散度作为 token-level reward
    3. 最后一个 token 加上外部 reward
    """
    for i in range(len(rewards)):
        # Token-level KL penalty
        masked_kl = full_kl_response * full_mask
        token_level_rewards = -kl_coef * masked_kl
        
        # 最后一个 token 加上外部 reward
        last_idx = full_mask.nonzero(as_tuple=True)[0][-1]
        token_level_rewards[last_idx] += rewards[i]
        
        # 折扣回报计算
        returns_for_seq = torch.zeros_like(token_level_rewards)
        running_return = 0.0
        for t in reversed(range(token_level_rewards.size(0))):
            running_return = token_level_rewards[t] + gamma * running_return
            returns_for_seq[t] = running_return
    
    return final_returns_chunks
```

### 3.4 CISPO (MiniMax-M1)

```python
def compute_cispo_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high):
    """
    CISPO from MiniMax-M1 (https://arxiv.org/abs/2506.13585)
    
    关键区别：
    - PPO: ratio 被 clip，梯度通过 ratio 流动
    - CISPO: ratio 被 stop-gradient clip，梯度通过 log_probs 流动
    
    这意味着 clipped tokens 仍然贡献梯度
    """
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    
    # 关键：梯度通过 log_probs 流动，而不是 ratio
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac = (ratio_truncated != ratio).float()
    
    return pg_losses, clipfrac
```

### 3.5 OPSM (Off-Policy Sequence Masking)

```python
def compute_opsm_mask(args, full_log_probs, full_old_log_probs, advantages, loss_masks):
    """
    Off-Policy Sequence Masking:
    对于 advantage < 0 且 seq_kl > delta 的序列进行 masking
    
    直觉：
    - 如果 advantage < 0（模型表现比平均差）
    - 且 KL 散度很大（策略偏离太远）
    - 则 mask 这个样本，避免有害的梯度更新
    """
    for full_log_prob, full_old_log_prob, advantage, loss_mask in zip(...):
        # 计算序列级 KL
        seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / loss_mask.sum()
        
        # 创建 mask: 0 if (advantage < 0 and seq_kl > delta), else 1
        mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
        opsm_clipfrac += mask.sum() / loss_mask.sum()
        
        opsm_mask_list.append(1 - mask)
    
    return opsm_mask, opsm_clipfrac
```

---

## 4. 推理引擎核心技术

### 4.1 PagedAttention (vLLM 核心创新)

**问题**：传统 KV Cache 需要连续内存分配，导致内存碎片和浪费（60-80%）

**解决方案**：借鉴 OS 虚拟内存分页机制

```
传统方式：
[Token 0][Token 1][Token 2]...[Token N]  (连续内存，浪费多)

PagedAttention：
Block 0: [Token 0][Token 1][Token 2][Token 3]
Block 1: [Token 4][Token 5][Token 6][Token 7]
...
通过 Block Table 映射到物理 GPU 内存
```

**内存布局**：
```python
# KV Cache 内存布局
K Cache: [num_blocks, num_kv_heads, head_size/x, block_size, x]
V Cache: [num_blocks, num_kv_heads, head_size, block_size]

# Block Table 映射
block_table[logical_block] = physical_block_index
```

**Kernel 实现要点**：
1. Query 加载到共享内存
2. 通过块表间接寻址加载 Key
3. QK 点积 + Softmax 归约
4. Value 加载与累积
5. 结果写回全局内存

### 4.2 Continuous Batching

**传统 Static Batching**：
```
Request 1: [Prefill] → [Decode] → [Decode] → ... → [Done]
Request 2:              [Prefill] → [Decode] → ... → [Done]
                         ↑ 等待 Request 1 完成
```

**Continuous Batching**：
```
Step 1: [Req1 Prefill] [Req2 Prefill]
Step 2: [Req1 Decode]  [Req2 Decode]  [Req3 Prefill]  ← 新请求加入
Step 3: [Req1 Decode]  [Req2 Decode]  [Req3 Decode]
...
```

**优势**：最大化 GPU 利用率，减少空闲等待

### 4.3 Chunked Prefill

**问题**：长 prompt 的 prefill 会阻塞 decode 请求

**解决方案**：将长 prompt 分块处理

```python
# 调度器配置
max_num_batched_tokens: int = 2048      # 单步最大 token 数
max_num_partial_prefills: int = 1        # 最大并行 partial prefill 数
long_prefill_token_threshold: int = 0    # 长 prefill 阈值
```

### 4.4 Prefix Caching

**原理**：相同前缀的请求共享 KV Cache 块

```python
# 通过块哈希实现
block_hash = hash(block_content)
# 相同 hash 的块可以复用
```

### 4.5 Tensor Parallelism vs Pipeline Parallelism

| 策略 | 切分方式 | 通信量 | 适用场景 |
|------|---------|--------|---------|
| **Tensor Parallelism** | 模型层内张量分割 | AllReduce | 单节点多卡 |
| **Pipeline Parallelism** | 模型层间流水线 | 点对点 | 跨节点 |
| **Expert Parallelism** | MoE 专家分割 | All2All | MoE 模型 |

---

## 5. 分布式训练与并行策略

### 5.1 Megatron-LM 并行策略

```python
# Vime 支持的并行策略
--tensor-model-parallel-size 2      # TP: 张量并行
--sequence-parallel                 # SP: 序列并行
--pipeline-model-parallel-size 1    # PP: 流水线并行
--context-parallel-size 2           # CP: 上下文并行
--expert-model-parallel-size 1      # EP: 专家并行
--expert-tensor-parallel-size 1     # ETP: 专家张量并行
```

### 5.2 权重同步策略

```python
# 4 种权重同步策略
1. UpdateWeightFromDistributed      # NCCL 直接广播（默认）
2. UpdateWeightFromDisk            # 写磁盘 HF checkpoint
3. UpdateWeightFromDistributedDelta # Delta 权重同步（只传输变化部分）
4. UpdateWeightFromTensor          # Colocate 模式下直接拷贝
```

**Delta Weight Sync 创新**：
```python
# 通过 CPU 快照检测字节级变化
# 只传输变化位置 + 值
# 支持 zstd 压缩
# 对于带宽受限的共享文件系统（<300 MB/s）非常有效
```

### 5.3 Context Parallelism (CP)

**问题**：超长序列的注意力计算

**解决方案**：将序列分割到多个 GPU

```python
# Zigzag CP 分割
GPU 0: [Token 0-127] [Token 256-383] ...
GPU 1: [Token 128-255] [Token 384-511] ...

# 通过 all_gather_with_cp 收集完整序列
full_kl_response = all_gather_with_cp(local_kl_chunk, total_len, response_len)
```

### 5.4 Dynamic Batch Size

```python
# Vime 特有优化
--use-dynamic-batch-size
--max-tokens-per-gpu 4608

# 动态 batch size 的优势：
# 1. 智能 pack 不同长度的样本
# 2. 总 token 数接近 max_tokens_per_gpu
# 3. 提高训练效率
# 4. 严格保证 per sample loss 或 per token loss 正确
```

---

## 6. Agent 训练关键技术

### 6.1 多轮对话轨迹管理

**TrajectoryManager 核心设计**：

```python
class TrajectoryManager:
    """
    管理多轮对话轨迹，处理 TITO (Text-In-Text-Out) re-tokenization 漂移
    
    核心挑战：
    - 每轮对话的 prompt 是前几轮的完整历史
    - 不同轮次的 tokenizer 可能产生不同的 token 序列
    - 需要正确处理 token 漂移
    """
    
    def record_turn(self, sid, turn, prompt_messages, response_message):
        """记录一轮对话"""
        # 1. 找到挂载点
        node, depth = self._find_mount_point(root, prompt_messages)
        
        # 2. 尝试合并 assistant rewrite
        node, depth = self._try_merge_assistant_rewrite(sid, node, prompt_messages, depth)
        
        # 3. 挂载 prompt messages
        node = self._mount_prompt_messages(node, prompt_messages[depth:])
        
        # 4. 附加 assistant leaf
        self._attach_assistant_leaf(sid, node, turn=turn, ...)
```

### 6.2 Token 漂移处理

```python
class DriftKind(enum.Enum):
    CLEAN = "clean"    # 无漂移：直接追加
    REALIGN = "realign" # 短漂移：覆盖并标记 loss_mask=0
    FORK = "fork"       # 大漂移：关闭当前 builder，开新的 fork

def classify_token_drift(self, turn):
    """分类 token 漂移"""
    realign_at = _common_prefix_len(self.tokens, turn.prompt_ids)
    drift = len(self.tokens) - realign_at
    
    if drift == 0:
        return DriftKind.CLEAN
    
    # REALIGN 只在最近 response span 内有效
    start = self.last_response_start_idx
    if start is not None and realign_at >= start and len(turn.output_ids) < self._fork_threshold:
        return DriftKind.REALIGN
    
    return DriftKind.FORK
```

### 6.3 Loss Masking 策略

```python
# Agent 训练中的 Loss Masking
- Model-generated tokens (思考、动作) → loss_mask = 1
- Tool/environment returned tokens (API 结果) → loss_mask = 0

# 具体实现
sample.loss_mask = [1] * len(model_tokens) + [0] * len(tool_tokens)
```

### 6.4 Sandbox 集成

```python
class Sandbox(Protocol):
    """异步沙箱接口"""
    async def exec(self, code: str) -> str: ...
    async def write_file(self, path: str, content: str) -> None: ...
    async def read_file(self, path: str) -> str: ...

class E2BSandbox(Sandbox):
    """基于 E2B 的云端沙箱"""
    # 支持 Python/JS/Shell 等多种语言
    # 内置 RPC 重试和错误处理
```

---

## 7. 面试高频考点与深挖点

### 7.1 基础概念题

**Q1: 什么是 RLHF？为什么需要 RLHF？**

**考察点**：
- SFT 的局限性：只学习模仿，不学习偏好
- Reward Model 的作用：学习人类偏好
- PPO 的作用：优化策略以最大化 reward

**深挖方向**：
- Reward Hacking 问题
- KL 约束的必要性
- RLHF vs DPO 的选择

**Q2: 解释 PPO 中的 clipping 机制**

**考察点**：
- 防止策略更新过大
- 保持训练稳定性
- Dual-clip PPO 的改进

**深挖方向**：
- Clipping 的数学原理
- 为什么需要 Dual-clip？
- 如何选择 clip 参数？

**Q3: 什么是 GAE？为什么需要 GAE？**

**考察点**：
- Bias-Variance Tradeoff
- Lambda 参数的作用
- Chunked GAE 的优化

**深挖方向**：
- GAE 的数学推导
- 如何选择 gamma 和 lambda？
- Chunked GAE 的并行化原理

### 7.2 系统设计题

**Q4: 设计一个大规模 RL 训练系统**

**考察点**：
- Training-Rollout 分离架构
- 权重同步策略
- 资源调度和管理

**深挖方向**：
- Colocate vs Disaggregated 模式
- Delta Weight Sync 的优势
- Partial Rollout 的价值

**Q5: 如何处理多轮 Agent 训练？**

**考察点**：
- TrajectoryManager 设计
- Token 漂移处理
- Loss Masking 策略

**深挖方向**：
- TITO re-tokenization 问题
- DriftKind 分类策略
- Sandbox 集成方案

### 7.3 算法实现题

**Q6: 实现 GRPO 的 returns 计算**

```python
def get_grpo_returns(rewards, kl):
    """
    输入：
    - rewards: [batch_size] 每个样本的 reward
    - kl: [batch_size, seq_len] 每个 token 的 KL 散度
    
    输出：
    - returns: [batch_size, seq_len] 每个 token 的 return
    """
    returns = []
    for i in range(len(rewards)):
        # GRPO: 直接用 reward 作为 returns
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**Q7: 实现 PPO Policy Loss**

```python
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high):
    """
    输入：
    - ppo_kl: [batch_size, seq_len] 策略比率的 log
    - advantages: [batch_size, seq_len] 优势估计
    - eps_clip: float 下界 clipping 参数
    - eps_clip_high: float 上界 clipping 参数
    
    输出：
    - pg_loss: [batch_size, seq_len] 策略梯度损失
    - clipfrac: float clipping 比例
    """
    ratio = (-ppo_kl).exp()
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()
    
    return clip_pg_losses1, clipfrac
```

### 7.4 推理优化题

**Q8: 解释 PagedAttention 的原理**

**考察点**：
- 传统 KV Cache 的问题
- 分页机制的优势
- Block Table 映射

**深挖方向**：
- 内存碎片化问题
- Prefix Caching 原理
- 与 Virtual Memory 的类比

**Q9: Continuous Batching 如何提升吞吐量？**

**考察点**：
- Static Batching 的问题
- 动态调度机制
- GPU 利用率优化

**深挖方向**：
- 请求级别的调度
- Chunked Prefill 的作用
- 与 Tensor Parallelism 的结合

---

## 8. 代码级面试准备

### 8.1 核心数据结构

```python
@dataclass
class Sample:
    """Vime 的核心数据结构"""
    index: int                          # 样本索引
    rollout_id: int                     # rollout 标识符
    prompt: str                         # 输入 prompt
    tokens: list[int]                   # 完整 token 序列
    response: str                       # 生成的文本
    response_length: int                # 响应长度
    reward: float                       # reward 值
    loss_mask: list[int]                # loss 掩码（1=训练，0=不训练）
    rollout_log_probs: list[float]      # rollout 时的 log probs
    status: Status                      # PENDING/COMPLETED/TRUNCATED/ABORTED/FAILED
    metadata: dict                      # 元数据
```

### 8.2 关键函数签名

```python
# KL 散度计算
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor

# Policy Loss 计算
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]

# GAE 计算
def get_advantages_and_returns_batch(
    total_lengths: list[int],
    response_lengths: list[int],
    values_list: list[torch.Tensor],
    rewards_list: list[torch.Tensor],
    gamma: float,
    lambd: float,
    chunked: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]

# Agent 轨迹管理
class TrajectoryManager:
    def record_turn(
        self,
        sid: str,
        *,
        turn: TurnRecord,
        prompt_messages: list[dict[str, Any]],
        response_message: dict[str, Any] | None,
        metadata: dict[str, Any] | None = None,
    ) -> None
    
    def get_trajectory(
        self,
        sid: str,
        *,
        base_sample: Sample,
        reward: float = 0.0,
        extra_metadata: dict[str, Any] | None = None,
    ) -> list[Sample]
```

### 8.3 配置参数速查

```bash
# GRPO 算法参数
--advantage-estimator grpo
--use-kl-loss
--kl-loss-coef 0.00
--kl-loss-type low_var_kl
--entropy-coef 0.00
--eps-clip 0.2
--eps-clip-high 0.28

# 训练参数
--rollout-batch-size 16
--n-samples-per-prompt 8
--num-steps-per-rollout 1
--global-batch-size 128
--num-rollout 3000

# 并行参数
--tensor-model-parallel-size 2
--sequence-parallel
--context-parallel-size 2
--use-dynamic-batch-size
--max-tokens-per-gpu 4608
```

---

## 9. 系统设计面试准备

### 9.1 设计一个 RL Training Framework

**需求分析**：
- 支持多种 RL 算法（PPO/GRPO/REINFORCE++）
- 支持大规模模型（70B+）
- 支持多轮 Agent 训练
- 高效的资源利用

**架构设计**：

```
┌─────────────────────────────────────────────────────────────┐
│                    Control Plane                            │
│  - 参数管理                                                 │
│  - 资源调度                                                 │
│  - 健康监控                                                 │
└─────────────────────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│   Training Engine       │  │   Rollout Engine        │
│  - Megatron/LiteLLM    │  │  - vLLM/SGLang          │
│  - 梯度累积             │  │  - 异步生成             │
│  - 权重更新             │  │  - Reward 计算          │
└─────────────────────────┘  └─────────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    Data Buffer                              │
│  - Prompt 管理                                              │
│  - Sample 存储                                              │
│  - 动态过滤                                                 │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 设计一个 Agent Training System

**核心挑战**：
- 多轮对话的轨迹管理
- Token 漂移处理
- Loss Masking
- Sandbox 集成

**解决方案**：

```python
# 1. TrajectoryManager 管理多轮轨迹
# 2. DriftKind 分类处理 token 漂移
# 3. Loss Masking 区分 model 和 tool tokens
# 4. Sandbox 提供执行环境

# 数据流：
User Query → Model Generate → Tool Execute → Observation → 
Model Generate → ... → Final Answer → Reward Calculation
```

### 9.3 性能优化策略

**训练侧**：
- Dynamic Batch Size: 智能 pack 不同长度的样本
- Gradient Checkpointing: 节省显存
- Mixed Precision Training: FP16/BF16 加速

**推理侧**：
- PagedAttention: 高效 KV Cache 管理
- Continuous Batching: 最大化 GPU 利用率
- Tensor Parallelism: 多卡并行

**系统侧**：
- Colocate 模式: 训练推理共享 GPU
- Delta Weight Sync: 减少同步开销
- Partial Rollout: 提高资源利用率

---

## 10. 实战项目介绍模板

### 10.1 项目背景

"我参与了一个基于 Vime 框架的 LLM Post-Training 项目，目标是通过强化学习提升模型在特定任务上的表现。"

### 10.2 技术方案

"我们采用了 GRPO 算法作为主要的 RL 算法，因为它不需要 Critic 网络，训练效率更高。具体实现中，我们：

1. **数据生成**：使用 vLLM 作为 rollout 后端，每个 prompt 生成 8 个候选响应
2. **Reward 计算**：设计了基于规则的 reward 函数，结合任务特定的评估指标
3. **策略优化**：使用 GRPO 算法，通过组内相对优势估计更新策略
4. **KL 约束**：使用 low_var_kl 估计器，防止策略偏离太远"

### 10.3 关键优化

"在性能优化方面，我们：

1. **Dynamic Batch Size**：智能 pack 不同长度的样本，提高 GPU 利用率
2. **Delta Weight Sync**：只传输变化的权重，减少同步开销
3. **Partial Rollout**：缓存未完成的样本，在下一轮继续生成"

### 10.4 效果与收获

"通过这个项目，我深入理解了：

1. RL 算法的原理和实现（PPO/GRPO/REINFORCE++）
2. 大规模分布式训练的并行策略
3. 推理引擎的优化技术（PagedAttention/Continuous Batching）
4. Agent 训练的关键技术（多轮对话、Loss Masking）"

### 10.5 面试深挖准备

**可能的深挖问题**：

1. "为什么选择 GRPO 而不是 PPO？"
   - GRPO 不需要 Critic 网络，训练效率更高
   - 通过组内采样计算相对优势，效果相当
   - 在大规模模型上，Critic 网络的训练成本很高

2. "如何处理多轮对话的 token 漂移？"
   - 使用 TrajectoryManager 管理多轮轨迹
   - 通过 DriftKind 分类处理（CLEAN/REALIGN/FORK）
   - 使用 Loss Masking 区分 model 和 tool tokens

3. "Delta Weight Sync 是如何工作的？"
   - 通过 CPU 快照检测字节级变化
   - 只传输变化位置 + 值
   - 支持 zstd 压缩，对带宽受限的系统有效

---

## 附录：推荐阅读材料

### 论文

1. **PPO**: "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347)
2. **GRPO**: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning" (https://arxiv.org/abs/2402.03300)
3. **REINFORCE++**: "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models" (https://arxiv.org/pdf/2501.03262)
4. **GSPO**: "GSPO: Group Sequence Policy Optimization" (https://arxiv.org/abs/2507.18071)
5. **CISPO**: "MiniMax-M1 Technical Report" (https://arxiv.org/abs/2506.13585)
6. **PagedAttention**: "Efficient Memory Management for Large Language Model Serving with PagedAttention" (ACM SOSP 2023)

### 代码仓库

1. **Vime**: https://github.com/vllm-project/vime
2. **Slime**: https://github.com/THUDM/slime
3. **vLLM**: https://github.com/vllm-project/vllm
4. **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM

### 技术博客

1. **KL 散度估计**: http://joschu.net/blog/kl-approx.html
2. **Off-Policy RL**: https://fengyao.notion.site/off-policy-rl
3. **vLLM 设计**: https://blog.vllm.ai/

---

> 本文档基于 Vime/Slime/vLLM 项目的深度代码分析，旨在帮助面试者系统性地准备 LLM & Agent 方向的算法实习面试。建议结合代码实践，深入理解每个技术点的原理和实现。
