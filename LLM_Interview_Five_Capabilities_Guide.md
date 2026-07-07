# LLM 算法实习面试：五类能力深度应对指南

> 基于 Vime/Slime/vLLM 项目的工程实践分析，针对技术面试五类核心能力的准备

---

## 目录

1. [能力一：底层原理深入理解](#能力一底层原理深入理解)
2. [能力二：实验和方案验证能力](#能力二实验和方案验证能力)
3. [能力三：问题定位能力](#能力三问题定位能力)
4. [能力四：工程落地能力](#能力四工程落地能力)
5. [能力五：业务与实际场景理解](#能力五业务与实际场景理解)

---

## 能力一：底层原理深入理解

> **考察重点**：不是回答清楚概念，而是讲清楚这个方法解决什么问题，存在哪些局限性，有哪些改进方法

### 1.1 PPO Clipping 机制：为什么这么设计？

**问题**：PPO 中的 clipping 机制有什么作用？为什么需要 Dual-clip PPO？

**回答框架**：

```
问题 → 解决方案 → 局限性 → 改进方向
```

**标准回答**：

**PPO Clipping 解决的问题**：
- Policy Gradient 的步长难以控制，太大会导致策略崩溃
- Importance Sampling 的 ratio 可能偏离 1 太远，导致估计不准
- Clipping 限制 ratio 在 `[1-ε, 1+ε]` 范围内，防止策略更新过大

**Dual-clip PPO 解决的额外问题**：
- 标准 PPO 对正负 advantage 的裁剪是不对称的
- 当 advantage < 0 时，ratio 可以无限增大（只被下界 clip），导致对坏动作的惩罚不够
- Dual-clip 增加上界 `clip_c`，对负 advantage 样本进行更严格的裁剪

**代码证据**（`vime/utils/ppo_utils.py:125-148`）：

```python
# 标准 PPO clip
pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages

# Dual-clip PPO
if eps_clip_c is not None:
    assert eps_clip_c > 1.0  # 防御性编程：确保下界 > 1.0
    pg_losses3 = -eps_clip_c * advantages
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    # 只对负 advantage 样本应用更严格的裁剪
    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
```

**局限性**：
- Clipping 是一种启发式方法，没有理论最优的 ε 值
- 对于不同任务，最优的 clip 参数可能不同
- Clipping 会降低样本效率（一些有效梯度被裁剪掉）

**改进方向**：
- CISPO（MiniMax-M1）：让被裁剪的 token 仍然贡献梯度
- 自适应 clip：根据训练动态调整 clip 范围
- KL-adaptive clipping：结合 KL 散度动态调整

### 1.2 KL 散度估计器：为什么选择 low_var_kl？

**问题**：为什么 low_var_kl 是更好的 KL 估计器？

**回答框架**：

```
三种估计器的对比 → 数学性质 → 实际效果
```

**标准回答**：

**三种 KL 估计器**：
- `k1`: `log_ratio` — 简单但可能为负（不是真正的 KL 散度）
- `k2`: `log_ratio² / 2` — Taylor 近似，总是非负但有偏
- `k3/low_var_kl`: `exp(-log_ratio) - 1 + log_ratio` — 非负、无偏、低方差

**为什么 low_var_kl 更好**：

| 性质 | k1 | k2 | low_var_kl |
|------|----|----|------------|
| 非负 | ❌ | ✅ | ✅ |
| 无偏 | ✅ | ❌ | ✅ |
| 低方差 | ❌ | ✅ | ✅ |

**代码证据**（`vime/utils/ppo_utils.py:12-51`）：

```python
if kl_loss_type in ["k3", "low_var_kl"]:
    # Schulman 博客中的推导：非负、无偏、低方差
    log_ratio = -log_ratio
    kl = log_ratio.exp() - 1 - log_ratio

# 数值稳定性处理：只有 k3 需要 clamp（因为 exp 运算可能爆炸）
if kl_loss_type == "low_var_kl":
    kl = torch.clamp(kl, min=-10, max=10)
```

**局限性**：
- low_var_kl 在 `log_ratio` 很大时仍然可能数值不稳定
- 需要 clamp 但 clamp 会引入偏差
- 不同任务可能需要不同的 KL 估计器

### 1.3 GAE：为什么需要 Advantage Normalization？

**问题**：为什么需要 Advantage Normalization？

**标准回答**：

**问题**：
- Advantage 的尺度在不同任务、不同训练阶段可能差异很大
- 大的 advantage 会导致大的梯度，导致训练不稳定
- 不同 token 的 advantage 方差可能不同

**解决方案**：
- Advantage Normalization（白化）：`(adv - mean) / std`
- 在 DP group 内做分布式统计，确保所有 rank 使用相同的 mean/std

**代码证据**（`vime/backends/megatron_utils/loss.py:772-821`）：

```python
if args.normalize_advantages:
    all_advs = torch.cat(advantages)
    # 在 DP group 内计算分布式统计
    dp_group = mpu.get_data_parallel_group()
    whitened_advs_flat = distributed_masked_whiten(
        all_advs, all_masks, process_group=dp_group, shift_mean=True
    )
```

**局限性**：
- Normalization 会改变 advantage 的相对大小
- 在某些任务上可能不适用（如 reward 本身就是绝对值）

### 1.4 GRPO vs PPO：设计哲学的差异

**问题**：为什么选择 GRPO 而不是 PPO？

**标准回答**：

**核心差异**：
- PPO: 需要 Critic 网络估计 Value，计算 GAE
- GRPO: 通过组内采样（n_samples_per_prompt）计算相对优势，无需 Critic

**为什么 GRPO 更受欢迎**：
- Critic 网络的训练本身也是一个 RL 问题，可能不稳定
- Critic 网络增加了显存和计算开销
- GRPO 通过组内相对排名估计优势，更简单直接

**代码证据**（`vime/utils/ppo_utils.py:244-251`）：

```python
def get_grpo_returns(rewards, kl):
    """GRPO: 直接用 reward 作为 returns（组内相对优势）"""
    returns = []
    for i in range(len(rewards)):
        returns.append(torch.ones_like(kl[i]) * rewards[i])
    return returns
```

**局限性**：
- GRPO 需要多次采样（n_samples_per_prompt），增加了推理开销
- 组内相对排名可能不如 GAE 的时间差分估计准确
- 对于稀疏 reward 任务，GRPO 可能不如 PPO

### 1.5 PagedAttention：与 OS 虚拟内存的类比

**问题**：解释 PagedAttention 的原理

**标准回答**：

**传统问题**：
- KV Cache 需要连续内存分配
- 不同请求的 KV Cache 大小不同，导致内存碎片
- 平均浪费 60-80% 的 GPU 内存

**解决方案**：
- 借鉴 OS 虚拟内存的分页机制
- 将 KV Cache 分成固定大小的块（Block）
- 通过 Block Table 映射到物理 GPU 内存
- 支持非连续内存分配，消除碎片

**代码证据**（vLLM `csrc/attention/attention_kernels.cu`）：

```cuda
// 通过块表间接寻址加载 Key
const int physical_block_number = block_table[block_idx];
const scalar_t* k_ptr = k_cache + physical_block_number * stride;
```

**局限性**：
- 块的大小是固定的，可能有内部碎片
- Block Table 引入了额外的间接寻址开销
- 需要管理空闲块的分配和回收

---

## 能力二：实验和方案验证能力

> **考察重点**：不仅关注你做了什么，更关注怎么证明它是有效的，追问实验细节

### 2.1 如何验证 KL 约束的有效性？

**问题**：你是如何验证 KL 约束的有效性的？

**回答框架**：

```
实验设计 → 指标选择 → 结果分析 → 结论
```

**标准回答**：

**实验设计**：
- 对比实验：有 KL 约束 vs 无 KL 约束
- 超参数搜索：不同的 kl_coef 值（0.001, 0.01, 0.1, 0.5）
- 监控指标：reward、KL 散度、生成多样性

**关键指标**：
```python
# reported_loss 中的指标
"ppo_kl": sum_of_sample_mean(ppo_kl),  # KL 散度
"kl_loss": kl_loss,                      # KL 损失（如果启用）
"opd_reverse_kl": opd_reverse_kl,        # OPD 的 reverse KL
```

**预期结果**：
- 无 KL 约束：reward 可能更高，但生成质量下降（reward hacking）
- 有 KL 约束：reward 略低，但生成更稳定、更多样
- 最优 kl_coef：在 reward 和 KL 之间找到平衡点

### 2.2 如何验证 Dynamic Sampling 的有效性？

**问题**：你是如何验证 Dynamic Sampling 的有效性的？

**标准回答**：

**实验设计**：
- 对比实验：Dynamic Sampling vs Static Sampling
- 过滤条件：`check_reward_nonzero_std`（确保组内 reward 有差异）
- 监控指标：数据利用率、训练效率、最终性能

**代码证据**（`vime/rollout/filter_hub/dynamic_sampling_filters.py`）：

```python
def check_reward_nonzero_std(args, samples, **kwargs):
    rewards = [sample.get_reward_value(args) for sample in samples]
    keep = torch.tensor(rewards, dtype=torch.float).std() > 0.0
    return DynamicFilterOutput(
        keep=keep,
        reason=None if keep else f"zero_std_{round(rewards[0], 1)}",
    )
```

**预期结果**：
- Dynamic Sampling 过滤掉 reward 无差异的样本
- 提高数据质量，避免同质化数据
- 可能降低数据利用率，但提高训练效率

### 2.3 如何验证 Chunked GAE 的正确性？

**问题**：你是如何验证 Chunked GAE 的正确性的？

**标准回答**：

**验证方法**：
- 数值对比：Chunked GAE vs Vanilla GAE
- 相对误差：`abs(chunked - vanilla) / abs(vanilla) < 1e-5`
- 边界条件：空序列、单 token 序列、超长序列

**代码证据**（`vime/utils/ppo_utils.py:549-689`）：

```python
def chunked_gae(rewards, values, gamma, lambd, chunk_size=128):
    """
    FlashLinearAttention 启发的并行前缀扫描 GAE
    
    数学等价性证明：
    - 块内：S_local = Δ @ M，其中 M[i,j] = w^(j-i)
    - 块间：S_global = S_local + s_prev * pow_vec
    - 这与 vanilla GAE 的反向递推完全等价
    """
```

**边界条件处理**：
- `w == 0` 时特殊处理，避免 `0^0` 的数值问题
- `T` 不是 `chunk_size` 倍数时做 padding

### 2.4 如何验证 Delta Weight Sync 的正确性？

**问题**：你是如何验证 Delta Weight Sync 的正确性的？

**标准回答**：

**验证方法**：
- 一致性检查：Delta Sync 后的权重与 Full Sync 完全一致
- 性能测试：同步时间、带宽利用率
- 压缩比：zstd 压缩前后的大小对比

**代码证据**（`vime/backends/megatron_utils/update_weight/update_weight_from_distributed_delta.py`）：

```python
# 通过 CPU 快照检测字节级变化
cpu_snapshot = {name: param.cpu().clone() for name, param in named_params.items()}
# ... 训练后 ...
# 检测变化
for name, param in named_params.items():
    cpu_param = cpu_snapshot[name]
    diff_mask = param.cpu() != cpu_param  # 字节级比较
```

---

## 能力三：问题定位能力

> **考察重点**：模型上线后能力突然下降，系统上线后突然十分缓慢，实验结果和预期不一致，这些问题是怎么排查的

### 3.1 训练 Loss 突然飙升

**问题**：训练过程中 loss 突然飙升，你怎么排查？

**回答框架**：

```
现象描述 → 可能原因 → 排查步骤 → 解决方案
```

**标准回答**：

**可能原因**：
1. 学习率过大，导致梯度爆炸
2. KL 散度过大，策略偏离太远
3. Advantage 异常（全正或全负）
4. 数据质量问题（异常样本）

**排查步骤**：
```python
# 1. 检查 KL 散度
if reported_loss["ppo_kl"] > 10:
    print("KL 散度过大，可能需要增大 kl_coef 或减小学习率")

# 2. 检查 Advantage 分布
all_advs = torch.cat(advantages)
print(f"Advantage: mean={all_advs.mean()}, std={all_advs.std()}")
if all_advs.std() < 0.01:
    print("Advantage 方差过小，模型可能陷入局部最优")

# 3. 检查 Reward 分布
rewards = [sample.reward for sample in batch]
print(f"Reward: mean={np.mean(rewards)}, std={np.std(rewards)}")
if np.std(rewards) == 0:
    print("Reward 无差异，Dynamic Sampling 可能过滤了太多样本")
```

**解决方案**：
- 减小学习率
- 增大 kl_coef
- 检查数据质量
- 使用梯度裁剪

### 3.2 Rollout 阶段吞吐量下降

**问题**：Rollout 阶段吞吐量突然下降，你怎么排查？

**标准回答**：

**可能原因**：
1. vLLM 引擎内存不足
2. 请求队列积压
3. 网络通信瓶颈
4. 负载不均衡

**排查步骤**：
```python
# 1. 检查 vLLM 引擎状态
# 查看 GPU 内存使用
nvidia-smi

# 2. 检查请求队列
# 查看 pending 请求数
print(f"Pending requests: {len(state.pending_tasks)}")

# 3. 检查 DP rank 负载
print(f"DP counts: {self.dp_counts}")
# 如果某个 rank 的 count 远高于其他，说明负载不均

# 4. 检查网络通信
# 查看 NCCL 通信时间
```

**解决方案**：
- 调整 `--vllm-gpu-memory-utilization`
- 优化 DP 负载均衡
- 增加并发数（`--vllm-server-concurrency`）
- 使用 Delta Weight Sync 减少同步开销

### 3.3 Agent 训练 Reward 异常

**问题**：Agent 训练中 reward 突然下降，你怎么排查？

**标准回答**：

**可能原因**：
1. Token 漂移导致训练数据错误
2. Loss Masking 策略有误
3. Sandbox 执行环境异常
4. Reward 函数有 bug

**排查步骤**：
```python
# 1. 检查 Token 漂移
# 查看 DriftKind 分布
drift_stats = {"CLEAN": 0, "REALIGN": 0, "FORK": 0}
for turn in turns:
    drift = builder.classify_token_drift(turn)
    drift_stats[drift.kind] += 1
print(f"Drift stats: {drift_stats}")
# 如果 FORK 比例过高，说明漂移问题严重

# 2. 检查 Loss Mask
# 查看 model tokens vs tool tokens 的比例
model_tokens = sum(loss_mask)
tool_tokens = len(loss_mask) - model_tokens
print(f"Model tokens: {model_tokens}, Tool tokens: {tool_tokens}")

# 3. 检查 Sandbox 执行
# 查看执行成功率
success_rate = success_count / total_count
print(f"Sandbox success rate: {success_rate}")
```

### 3.4 权重同步失败

**问题**：权重同步失败，你怎么排查？

**标准回答**：

**可能原因**：
1. NCCL 通信超时
2. GPU 内存不足
3. 进程组配置错误
4. 分布式锁死锁

**排查步骤**：
```python
# 1. 检查 NCCL 通信
# 查看 NCCL 日志
export NCCL_DEBUG=INFO

# 2. 检查 GPU 内存
nvidia-smi

# 3. 检查进程组
print(f"TP group size: {mpu.get_tensor_model_parallel_world_size()}")
print(f"DP group size: {mpu.get_data_parallel_world_size()}")

# 4. 检查锁状态
# 查看 rollout_engine_lock 是否被持有
```

**代码证据**（`vime/backends/megatron_utils/update_weight/update_weight_from_distributed.py`）：

```python
# 防死锁的锁设计
while not ray.get(self.rollout_engine_lock.acquire.remote()):
    time.sleep(0.1)
try:
    # ... broadcast ...
finally:
    # 确保锁被释放
    ray.get(self.rollout_engine_lock.release.remote())
```

---

## 能力四：工程落地能力

> **考察重点**：不仅看理论，更看实际动手与工程落地能力，理论结合实际

### 4.1 如何处理 Context Parallelism 下的死锁问题？

**问题**：CP 场景下如何防止训练死锁？

**标准回答**：

**问题描述**：
- 在 allgather-CP 模式下，某些 CP rank 可能没有任何参与 loss 计算的 token（全是 padding）
- 如果不加处理，梯度不会流过这些 rank 的 attention 路径
- 导致 CP gather 的 backward (reduce-scatter) 不会被调用
- 其他等待该通信的 CP rank 会被死锁

**解决方案**：

```python
# vime/backends/megatron_utils/loss.py:1279-1282
# 强制梯度流过所有 CP rank
if args.allgather_cp and mpu.get_context_parallel_world_size() > 1:
    loss = loss + 0 * logits.sum()  # 不改变 loss 值，但保留计算图
```

**原理**：
- `0 * logits.sum()` 不改变 loss 的值
- 但强制 autograd 保留对 logits 的引用
- 使得反向传播可以正常触发梯度通信

### 4.2 如何处理多轮对话的 Token 漂移？

**问题**：多轮对话中如何处理 Token 漂移？

**标准回答**：

**问题描述**：
- TITO (Token-In-Token-Out) round-trip 会导致 token ID 变化
- Chat-template 重新渲染也会改变 token 序列
- 如果不处理，多轮对话的训练数据会出错

**解决方案**：TrajectoryManager 的三分类机制

```python
# vime/agent/trajectory.py
class DriftKind(enum.Enum):
    CLEAN = "clean"    # 无漂移：直接追加
    REALIGN = "realign" # 短漂移：覆盖并标记 loss_mask=0
    FORK = "fork"       # 大漂移：关闭当前 builder，开新的 fork

def classify_token_drift(self, turn):
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

**工程权衡**：
- `fork_threshold` 太小：频繁 fork，产生短序列，降低训练效率
- `fork_threshold` 太大：把漂移吸收到 loss_mask=0 的 token 中，浪费计算

### 4.3 如何设计容错机制？

**问题**：如何设计系统的容错机制？

**标准回答**：

**Vime 的三层容错设计**：

```python
# 1. HealthMonitor — 健康检查
def _check_engine_health(self):
    try:
        # 检查 vLLM 引擎是否存活
        await ray.get(engine.check_health.remote(), timeout=5)
    except Exception as e:
        logger.warning(f"Engine {i} unhealthy: {e}")
        self._kill_engine(g_idx, i)

# 2. Abort 机制 — 幂等性
async def _abort_one(url: str) -> None:
    try:
        await post(f"{url.rstrip('/')}/abort_requests", {}, max_retries=3)
    except Exception as e:
        logger.warning(f"Failed to abort: {e}")  # 失败只 log，不 raise

# 3. Recover — 恢复机制
def recover(self):
    # 1. 记录死亡的 engine indices
    dead_per_group = [[i for i, engine in enumerate(g.all_engines) if engine is None] 
                      for g in self.server_groups]
    # 2. 并发重启所有组
    for g in self.server_groups:
        handles, port_cursors = g.start_engines(port_cursors)
    # 3. 对新 engine 做权重同步
```

### 4.4 如何处理 MoE 模型的权重同步？

**问题**：MoE 模型的权重同步有什么特殊挑战？

**标准回答**：

**挑战**：
- MoE 模型有专家参数和非专家参数
- 专家参数需要 Expert Parallelism (EP)
- 需要分别处理两种参数的同步

**解决方案**：

```python
# vime/backends/megatron_utils/update_weight/common.py
for name, param in named_params.items():
    if ".experts." in name:
        # 专家参数：TP all-gather → buffer → EP all-gather → HF 转换 → broadcast
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        # 非专家参数：TP all-gather → HF 转换 → broadcast
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()
```

---

## 能力五：业务与实际场景理解

> **考察重点**：项目真正需要产生的是场景价值和业务价值，面试官会问方案适合什么场景，上线成本有多高

### 5.1 这个方案适合什么场景？

**问题**：GRPO 方案适合什么场景？

**标准回答**：

**适合的场景**：
- 数学推理：reward 可以精确计算（答案正确/错误）
- 代码生成：reward 可以通过测试用例验证
- 格式遵循：reward 可以通过规则检查

**不适合的场景**：
- 开放式对话：reward 难以定义
- 创意写作：reward 主观性强
- 多轮交互：reward 信号稀疏

**成本分析**：

| 资源 | 成本 | 优化方向 |
|------|------|---------|
| GPU 计算 | 高 | 使用 Dynamic Batch Size、Delta Weight Sync |
| 推理开销 | 高 | 使用 GRPO（无需 Critic） |
| 存储 | 中 | 使用 Partial Rollout、Delta Checkpoint |
| 网络通信 | 中 | 使用 Colocate 模式、Delta Weight Sync |

### 5.2 上线成本有多高？

**问题**：这个方案的上线成本有多高？

**标准回答**：

**硬件成本**：
- 最小配置：8 x H100 GPU（训练 4 + 推理 4）
- 推荐配置：16 x H100 GPU（训练 8 + 推理 8）
- 大规模配置：32+ x H100 GPU

**软件成本**：
- 依赖：Megatron-LM、vLLM、Ray
- 部署：Docker 镜像、Kubernetes 集群
- 监控：Prometheus、Grafana

**时间成本**：
- 环境搭建：1-2 天
- 数据准备：1-3 天
- 训练调参：3-7 天
- 评估测试：1-2 天

### 5.3 如果资源有限，首先优化哪些部分？

**问题**：如果资源有限，首先优化哪些部分？

**标准回答**：

**优先级排序**：

1. **数据质量**（最高优先级）
   - 使用 Dynamic Sampling 过滤低质量数据
   - 优化 Reward 函数，确保信号清晰
   - 数据清洗和去重

2. **训练效率**
   - 使用 Dynamic Batch Size
   - 使用 Gradient Checkpointing
   - 使用 Mixed Precision Training

3. **推理效率**
   - 使用 Continuous Batching
   - 使用 Tensor Parallelism
   - 优化 vLLM 配置

4. **系统稳定性**（最后考虑）
   - 容错机制
   - 监控告警
   - 日志收集

### 5.4 实际生产中的监控指标

**问题**：实际生产中需要监控哪些指标？

**标准回答**：

**训练指标**：
```python
# 每个训练 step 记录
"loss": loss,
"pg_loss": pg_loss,
"entropy_loss": entropy_loss,
"pg_clipfrac": pg_clipfrac,
"ppo_kl": ppo_kl,
```

**推理指标**：
- 吞吐量：tokens/sec
- 延迟：TTFT (Time To First Token), TPS (Tokens Per Second)
- GPU 利用率
- 内存使用

**系统指标**：
- 权重同步时间
- Rollout 耗时
- 数据 Buffer 大小
- 引擎健康状态

### 5.5 如何保证数据回滚？

**问题**：如果训练出问题，如何保证数据回滚？

**标准回答**：

**Checkpoint 机制**：
```python
# vime 支持定期保存 checkpoint
--save-interval 20  # 每 20 步保存一次

# 恢复训练
--load /path/to/checkpoint  # 从 checkpoint 恢复
```

**数据版本管理**：
- 每个 rollout 有唯一的 rollout_id
- 数据 Buffer 支持按 rollout_id 过滤
- Partial Rollout 样本记录首次生成的 rollout_id

**回滚策略**：
1. 停止当前训练
2. 加载最近的 checkpoint
3. 调整超参数
4. 重新开始训练

---

## 附录：面试问题速查表

### 底层原理类

| 问题 | 关键点 | 代码位置 |
|------|--------|---------|
| PPO Clipping 的作用 | 防止策略更新过大 | `ppo_utils.py:125-148` |
| 为什么用 low_var_kl | 非负、无偏、低方差 | `ppo_utils.py:34-49` |
| GRPO vs PPO | 无需 Critic，更简单 | `ppo_utils.py:244-251` |
| Chunked GAE 的优势 | 并行化，O(T/chunk_size) | `ppo_utils.py:549-689` |
| PagedAttention 原理 | 分页管理 KV Cache | vLLM `csrc/attention/` |

### 实验验证类

| 问题 | 关键点 | 代码位置 |
|------|--------|---------|
| 如何验证 KL 约束 | 对比实验，监控 KL 和 reward | `loss.py:696-709` |
| 如何验证 Dynamic Sampling | 过滤低质量数据，提高效率 | `filter_hub/` |
| 如何验证 Chunked GAE | 数值对比，边界条件 | `ppo_utils.py:549-689` |

### 问题定位类

| 问题 | 关键点 | 代码位置 |
|------|--------|---------|
| Loss 突然飙升 | 检查 KL、Advantage、Reward | `loss.py:1074-1098` |
| Rollout 吞吐量下降 | 检查内存、队列、负载 | `vllm_rollout.py` |
| Agent Reward 异常 | 检查漂移、Loss Mask、Sandbox | `trajectory.py` |

### 工程落地类

| 问题 | 关键点 | 代码位置 |
|------|--------|---------|
| CP 死锁防护 | 零 loss 注入 | `loss.py:1279-1282` |
| Token 漂移处理 | DriftKind 三分类 | `trajectory.py:129-190` |
| 容错机制 | HealthMonitor + Abort + Recover | `health_monitor.py` |
| MoE 权重同步 | 专家参数特殊处理 | `update_weight/common.py` |

### 业务场景类

| 问题 | 关键点 | 考虑因素 |
|------|--------|---------|
| 方案适合什么场景 | 数学、代码、格式遵循 | Reward 可定义性 |
| 上线成本 | 硬件、软件、时间 | 最小配置 8xH100 |
| 资源有限时的优化 | 数据质量 > 训练效率 > 推理效率 | ROI 排序 |
| 监控指标 | 训练、推理、系统指标 | 全方位可观测 |

---

> 本文档基于 Vime/Slime/vLLM 项目的工程实践分析，针对技术面试五类核心能力的准备。建议结合代码实践，深入理解每个问题的排查思路和解决方案。
