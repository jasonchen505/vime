# 8卡 RTX 4090 复现 Vime 框架完整 Plan

> 目标：用 8 卡 RTX 4090 (24GB/卡) 完整全流程复现 Vime 框架的 LLM Post-Training RL 训练

---

## 一、硬件资源评估

### 1.1 RTX 4090 规格

| 参数 | RTX 4090 | 对比 H100 |
|------|----------|-----------|
| 显存 | 24 GB GDDR6X | 80 GB HBM3 |
| FP16 算力 | 82.6 TFLOPS | 989 TFLOPS |
| NVLink | ❌ 不支持 | ✅ 900 GB/s |
| 互联带宽 | PCIe 4.0 x16 (32 GB/s) | NVLink (900 GB/s) |

### 1.2 显存预算分析

```
Qwen2.5-0.5B 模型显存估算：

模型参数 (FP16):
- 参数量: 0.5B = 500M 参数
- 显存: 500M × 2 bytes = 1 GB

优化器状态 (Adam):
- 显存: 500M × 8 bytes = 4 GB

梯度:
- 显存: 500M × 2 bytes = 1 GB

KV Cache (vLLM):
- 取决于 max_response_len 和 batch_size

总计 (训练): ~6 GB + KV Cache
```

### 1.3 与官方配置的差异

| 配置项 | 官方 (H100) | 4090 调整 |
|--------|-------------|-----------|
| 模型大小 | Qwen3-4B | Qwen2.5-0.5B |
| TP Size | 2 | 1 |
| CP Size | 2 | 1 |
| PP Size | 1 | 1 |
| max_tokens_per_gpu | 4608 | 2048 |
| rollout_num_gpus_per_engine | 2 | 1 |
| Colocate | 可选 | 必须 |

---

## 二、环境准备阶段

### Phase 1: 基础环境搭建 (Day 1)

**任务清单**：
- [ ] 1.1 安装 NVIDIA 驱动 (≥ 535.129.03)
- [ ] 1.2 安装 CUDA Toolkit (≥ 12.1)
- [ ] 1.3 安装 Docker + NVIDIA Container Toolkit
- [ ] 1.4 拉取 Vime Docker 镜像
- [ ] 1.5 验证 GPU 可用性

**执行步骤**：

```bash
# 1. 检查驱动
nvidia-smi

# 2. 拉取镜像
docker pull vllm/vime:latest

# 3. 启动容器
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /data/home/yizhou:/root/workspace \
  -it vllm/vime:latest /bin/bash

# 4. 验证 GPU
python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
```

### Phase 2: 代码和数据准备 (Day 1-2)

**任务清单**：
- [ ] 2.1 克隆 Vime 仓库
- [ ] 2.2 下载 Qwen2.5-0.5B 模型
- [ ] 2.3 下载 GSM8K 训练数据
- [ ] 2.4 下载评估数据

**执行步骤**：

```bash
# 1. 克隆仓库
cd /root/workspace
git clone https://github.com/vllm-project/vime.git
cd vime
pip install -e . --no-deps

# 2. 下载模型 (约 1GB)
pip install huggingface_hub
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir /root/models/Qwen2.5-0.5B-Instruct

# 3. 下载数据集
# GSM8K 数据集
python -c "
from datasets import load_dataset
import json

# 下载 GSM8K
ds = load_dataset('openai/gsm8k', 'main', cache_dir='/root/data')

# 保存为 parquet
ds['train'].to_parquet('/root/data/gsm8k/train.parquet')
ds['test'].to_parquet('/root/data/gsm8k/test.parquet')
print('GSM8K downloaded successfully')
"
```

---

## 三、模型权重转换阶段

### Phase 3: HF → Megatron 格式转换 (Day 2)

**任务清单**：
- [ ] 3.1 安装 Megatron-LM
- [ ] 3.2 运行权重转换脚本
- [ ] 3.3 验证转换结果

**执行步骤**：

```bash
# 1. 克隆 Megatron-LM
cd /root
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
git checkout v4.6.0

# 2. 设置环境变量
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

# 3. 运行转换脚本
cd /root/workspace/vime

# 加载模型配置
source scripts/models/qwen2.5-0.5B.sh

# 转换权重
python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct \
    --save /root/models/Qwen2.5-0.5B-Instruct_torch_dist

# 4. 验证转换结果
ls -la /root/models/Qwen2.5-0.5B-Instruct_torch_dist/
```

**预期输出**：
```
/root/models/Qwen2.5-0.5B-Instruct_torch_dist/
├── iter_0000001/
│   ├── mp_rank_00/
│   │   └── model_optim_rng.pt
│   └── latest_checkpointed_iteration.txt
```

---

## 四、训练配置阶段

### Phase 4: 创建 4090 适配的训练脚本 (Day 2-3)

**任务清单**：
- [ ] 4.1 创建适配 4090 的训练脚本
- [ ] 4.2 配置参数
- [ ] 4.3 测试启动

**创建文件 `/root/workspace/vime/scripts/run-qwen2.5-0.5B-4090.sh`**：

```bash
#!/bin/bash

# 4090 适配版本
# 目标: 8卡 RTX 4090, 单卡 24GB

set -ex

export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
source "${SCRIPT_DIR}/models/qwen2.5-0.5B.sh"

CKPT_ARGS=(
   --hf-checkpoint /root/models/Qwen2.5-0.5B-Instruct
   --ref-load /root/models/Qwen2.5-0.5B-Instruct_torch_dist
)

ROLLOUT_ARGS=(
   --prompt-data /root/data/gsm8k/train.parquet
   --input-key messages
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type math
   --num-rollout 100
   --rollout-batch-size 8
   --n-samples-per-prompt 4
   --rollout-max-response-len 512
   --rollout-temperature 1
   --global-batch-size 32
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data gsm8k /root/data/gsm8k/test.parquet
   --n-samples-per-eval-prompt 1
   --eval-max-response-len 512
   --eval-top-k 1
)

PERF_ARGS=(
   --tensor-model-parallel-size 1
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 2048
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --kl-coef 0.00
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

VLLM_ARGS=(
   --rollout-num-gpus-per-engine 1
   --vllm-gpu-memory-utilization 0.6
   --vllm-attention-backend flashinfer
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

# 启动 Ray 集群
ray start --head --node-ip-address 127.0.0.1 --num-gpus 8 --disable-usage-stats
sleep 3

# 提交训练任务
ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json='{
     "env_vars": {
        "PYTHONPATH": "/root/Megatron-LM",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_ALGO": "Ring",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8"
     }
   }' \
   -- python3 train.py \
   --actor-num-nodes 1 \
   --actor-num-gpus-per-node 8 \
   --colocate \
   --calculate-per-token-loss \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${VLLM_ARGS[@]} \
   ${MISC_ARGS[@]}
```

**关键调整说明**：

| 参数 | 原值 | 4090 值 | 原因 |
|------|------|---------|------|
| `rollout-batch-size` | 16 | 8 | 减少显存占用 |
| `n-samples-per-prompt` | 8 | 4 | 减少采样数 |
| `rollout-max-response-len` | 8192 | 512 | 减少 KV Cache |
| `max-tokens-per-gpu` | 4608 | 2048 | 适配显存 |
| `vllm-gpu-memory-utilization` | 0.8 | 0.6 | 留更多显存给训练 |
| `rollout-num-gpus-per-engine` | 2 | 1 | TP=1 |

---

## 五、训练执行阶段

### Phase 5: 小规模测试 (Day 3)

**任务清单**：
- [ ] 5.1 运行 10 个 rollout 的测试训练
- [ ] 5.2 监控 GPU 显存使用
- [ ] 5.3 检查训练日志
- [ ] 5.4 验证 reward 变化

**执行步骤**：

```bash
# 1. 进入容器
docker exec -it <container_id> bash

# 2. 修改 num-rollout 为 10 进行测试
cd /root/workspace/vime
vim scripts/run-qwen2.5-0.5B-4090.sh
# 修改: --num-rollout 10

# 3. 运行测试
bash scripts/run-qwen2.5-0.5B-4090.sh

# 4. 监控 GPU
watch -n 1 nvidia-smi

# 5. 查看日志
# Ray Dashboard: http://localhost:8265
```

**预期结果**：
- GPU 显存使用 < 20 GB
- 训练 loss 逐步下降
- Reward 逐步提升

### Phase 6: 完整训练 (Day 3-5)

**任务清单**：
- [ ] 6.1 运行 100 个 rollout 的完整训练
- [ ] 6.2 监控训练进度
- [ ] 6.3 定期评估模型
- [ ] 6.4 保存 checkpoint

**执行步骤**：

```bash
# 1. 恢复 num-rollout 为 100
vim scripts/run-qwen2.5-0.5B-4090.sh
# 修改: --num-rollout 100

# 2. 运行完整训练
bash scripts/run-qwen2.5-0.5B-4090.sh

# 3. 监控 Wandb (如果启用)
# https://wandb.ai/glm-zero/vime-dev
```

**预期时间**：
- 每个 rollout 约 5-10 分钟
- 100 个 rollout 约 8-17 小时

---

## 六、评估与导出阶段

### Phase 7: 模型评估 (Day 5-6)

**任务清单**：
- [ ] 7.1 在 GSM8K 测试集上评估
- [ ] 7.2 对比训练前后的准确率
- [ ] 7.3 分析训练曲线

**执行步骤**：

```bash
# 1. 转换 Megatron 权重回 HF 格式
cd /root/workspace/vime
export PYTHONPATH=/root/Megatron-LM:$PYTHONPATH

python tools/convert_torch_dist_to_hf.py \
  --input-dir /root/models/Qwen2.5-0.5B-Instruct_torch_dist/iter_000100/ \
  --output-dir /root/models/Qwen2.5-0.5B-rl-trained \
  --origin-hf-dir /root/models/Qwen2.5-0.5B-Instruct

# 2. 运行评估脚本
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

# 加载训练后的模型
model = AutoModelForCausalLM.from_pretrained('/root/models/Qwen2.5-0.5B-rl-trained')
tokenizer = AutoTokenizer.from_pretrained('/root/models/Qwen2.5-0.5B-rl-trained')

# 加载测试数据
ds = load_dataset('openai/gsm8k', 'main', split='test')

# 简单评估
correct = 0
total = 0
for sample in ds.select(range(100)):  # 评估 100 个样本
    messages = sample['messages']
    inputs = tokenizer.apply_chat_template(messages, return_tensors='pt')
    outputs = model.generate(inputs, max_new_tokens=256)
    response = tokenizer.decode(outputs[0][inputs.shape[1]:])
    
    # 检查答案
    if sample['answer'][-3:] in response[-50:]:
        correct += 1
    total += 1

print(f'Accuracy: {correct/total*100:.2f}%')
"
```

### Phase 8: 文档总结 (Day 6-7)

**任务清单**：
- [ ] 8.1 总结训练过程
- [ ] 8.2 记录遇到的问题和解决方案
- [ ] 8.3 整理学习笔记
- [ ] 8.4 更新增量学习文档

---

## 七、潜在问题与解决方案

### 7.1 显存不足 (OOM)

**症状**：`CUDA out of memory`

**解决方案**：
```bash
# 1. 减小 max_tokens_per-gpu
--max-tokens-per-gpu 1024  # 从 2048 减小

# 2. 减小 rollout-batch-size
--rollout-batch-size 4  # 从 8 减小

# 3. 减小 rollout-max-response-len
--rollout-max-response-len 256  # 从 512 减小

# 4. 启用 gradient checkpointing
--recompute-granularity full
--recompute-method uniform
--recompute-num-layers 2
```

### 7.2 NCCL 通信错误

**症状**：`NCCL error`

**解决方案**：
```bash
# 1. 设置 NCCL 环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 设置正确的网络接口

# 2. 检查 Ray 集群状态
ray status
ray list actors
```

### 7.3 权重同步失败

**症状**：权重同步超时或失败

**解决方案**：
```bash
# 1. 增加超时时间
--weight-sync-timeout 600  # 默认 300 秒

# 2. 使用 Disk 模式替代 NCCL
--update-weight-transport disk
```

### 7.4 训练不收敛

**症状**：loss 不下降或 reward 不提升

**解决方案**：
```bash
# 1. 检查 learning rate
--lr 5e-7  # 从 1e-6 减小

# 2. 增加 KL 约束
--use-kl-loss
--kl-loss-coef 0.01

# 3. 调整 clip 参数
--eps-clip 0.1  # 更严格的 clipping
```

---

## 八、时间线总结

| 阶段 | 任务 | 时间 | 产出 |
|------|------|------|------|
| Phase 1 | 环境搭建 | Day 1 | Docker 容器运行 |
| Phase 2 | 数据下载 | Day 1-2 | 模型和数据就绪 |
| Phase 3 | 权重转换 | Day 2 | Megatron 格式权重 |
| Phase 4 | 脚本编写 | Day 2-3 | 4090 适配脚本 |
| Phase 5 | 小规模测试 | Day 3 | 测试通过 |
| Phase 6 | 完整训练 | Day 3-5 | 训练完成 |
| Phase 7 | 模型评估 | Day 5-6 | 评估报告 |
| Phase 8 | 文档总结 | Day 6-7 | 学习笔记 |

**总计**：7 天完成全流程复现

---

## 九、成功标准

### 9.1 技术指标

- [ ] 训练 loss 下降 > 30%
- [ ] GSM8K 准确率提升 > 5%
- [ ] 无 OOM 错误
- [ ] 无 NCCL 通信错误

### 9.2 学习指标

- [ ] 理解 GRPO 算法原理
- [ ] 理解 Vime 架构设计
- [ ] 掌握训练参数调优
- [ ] 能够独立排查问题

### 9.3 产出指标

- [ ] 完整的训练脚本
- [ ] 训练日志和曲线
- [ ] 评估结果报告
- [ ] 学习笔记文档

---

## 附录 A：常用调试命令

```bash
# 查看 GPU 使用
nvidia-smi
watch -n 1 nvidia-smi

# 查看 Ray 状态
ray status
ray list actors
ray dashboard

# 查看训练日志
tail -f /tmp/ray/session_latest/logs/worker-*.log

# 杀死进程
pkill -9 -f 'vllm serve|VLLM'
ray stop --force
```

## 附录 B：参考资源

- [Vime 官方文档](https://docs.vllm.ai/projects/vime/en/latest/)
- [Vime GitHub](https://github.com/vllm-project/vime)
- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [vLLM 文档](https://docs.vllm.ai/)
