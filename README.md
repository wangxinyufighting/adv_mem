# AdvMem Training

AdvMem 通过交替 GRPO 训练 Attacker 和 Memory Builder。Attacker 寻找当前记忆
`M_t` 缺失的能力，Memory Builder 通过编辑记忆修复这些能力。

## 训练流程

```text
Full Memory Graph
  → Graph Router 采样 Route
  → 构建 Attacker Parquet
  → Verl GRPO 训练 Attacker
  → Attacker 生成 Question
  → Oracle 验证并生成标准答案和 Evidence
  → Answer Agent 分别使用 Golden Corpus 和 M_t 回答
  → 回答正确：写入 Success Pool
  → 回答错误：构建 Memory Builder Parquet
  → Verl GRPO 训练 Memory Builder
  → Memory Builder 生成 ADD / MERGE / DELETE / NOOP
  → Reward > 0：提交 M_temp 为 M_t+1
  → Reward <= 0：保留 M_t，写入 High-Priority Buffer
  → 连续多轮无有效攻击且无无损压缩：停止
```

Attacker 和 Memory Builder 按轮次交替训练，不同时更新。Retrieval 在训练和
测试中始终使用同一套实现。

## 环境

训练需要 Linux x86_64、CUDA GPU、525 或更新的 NVIDIA Driver，以及
Python 3.10-3.12。训练环境固定为：

- CUDA 12.4
- PyTorch 2.6.0+cu124
- vLLM 0.8.5.post1
- FlashAttention 2.7.4.post1
- Verl 0.4.1.dev

首次执行 `train.sh` 会自动创建 `.venv-cu124`，不使用 `uv`。也可先手动安装：

```bash
bash scripts/setup_cuda124.sh
```

检查环境：

```bash
.venv-cu124/bin/python -c \
  'import torch, vllm; print(torch.__version__, torch.version.cuda, vllm.__version__)'
```

应输出 `2.6.0+cu124 12.4 0.8.5.post1`。旧的 `verl/.venv` 不再使用。

## 服务配置

训练前需要以下服务：

- DeepSeek：Oracle、Reward Judge 和 Retrieval Query Parser。
- Qwen3-0.6B：固定的 Answer Agent。
- `text-embedding-v4`：Memory Retrieval embedding。
- BGE Reranker：提供 `/v1/rerank` 接口。

配置环境变量：

```bash
export DEEPSEEK_API_KEY=xxxx
export DEEPSEEK_API_BASE=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat

export MOS_EMBEDDER_API_KEY=xxxx
export MOS_EMBEDDER_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export MOS_EMBEDDER_MODEL=text-embedding-v4
export EMBEDDING_DIMENSION=1024

export BGE_RERANKER_URL=http://localhost:8000/v1/rerank
export BGE_RERANKER_MODEL=bge-reranker-v2-m3

export ANSWER_AGENT_API_BASE=http://localhost:8001/v1
export ANSWER_AGENT_MODEL=Qwen/Qwen3-0.6B
```

启动固定 Answer Agent：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_answer_agent.sh
```

BGE Reranker 可以运行在另一块 GPU 或远程服务上。确保
`BGE_RERANKER_URL` 可访问即可。

## 启动训练

先生成配置文件并填入 API Key：

```bash
cp .env.example .env
```

默认配置使用 GPU 0 训练、GPU 1 运行 Answer Agent、GPU 2 运行
BGE Reranker。一键启动：

```bash
bash train.sh --rounds 20
```

`train.sh` 会启动 Answer Agent 和 BGE Reranker、等待服务就绪、执行交替
训练，并在结束时关闭由它启动的服务。

使用远程 Answer Agent 或 Reranker 时，在 `.env` 中设置对应 URL，并将
`START_ANSWER_AGENT` 或 `START_RERANKER` 设为 `0`。

默认使用：

- Full Graph：`data/longmemeval/memory_graph_fullgraph5.json`
- Backbone：`Qwen/Qwen3-0.6B`
- 每个 case 采样 16 条 Route
- 每轮为每个 case 处理 8 个 Question
- Attacker 和 Memory Builder 各训练 1 epoch

常用参数：

```bash
bash train.sh \
  --rounds 5 \
  --routes-per-case 16 \
  --candidates-per-case 8 \
  --epochs 1 \
  --batch-size 8 \
  --gpus 1 \
  --stop-patience 2 \
  --stop-min-valid 4 \
  --compaction-neighborhoods 8 \
  --work-dir data/training
```

每个 case 独立计算 Stop Condition，需要连续 `--stop-patience` 轮同时满足：

- 至少 `--stop-min-valid` 个独立审计问题都能由 `M_t` 回答，且
  High-Priority Buffer 为空。
- Memory Builder 在指定数量的 neighborhood 中找不到不引起
  `linked_questions` 回归的 DELETE/MERGE。

全部 case 停止后训练结束，`--rounds` 是仍然生效的硬上限。

## 数据构建

Full Graph 不会直接交给 Verl。数据经过以下组件转换：

```text
LongMemEvalGraphReader
  → GraphRouterPolicy
  → AttackerDatasetBuilder
  → write_verl_dataset
  → train.parquet / val.parquet
```

Memory Builder 数据由防御失败的 `PendingMemoryEdit` 构建：

```text
PendingMemoryEdit
  → memory_builder_records
  → write_verl_dataset
  → train.parquet / val.parquet
```

## 输出和恢复

```text
data/training/
  run_state.json
  round_000/
    attacker_data/
    attacker/checkpoints/
    attacker/model/
    memory_builder_data/
    memory_builder/checkpoints/
    memory_builder/model/
  round_001/
    ...
```

`run_state.json` 保存：

- 每个 case 独立的 `M_t`
- 每个 case 独立的 Success Pool、High-Priority Buffer 和问题档案
- 每个 case 独立的 Stop State
- Attacker 和 Memory Builder 模型路径
- 下一个训练轮次

使用相同的 `--work-dir` 重新执行命令会从下一轮继续。使用新的
`--work-dir` 即可开始一个全新实验。旧版单一 `M_t` 的 `run_state.json`
不能继续使用。

## 核心文件

- `training/run_alternating.py`：交替训练主流程。
- `training/dataset_builder.py`：构建 Verl Parquet。
- `training/verl_runner.py`：启动 GRPO 并合并 checkpoint。
- `attacker/verl_reward.py`：Attacker Reward 入口。
- `defender/verl_reward.py`：Memory Builder Reward 入口。
- `memory/store.py`：执行记忆编辑和题库更新。

# adv_mem
