# AdvMem Training

AdvMem 通过交替 GRPO 训练 Route Selector 和 Memory Builder。Graph Router 只
提出结构合法的候选路径；Route Selector 根据当前记忆 `M_t`、候选证据、固定
Probe Question 和历史攻击结果，学习选择最可能暴露缺口的路径。Memory Builder
再通过编辑记忆修复这些能力。

## 训练流程

```text
Full Memory Graph
  → Graph Router 提出候选 Route Pool
  → 冻结生成器为新 Route 生成一次 Probe Question
  → Oracle 验证并缓存标准答案和 Evidence
  → 每个 GRPO Prompt 同时包含多条候选 Route
  → Verl GRPO 训练 Route Selector 选择攻击位置
  → 按 provenance 区分 Storage / Retrieval / Reasoning Gap
  → Answer Agent 依次使用 Golden Corpus、无上下文和 M_t 回答
  → 无上下文已能回答或 Golden Corpus 仍不能回答：丢弃 Question
  → M_t 回答正确：写入 Success Pool
  → M_t 回答错误：构建 Memory Builder Parquet
  → Verl GRPO 训练 Memory Builder
  → Memory Builder 按 gap_type 生成 ADD / MERGE
  → Reward 在无 provenance 的 M_temp 上验证结构、当前答案和局部回归
  → Commit 在可信快照上复验当前问题和全部历史成功问题
  → 全部通过：原子提交 M_temp；否则回滚并写入 High-Priority Buffer
  → 连续多轮无有效攻击且 High-Priority Buffer 为空：停止
```

Route Selector 和 Memory Builder 按轮次交替训练，不同时更新。Probe Question
一经 Oracle 验证便按 Route 缓存，不参与 GRPO 更新，因此 Route 的 reward 不会
被问题措辞混淆。Retrieval 在训练和测试中始终使用同一套实现。

## 缺口定义

- `storage_gap`：Oracle 所需 provenance/source 未被当前 active memory 覆盖。
- `retrieval_gap`：支持信息存在于 `M_t`，但没有完整进入固定 top-k。
- `reasoning_gap`：支持信息已检索到，但固定 Answer Agent 仍回答错误。
- `none`：固定 Answer Agent 能从 `M_t` 正确回答。

Route Selector 的主 reward 来自缺口类型，另加随历史尝试次数衰减的探索奖励。
Question validity、标准答案和 evidence 在 Probe 创建阶段完成，不再由每个 rollout
重复判断。

Memory Builder 的 repair 与 compaction 分离：repair 只允许 `ADD/MERGE`；
`DELETE/NOOP` 只用于可选的 compaction。结构 Judge 只返回 `grounded`、
`evidence_covered`、`targets_preserved` 三个布尔值；答案正确性与历史能力回归统一
交给同一个语义 Judge。Compaction 的候选必须通过结构验证和全部 Success Pool
回归测试，但压缩成功或失败都不作为收敛证据。Compaction 默认关闭；只有显式
设置 `--compaction-neighborhoods` 才会启用。

## 环境

训练需要 Linux x86_64、CUDA GPU、525 或更新的 NVIDIA Driver，以及
Python 3.10-3.12。训练环境固定为：

- CUDA 12.4
- PyTorch 2.6.0+cu124
- vLLM 0.8.5.post1
- FlashAttention 2.7.4.post1
- Setuptools 80.9.0
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
- DeepSeek（默认）：冻结的 Probe Question Generator；可配置为其他兼容接口。
- Qwen3-0.6B：固定的 Answer Agent。
- `text-embedding-v4`：Memory Retrieval embedding。
- BGE Reranker：提供 `/v1/rerank` 接口。

配置环境变量：

```bash
export DEEPSEEK_API_KEY=xxxx
export DEEPSEEK_API_BASE=https://api.deepseek.com
export DEEPSEEK_MODEL=deepseek-chat

# 可选；留空时复用 DEEPSEEK_*。
export PROBE_GENERATOR_API_KEY=
export PROBE_GENERATOR_API_BASE=
export PROBE_GENERATOR_MODEL=

export MOS_EMBEDDER_API_KEY=xxxx
export MOS_EMBEDDER_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
export MOS_EMBEDDER_MODEL=text-embedding-v4
export EMBEDDING_DIMENSION=1024
export EMBEDDING_BATCH_SIZE=10

export BGE_RERANKER_URL=http://localhost:8000/v1/rerank
export BGE_RERANKER_MODEL=bge-reranker-v2-m3

export ANSWER_AGENT_API_BASE=http://localhost:8001/v1
export ANSWER_AGENT_MODEL=Qwen/Qwen3-0.6B
export TRAIN_MODEL=Qwen/Qwen3-0.6B
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
- 每个 case 提出 16 条候选 Route
- 每个 Route Selector prompt 包含 8 条候选 Route
- 每轮为每个 case 处理 8 个 Question
- Route Selector 和 Memory Builder 各训练 1 epoch

常用参数：

```bash
bash train.sh \
  --rounds 5 \
  --routes-per-case 16 \
  --selector-candidates 8 \
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

- 至少 `--stop-min-valid` 个由 Selector 在新 Route Pool 中选出的独立 Probe
  都能由 `M_t` 回答，且
  High-Priority Buffer 为空。

达到条件后可在最多 `--compaction-neighborhoods` 个相关 memory pair 上尝试
`DELETE/MERGE`。只有全部 Success Pool 问题都不回归时才提交压缩；没有可压缩项
不会额外增加或清零收敛轮数。

全部 case 停止后训练结束，`--rounds` 是仍然生效的硬上限。

## 数据构建

从 Neo4j 导出指定 version 的前 `n` 个 case：

```bash
python scripts/export_memory_graph.py --version fullgraph5 --num-cases 5
```

脚本会导出 case `0..n-1`，默认写入
`data/longmemeval/memory_graph_<version>.json`。Neo4j 连接使用
`.env.example` 中的 `NEO4J_*` 变量，也可用 `--output` 指定其他路径。

Full Graph 不会直接交给 Verl。数据经过以下组件转换：

```text
LongMemEvalGraphReader
  → GraphRouterPolicy
  → RouteProposalBuilder
  → ProbeFactory
  → RouteSelectorDatasetBuilder
  → write_verl_dataset
  → train.parquet / val.parquet
```

`RouteProposalBuilder` 不决定最终攻击目标。`RouteSelectorDatasetBuilder` 将多条
Route 组成同一个 prompt，使同一 GRPO group 的 rollout 能比较不同攻击位置。
旧版 `run_state.json` 可以读取；检测到旧 Question-Attacker checkpoint 时会保留
Memory Builder 与记忆状态，但从 `--model` 重新初始化 Route Selector。

Memory Builder 数据由防御失败的 `PendingMemoryEdit` 构建：

```text
PendingMemoryEdit
  → memory_builder_records
  → write_verl_dataset
  → train.parquet / val.parquet
```

Observation 会显式包含 `gap_type`、当前 top-k neighborhood 和按 provenance 找到的
隐藏 support。`storage_gap` 且没有 support 时只能 `ADD`；`retrieval_gap` 与
`reasoning_gap` 必须 `MERGE` 全部 support，避免重复写入已经存在的事实。

## 输出和恢复

```text
data/training/
  services/training.log
  run_state.json
  round_000/
    attacker_data/
    attacker/checkpoints/  # Route Selector
      rollouts/
        <step>.jsonl
        reward_trace.jsonl
    attacker/model/
    memory_builder_data/
    memory_builder/checkpoints/
      rollouts/<step>.jsonl
    memory_builder/model/
  round_001/
    ...
```

`services/training.log` 保存本次启动的终端日志，再次启动时会覆盖旧日志。
Answer Agent 和 Reranker 日志也保存在 `services/` 中。
`rollouts/<step>.jsonl` 保存 prompt、模型输出和 reward 分项；Route Selector 的
`reward_trace.jsonl` 额外保存 route choice、gap type、coverage 和 Memory Answer。

`run_state.json` 保存：

- 每个 case 独立的 `M_t`
- 每个 case 独立的 Probe Cache、Success Pool、High-Priority Buffer 和问题档案
- 每条 Route 的跨轮攻击次数、gap 类型和最近验证的 memory version
- 每个 case 独立的 Stop State
- Route Selector 和 Memory Builder 模型路径
- 下一个训练轮次

使用相同的 `--work-dir` 重新执行命令会从下一轮继续。使用新的
`--work-dir` 即可开始一个全新实验。旧版单一 `M_t` 的 `run_state.json`
不能继续使用。

## 核心文件

- `attacker/selector.py`：候选 Route 观察、选择输出和 Verl record。
- `attacker/probe.py`：冻结 Probe 生成、Oracle 验证与缓存对象构建。
- `attacker/gap.py`：Storage / Retrieval / Reasoning Gap 归因。
- `training/run_alternating.py`：分层选择与 Memory Builder 交替训练主流程。
- `training/dataset_builder.py`：构建 Route Pool 和 Selector Verl Parquet。
- `training/verl_runner.py`：启动 GRPO 并合并 checkpoint。
- `attacker/verl_reward.py`：Route Selector Reward 入口。
- `defender/verl_reward.py`：Memory Builder Reward 入口。
- `memory/store.py`：执行记忆编辑和题库更新。
