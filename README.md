# AdvMem Training

AdvMem 用两个交替更新的 GRPO policy 构建长期记忆：

- Route Selector 从固定 Probe Bank 中选择最可能暴露当前记忆缺口的问题。
- Memory Builder 只为确定好的 repair plan 写一段 memory content。

Graph Router、问题生成、Oracle 验证只在离线阶段运行一次。训练过程中不再生成
Route 或改写问题；ADD/MERGE 与 target memory 由 provenance 规则确定。

完整设计、数据结构和逐文件说明见 [MINIMAL_PIPELINE.md](MINIMAL_PIPELINE.md)。

## 核心流程

```text
离线：Graph → Route → fixed question → Oracle → Probe Bank

训练：Probe Bank → Route Selector → gap evaluation
                         ↓ failure
        Repair Controller → Memory Builder(content only)
                         ↓
        grounding → answer/retention → guarded commit
```

训练执行固定 `--rounds`。没有在线 Probe 生成、第二套 audit route pool、复杂 stop
condition 或 compaction。

## 环境

需要 Linux x86_64、CUDA GPU、NVIDIA Driver 525+ 和 Python 3.10-3.12。
首次运行 `train.sh` 会建立 CUDA 12.4 环境，也可提前执行：

```bash
bash scripts/setup_cuda124.sh
cp .env.example .env
```

主要服务：

- DeepSeek：Probe Oracle、answer judge、memory grounding judge、query parser。
- Qwen3-1.7B：训练 backbone 和冻结 Answer Agent。
- `text-embedding-v4` 与 BGE reranker：训练和验证共用的 memory retrieval。

## 1. 离线生成 Probe Bank

先启动冻结 Answer Agent：

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/serve_answer_agent.sh
```

然后一次性生成 Bank。建议每个 case 至少 32 个有效 Probe，而不是每轮临时生成
16 条 Route：

```bash
PYTHONPATH=. .venv-cu124/bin/python -m scripts.build_probe_bank \
  --graph ./data/longmemeval/memory_graph_v4_10.json \
  --graph-version v4 \
  --output ./data/longmemeval/probe_bank_v4_10.json \
  --probes-per-case 32 \
  --routes-per-batch 16 \
  --max-routes-per-case 512
```

脚本按 case 保存进度；再次运行会复用已完成的 Probe。Bank 中每个 Probe 都包含
固定 question、canonical answer、supporting evidence 和 Route，不随 GRPO 更新。

## 2. 训练

使用一个新的 `--work-dir` 启动：

```bash
bash train.sh \
  --probe-bank ./data/longmemeval/probe_bank_v4_10.json \
  --model /root/autodl-tmp/models/Qwen3-1.7B \
  --work-dir ./data/training_10_minimal_v1 \
  --rounds 5 \
  --epochs 1 \
  --batch-size 2 \
  --candidates-per-case 8 \
  --selector-candidates 2
```

`train.sh` 会启动 Answer Agent 和 reranker。使用远程服务时，把 `.env` 中的
`START_ANSWER_AGENT` 或 `START_RERANKER` 设为 `0`。

旧版 `run_state.json` 与新 Builder 输出协议不兼容；程序会明确报错并要求新的
`--work-dir`，避免静默复用错误 checkpoint。

## 输出

```text
work_dir/
  run_state.json
  services/
  round_000/
    attacker_data/
    attacker/
    memory_builder_data/
    memory_builder/
  round_001/
    ...
```

每个 case 的 `MemoryState`、两个 policy checkpoint 和下一轮编号都保存在
`run_state.json`。Probe Bank 是独立的只读输入，不再复制进每轮状态。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall attacker defender memory training scripts tests
```
