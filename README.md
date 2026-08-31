# AdvMem Training

AdvMem 用两个交替更新的 GRPO policy 从完整 Memory Graph 构建长期记忆：

- Route Selector 从 Graph Router 每轮提出的 Path 中选择当前更可能存在缺口的位置；
- Memory Builder 只为确定好的 ADD/MERGE plan 写一段 memory content。

Graph 始终是训练的信息源。问题只在 Route 被 policy rollout 或正式攻击选中后，由冻结
Question Generator 生成并经 Oracle 验证。合法问题写入 `probe_cache.json` 复用，但
cache 不限制后续 Router 继续探索 Graph。

文档：

- [详细代码与算法逻辑](docs/ALGORITHM_AND_CODE_GUIDE.md)
- [极简架构摘要](MINIMAL_PIPELINE.md)

## 核心流程

```text
每轮：Graph → Router 动态 Paths → Route Selector
                                  ↓ selected
               Lazy Question → Oracle → gap evaluation
                                             ↓ failure
               Repair Controller → Memory Builder(content only)
                                             ↓
               grounding → answer/retention → guarded commit
```

系统保留三项轻量保障状态：

- `node_visit_counts`：每轮至少尝试一条包含最少访问 eligible evidence 的 Route；
- `success_pool`：正式 commit 必须回归全部已验证能力；
- `high_priority_buffer`：修复失败或服务不可用的问题下一轮优先重放。

## 环境

需要 Linux x86_64、CUDA GPU、NVIDIA Driver 525+ 和 Python 3.10–3.12。
首次运行 `train.sh` 会建立 CUDA 12.4 环境，也可提前执行：

```bash
bash scripts/setup_cuda124.sh
cp .env.example .env
```

主要服务：

- DeepSeek：惰性问题生成、Probe Oracle、answer judge、memory grounding judge；
- Qwen3-1.7B：两个 GRPO policy 和冻结 Answer Agent；
- `text-embedding-v4` 与 BGE reranker：memory retrieval。

## 训练

不需要提前构建 Probe Bank，直接传入原始 Graph：

```bash
bash train.sh \
  --graph ./data/longmemeval/memory_graph_v4_10.json \
  --graph-version v4 \
  --model /root/autodl-tmp/models/Qwen3-1.7B \
  --work-dir ./data/training_10_v3.0 \
  --rounds 5 \
  --epochs 1 \
  --routes-per-case 32 \
  --candidates-per-case 8 \
  --selector-candidates 2 \
  --batch-size 2
```

关键参数：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--routes-per-case` | 每个 case、每轮从 Graph 提出的动态 Route 数 | 32 |
| `--candidates-per-case` | 每个 case、每轮实际测试/修复的最大 Probe 数 | 8 |
| `--selector-candidates` | 每个 Selector prompt 中比较的 Route 数 | 2 |
| `--rounds` | 固定交替训练轮数 | 1 |
| `--epochs` | 每次 GRPO 调用的数据 epoch 数 | 2 |

冷启动时 $M_0$ 为空，绝大多数 Route 都会失败。Attacker 仍然训练，初期主要依靠
evidence novelty 区分 Route；Router 的最少访问节点规则负责覆盖探索，Builder 自然
主要执行 ADD。随着 Memory 增长，回答正确率和重复惩罚开始提供更强的选择信号。

旧版 `minimal_memory_loop_v1` 的 `run_state.json` 与当前动态图流程不兼容。第一次运行
请使用新的 `--work-dir`，程序会拒绝静默复用旧状态。

## 输出

```text
work_dir/
  run_state.json            # MemoryState、模型路径、Graph 覆盖计数
  probe_cache.json          # 已选 Route 的合法问题缓存
  services/
  round_000/
    attacker_data/
    attacker/
    memory_builder_data/
    memory_builder/
```

`probe_cache.json` 是运行状态的一部分，不是固定训练集。Router 每轮仍从 Graph 提出
新 Route；同一 Route 在后续 memory version 也允许重新测试。

## 测试

```bash
python -m unittest discover -s tests -v
python -m compileall attacker defender memory training scripts tests
```
