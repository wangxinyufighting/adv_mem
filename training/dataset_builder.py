import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from attacker.attacker import Attacker
from attacker.graph_router import GraphRouterPolicy, NoRouteFoundError
from attacker.models import AttackMode, MemoryGraphView, RouterConfig
from defender.memory_builder import MemoryBuilder
from memory.models import MemoryState
from training.alternating import PendingMemoryEdit
from utils.longmemeval_graph_reader import LongMemEvalGraphReader


@dataclass(frozen=True)
class DatasetFiles:
    train: Path
    val: Path
    train_size: int
    val_size: int
    train_modes: dict[str, int] = field(default_factory=dict)
    val_modes: dict[str, int] = field(default_factory=dict)


def write_verl_dataset(
    records: Iterable[dict[str, Any]],
    output_dir: str | Path,
    val_fraction: float = 0.1,
    seed: int = 0,
    minimum_train_size: int = 8,
) -> DatasetFiles:
    """Split verl prompt records and materialize the two Parquet files."""
    items = list(records)
    if not items:
        raise ValueError("Cannot build an empty verl dataset")

    rng = random.Random(seed)
    if all(_attack_mode(item) for item in items):
        train, val = _stratified_split(items, val_fraction, rng)
        train = _balance_modes(train, rng)
    else:
        rng.shuffle(items)
        val_size = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 1
        val = items[:val_size]
        train = items[val_size:] or items
    train = _repeat_to_size(train, minimum_train_size)

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    train_path = directory / "train.parquet"
    val_path = directory / "val.parquet"
    pq.write_table(pa.Table.from_pylist(train), train_path)
    pq.write_table(pa.Table.from_pylist(val), val_path)
    return DatasetFiles(
        train_path,
        val_path,
        len(train),
        len(val),
        _mode_counts(train),
        _mode_counts(val),
    )


class AttackerDatasetBuilder:
    """Sample Full Memory Graph routes and freeze the current M_t in each record."""

    def __init__(
        self,
        reader: LongMemEvalGraphReader,
        retriever: Any,
        attacker: Attacker | None = None,
        seed: int = 0,
        tokenizer: Any | None = None,
        max_prompt_tokens: int = 4096,
    ):
        self.reader = reader
        self.retriever = retriever
        self.attacker = attacker or Attacker()
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens
        self.stats = {
            stage: {mode.value: 0 for mode in AttackMode}
            for stage in (
                "sampled",
                "unique",
                "attempted",
                "within_limit",
                "selected",
            )
        }

    def routes(self, case_index: int, count: int) -> tuple:
        config = RouterConfig(random_seed=self.seed, fallback_to_single_fact=False)
        router = GraphRouterPolicy(config)
        graph = MemoryGraphView.from_case(
            self.reader.get_case(case_index),
            self.reader.version,
        )
        routes = []
        modes = []
        for mode in AttackMode:
            try:
                routes.append(router.route(graph, mode))
                modes.append(mode)
            except NoRouteFoundError:
                pass
        while len(routes) < count:
            routes.append(router.route(graph, modes[len(routes) % len(modes)]))
        return tuple(routes[:count])

    def records(
        self,
        case_index: int,
        memory: MemoryState,
        count: int,
    ) -> tuple[dict[str, Any], ...]:
        buckets = {mode: [] for mode in AttackMode}
        seen = set()
        for route in self.routes(case_index, count * len(AttackMode)):
            mode = route.attack_mode.value
            self.stats["sampled"][mode] += 1
            if route.route_signature in seen:
                continue
            seen.add(route.route_signature)
            self.stats["unique"][mode] += 1
            buckets[route.attack_mode].append(route)

        selected = []
        offsets = {mode: 0 for mode in AttackMode}
        while len(selected) < count:
            added = False
            for mode in AttackMode:
                while offsets[mode] < len(buckets[mode]):
                    route = buckets[mode][offsets[mode]]
                    offsets[mode] += 1
                    self.stats["attempted"][mode.value] += 1
                    record = self.attacker.to_verl_record(
                        self.attacker.observe(route, memory, self.retriever),
                        memory,
                    )
                    if not self._fits(record):
                        continue
                    selected.append(record)
                    self.stats["within_limit"][mode.value] += 1
                    self.stats["selected"][mode.value] += 1
                    added = True
                    break
                if len(selected) == count:
                    break
            if not added:
                break
        return tuple(selected)

    def _fits(self, record: dict[str, Any]) -> bool:
        if self.tokenizer is None:
            return True
        tokens = self.tokenizer.apply_chat_template(
            record["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens) <= self.max_prompt_tokens


def memory_builder_records(
    pending: Iterable[PendingMemoryEdit],
    memory: MemoryState,
    builder: MemoryBuilder | None = None,
) -> tuple[dict[str, Any], ...]:
    editor = builder or MemoryBuilder()
    return tuple(
        editor.to_verl_record(
            item.observation,
            memory,
            item.candidate.oracle,
            item.before_correctness,
        )
        for item in pending
    )


def _repeat_to_size(items: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    return [items[index % len(items)] for index in range(max(size, len(items)))]


def _attack_mode(record: dict[str, Any]) -> str | None:
    route = record["extra_info"].get("route")
    return route.get("attack_mode") if route else None


def _mode_counts(items: list[dict[str, Any]]) -> dict[str, int]:
    return {
        mode.value: sum(_attack_mode(item) == mode.value for item in items)
        for mode in AttackMode
        if any(_attack_mode(item) == mode.value for item in items)
    }


def _stratified_split(
    items: list[dict[str, Any]],
    val_fraction: float,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = []
    val = []
    for mode in AttackMode:
        bucket = [item for item in items if _attack_mode(item) == mode.value]
        if not bucket:
            continue
        rng.shuffle(bucket)
        val_size = min(max(1, round(len(bucket) * val_fraction)), len(bucket) - 1)
        if len(bucket) == 1:
            val.extend(bucket)
            train.extend(bucket)
        else:
            val.extend(bucket[:val_size])
            train.extend(bucket[val_size:])
    return train, val


def _balance_modes(
    items: list[dict[str, Any]],
    rng: random.Random,
) -> list[dict[str, Any]]:
    buckets = {
        mode: [item for item in items if _attack_mode(item) == mode.value]
        for mode in AttackMode
    }
    nonempty = [bucket for bucket in buckets.values() if bucket]
    target = (len(items) + len(nonempty) - 1) // len(nonempty)
    balanced = [
        bucket[index % len(bucket)]
        for bucket in nonempty
        for index in range(target)
    ]
    rng.shuffle(balanced)
    return balanced
