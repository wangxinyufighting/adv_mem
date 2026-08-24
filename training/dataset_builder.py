import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from attacker.attacker import Attacker
from attacker.graph_router import GraphRouterPolicy
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

    random.Random(seed).shuffle(items)
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
    return DatasetFiles(train_path, val_path, len(train), len(val))


class AttackerDatasetBuilder:
    """Sample Full Memory Graph routes and freeze the current M_t in each record."""

    def __init__(
        self,
        reader: LongMemEvalGraphReader,
        retriever: Any,
        attacker: Attacker | None = None,
        seed: int = 0,
    ):
        self.reader = reader
        self.retriever = retriever
        self.attacker = attacker or Attacker()
        self.seed = seed

    def routes(self, case_index: int, count: int) -> tuple:
        config = RouterConfig(random_seed=self.seed)
        router = GraphRouterPolicy(config)
        modes = tuple(AttackMode)
        graph = MemoryGraphView.from_case(
            self.reader.get_case(case_index),
            self.reader.version,
        )
        return tuple(
            router.route(graph, modes[index % len(modes)])
            for index in range(count)
        )

    def records(
        self,
        case_index: int,
        memory: MemoryState,
        count: int,
    ) -> tuple[dict[str, Any], ...]:
        return tuple(
            self.attacker.to_verl_record(
                self.attacker.observe(route, memory, self.retriever),
                memory,
            )
            for route in self.routes(case_index, count)
        )


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
