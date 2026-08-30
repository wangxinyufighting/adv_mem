import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from attacker.gap import support_coverage
from attacker.graph_router import GraphRouterPolicy, NoRouteFoundError
from attacker.models import AttackMode, MemoryGraphView, RouteProbe, RouterConfig
from attacker.selector import RouteSelector
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
) -> DatasetFiles:
    """Split verl prompt records and materialize the two Parquet files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    items = list(records)
    if not items:
        raise ValueError("Cannot build an empty verl dataset")

    rng = random.Random(seed)
    if all(_attack_mode(item) for item in items):
        train, val = _stratified_split(items, val_fraction, rng)
        val = val or train[:1]
    else:
        rng.shuffle(items)
        val_size = max(1, round(len(items) * val_fraction)) if len(items) > 1 else 1
        val = items[:val_size]
        train = items[val_size:] or items
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


class RouteProposalBuilder:
    """Generate structurally valid route candidates without choosing an attack."""

    def __init__(
        self,
        reader: LongMemEvalGraphReader,
        seed: int = 0,
    ):
        self.reader = reader
        self.seed = seed

    def routes(self, case_index: int, count: int) -> tuple:
        config = RouterConfig(random_seed=self.seed, fallback_to_single_fact=False)
        router = GraphRouterPolicy(config)
        graph = MemoryGraphView.from_case(
            self.reader.get_case(case_index),
            self.reader.version,
        )
        routes = []
        while len(routes) < count:
            added = False
            for mode in AttackMode:
                try:
                    routes.append(router.route(graph, mode))
                    added = True
                except NoRouteFoundError:
                    pass
                if len(routes) == count:
                    break
            if not added:
                break
        return tuple(routes)


class RouteSelectorDatasetBuilder:
    """Build pairwise covered-versus-uncovered route prompts."""

    def __init__(
        self,
        retriever: Any,
        selector: RouteSelector | None = None,
        seed: int = 0,
        tokenizer: Any | None = None,
        max_prompt_tokens: int = 4096,
    ):
        self.retriever = retriever
        self.selector = selector or RouteSelector()
        self.seed = seed
        self.tokenizer = tokenizer
        self.max_prompt_tokens = max_prompt_tokens

    def records(
        self,
        probes: tuple[RouteProbe, ...],
        memory: MemoryState,
    ) -> tuple[dict[str, Any], ...]:
        covered = []
        uncovered = []
        for probe in probes:
            target = (
                covered
                if support_coverage(probe, memory.active_nodes) >= 1.0
                else uncovered
            )
            target.append(probe)
        if not covered or not uncovered:
            return ()

        rng = random.Random(self.seed)
        rng.shuffle(covered)
        rng.shuffle(uncovered)
        records = []
        for index in range(max(len(covered), len(uncovered))):
            pair = [
                covered[index % len(covered)],
                uncovered[index % len(uncovered)],
            ]
            rng.shuffle(pair)
            window = tuple(pair)
            observation = self.selector.observe(window, memory, self.retriever)
            record = self.selector.to_verl_record(observation, window, memory)
            if self._fits(record):
                records.append(record)
        return tuple(records)

    def _fits(self, record: dict[str, Any]) -> bool:
        if self.tokenizer is None:
            return True
        tokens = self.tokenizer.apply_chat_template(
            record["prompt"],
            tokenize=True,
            add_generation_prompt=True,
        )
        return len(tokens) <= self.max_prompt_tokens


# The old name remains importable, but now denotes proposal generation only.
AttackerDatasetBuilder = RouteProposalBuilder


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
            train.extend(bucket)
        else:
            val.extend(bucket[:val_size])
            train.extend(bucket[val_size:])
    rng.shuffle(train)
    rng.shuffle(val)
    return train, val
