import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from attacker.gap import route_novelty_values, route_support_coverage
from attacker.graph_router import GraphRouterPolicy, NoRouteFoundError
from attacker.models import (
    AttackMode,
    GraphRouteBundle,
    MemoryGraphView,
    RouterConfig,
    RouterState,
)
from attacker.probe_cache import ProbeCache
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
    return DatasetFiles(train_path, val_path, len(train), len(val))


class RouteProposalBuilder:
    """Generate structurally valid route candidates without choosing an attack."""

    def __init__(
        self,
        reader: LongMemEvalGraphReader,
        seed: int = 0,
    ):
        self.reader = reader
        self.seed = seed

    def routes(
        self,
        case_index: int,
        count: int,
        state: RouterState | None = None,
    ) -> tuple[GraphRouteBundle, ...]:
        config = RouterConfig(random_seed=self.seed, fallback_to_single_fact=False)
        router = GraphRouterPolicy(config, state)
        graph = MemoryGraphView.from_case(
            self.reader.get_case(case_index),
            self.reader.version,
        )
        if count <= 0:
            return ()
        try:
            routes = [router.coverage_route(graph)]
        except NoRouteFoundError:
            return ()
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
    """Build pairwise route-value contrast prompts."""

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
        routes: tuple[GraphRouteBundle, ...],
        memory: MemoryState,
        probe_cache: ProbeCache | None = None,
    ) -> tuple[dict[str, Any], ...]:
        values = {
            route.route_id: value
            for route, value in zip(
                routes,
                route_novelty_values(routes, memory),
                strict=True,
            )
        }
        covered = []
        uncovered = []
        for route in routes:
            target = (
                covered
                if route_support_coverage(route, memory.active_nodes) >= 1.0
                else uncovered
            )
            target.append(route)

        rng = random.Random(self.seed)
        rng.shuffle(covered)
        rng.shuffle(uncovered)
        if covered and uncovered:
            pairs = [
                (
                    covered[index % len(covered)],
                    uncovered[index % len(uncovered)],
                )
                for index in range(max(len(covered), len(uncovered)))
            ]
        else:
            ranked = sorted(routes, key=lambda item: values[item.route_id])
            pairs = []
            while len(ranked) >= 2:
                pairs.append((ranked.pop(0), ranked.pop()))

        records = []
        for pair in pairs:
            pair = list(pair)
            rng.shuffle(pair)
            window = tuple(pair)
            observation = self.selector.observe(window, memory, self.retriever)
            record = self.selector.to_verl_record(
                observation,
                window,
                memory,
                tuple(values[item.route_id] for item in window),
                tuple(
                    probe
                    for item in window
                    if probe_cache and (probe := probe_cache.get(item))
                ),
            )
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
