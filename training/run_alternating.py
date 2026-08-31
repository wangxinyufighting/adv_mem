import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

from attacker.answer_agent import QwenAnswerAgent
from attacker.models import GraphRouteBundle
from attacker.oracle import DeepSeekOracle
from attacker.probe import FixedProbeQuestionGenerator, ProbeFactory
from attacker.probe_cache import ProbeCache
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.selector import RouteSelector
from defender.memory_builder import MemoryBuilder
from defender.reward import MemoryBuilderReward
from memory.store import MemoryStore
from training.alternating import MemoryTrainingFlow, PendingMemoryEdit
from training.dataset_builder import (
    RouteProposalBuilder,
    RouteSelectorDatasetBuilder,
    memory_builder_records,
    write_verl_dataset,
)
from training.policy_server import VLLMPolicyServer
from training.run_state import RunState
from training.verl_runner import VerlConfig, VerlRunner
from utils.longmemeval_graph_reader import LongMemEvalGraphReader
from utils.memory_retrieval import HybridMemoryRetriever


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    rounds: int
    candidates_per_case: int
    selector_candidates: int
    work_dir: Path
    policy_port: int
    gpu_memory_utilization: float


@dataclass
class CaseRound:
    store: MemoryStore
    flow: MemoryTrainingFlow
    selected: int
    replayed: int
    explored: int
    pending: tuple[PendingMemoryEdit, ...]
    defended: int
    deferred: int
    committed: int = 0
    discarded: int = 0


def run(config: RunConfig, args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    if args.routes_per_case < 2 or config.selector_candidates < 2:
        raise ValueError("Route and selector candidate counts must be at least two")
    if config.candidates_per_case < 1:
        raise ValueError("candidates-per-case must be positive")
    reader = LongMemEvalGraphReader(args.graph, args.longmemeval, args.graph_version)
    case_indices = reader.available_cases()
    if not case_indices:
        raise ValueError("Graph contains no cases for the requested graph version")
    state_path = config.work_dir / "run_state.json"
    state = RunState.load(
        state_path,
        args.model,
        case_indices,
        args.graph_version,
    )
    probe_cache = ProbeCache(args.graph_version, config.work_dir / "probe_cache.json")
    retriever = HybridMemoryRetriever.from_env()
    selector = RouteSelector()
    builder = MemoryBuilder()
    answer_agent = QwenAnswerAgent.from_env()
    answer_judge = DeepSeekRewardJudge.from_env()
    probe_factory = ProbeFactory(
        FixedProbeQuestionGenerator.from_env(),
        DeepSeekOracle(),
        answer_agent,
        answer_judge,
    )
    reward_model = MemoryBuilderReward.from_env()
    runner = VerlRunner(
        ROOT,
        VerlConfig(
            args.epochs,
            args.batch_size,
            args.gpus,
            args.max_prompt_length,
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(state.attacker_model)

    for round_index in range(state.next_round, state.next_round + config.rounds):
        round_dir = config.work_dir / f"round_{round_index:03d}"
        proposals = {}
        for case, case_state in state.cases.items():
            proposals[case] = RouteProposalBuilder(
                reader,
                args.seed + round_index * 10_000 + case,
            ).routes(case, args.routes_per_case, case_state.router_state())

        data_builder = RouteSelectorDatasetBuilder(
            retriever,
            selector,
            seed=args.seed + round_index,
            tokenizer=tokenizer,
            max_prompt_tokens=args.max_prompt_length,
        )
        attacker_records = [
            record
            for case, case_state in state.cases.items()
            for record in data_builder.records(
                proposals[case],
                case_state.memory,
                probe_cache,
            )
        ]
        if attacker_records:
            data = write_verl_dataset(
                attacker_records,
                round_dir / "attacker_data",
                seed=args.seed + round_index,
            )
            print(
                f"Round {round_index}: training Route Selector on "
                f"{data.train_size} prompts"
            )
            state.attacker_model = runner.train(
                "attacker",
                state.attacker_model,
                data,
                round_dir / "attacker",
                probe_cache.path,
            )
            probe_cache.refresh()

        candidates = {}
        replayed = {}
        explored = {}
        with VLLMPolicyServer(
            runner.verl_dir,
            state.attacker_model,
            config.policy_port,
            round_dir / "attacker_server.log",
            config.gpu_memory_utilization,
        ) as policy:
            for case, case_state in state.cases.items():
                by_id = probe_cache.by_question()
                priority_limit = (
                    max(1, config.candidates_per_case // 2)
                    if config.candidates_per_case
                    else 0
                )
                priority = tuple(
                    by_id[question_id]
                    for question_id in case_state.memory.high_priority_buffer
                    if question_id in by_id
                    and by_id[question_id].route.case_index == case
                )[:priority_limit]
                priority_routes = {probe.route.route_id for probe in priority}
                passed_routes = {
                    record.route_id
                    for record in case_state.memory.capability_ledger.values()
                    if record.passed
                }
                pool = [
                    route
                    for route in proposals[case]
                    if route.route_id not in priority_routes | passed_routes
                ]
                budget = config.candidates_per_case - len(priority)
                coverage = tuple(pool[:1]) if budget else ()
                coverage_ids = {route.route_id for route in coverage}
                pool = [route for route in pool if route.route_id not in coverage_ids]
                random.Random(
                    args.seed + round_index * 10_000 + case
                ).shuffle(pool)
                fresh = selector.select_many(
                    policy,
                    tuple(pool),
                    case_state.memory,
                    retriever,
                    budget - len(coverage),
                    config.selector_candidates,
                )
                selected_routes = coverage + fresh
                for route in selected_routes:
                    _record_visit(case_state.node_visit_counts, route)
                materialized = []
                for route in selected_routes:
                    probe = probe_cache.get(route)
                    if probe is None:
                        probe = probe_factory.build(route)
                        if probe:
                            probe_cache.put(probe)
                    if probe:
                        materialized.append(probe)
                candidates[case] = priority + tuple(materialized)
                replayed[case] = len(priority)
                explored[case] = len(coverage)

        case_rounds = {}
        builder_records = []
        for case, selected in candidates.items():
            store = MemoryStore(state.cases[case].memory)
            flow = MemoryTrainingFlow(
                store, retriever, answer_agent, answer_judge, builder
            )
            evaluated = tuple(flow.try_process_question(probe) for probe in selected)
            deferred = 0
            for probe, (available, _) in zip(selected, evaluated, strict=True):
                if not available:
                    flow.defer_question(probe)
                    deferred += 1
            pending = tuple(
                item for available, item in evaluated if available and item is not None
            )
            case_rounds[case] = CaseRound(
                store=store,
                flow=flow,
                selected=len(selected),
                replayed=replayed[case],
                explored=explored[case],
                pending=pending,
                defended=sum(
                    available and item is None for available, item in evaluated
                ),
                deferred=deferred,
            )
            builder_records.extend(memory_builder_records(pending, store.state, builder))

        if builder_records:
            data = write_verl_dataset(
                builder_records,
                round_dir / "memory_builder_data",
                seed=args.seed + round_index,
            )
            print(
                f"Round {round_index}: training Memory Builder on "
                f"{data.train_size} prompts"
            )
            state.builder_model = runner.train(
                "memory_builder",
                state.builder_model,
                data,
                round_dir / "memory_builder",
            )
            _repair_cases(
                case_rounds,
                builder,
                reward_model,
                runner,
                state.builder_model,
                config,
                round_dir,
            )

        for case, case_round in case_rounds.items():
            case_round.store.advance_iteration()
            state.cases[case].memory = case_round.store.state
            print(
                f"Round {round_index}, case {case}: "
                f"selected={case_round.selected} replayed={case_round.replayed} "
                f"explored={case_round.explored} deferred={case_round.deferred} "
                f"defended={case_round.defended} "
                f"committed={case_round.committed} "
                f"discarded={case_round.discarded} "
                f"memory_nodes={len(case_round.store.state.active_nodes)}"
            )
        state.next_round = round_index + 1
        state.save(state_path)


def _record_visit(counts: dict[str, int], route: GraphRouteBundle) -> None:
    for node in route.evidence_nodes:
        counts[node.id] = counts.get(node.id, 0) + 1


def _repair_cases(
    case_rounds: dict[int, CaseRound],
    builder: MemoryBuilder,
    reward_model: MemoryBuilderReward,
    runner: VerlRunner,
    model: str,
    config: RunConfig,
    round_dir: Path,
) -> None:
    with VLLMPolicyServer(
        runner.verl_dir,
        model,
        config.policy_port,
        round_dir / "memory_builder_server.log",
        config.gpu_memory_utilization,
    ) as policy:
        for case_round in case_rounds.values():
            for old in case_round.pending:
                available, pending = case_round.flow.try_process_question(old.candidate)
                if not available:
                    case_round.flow.defer_question(old.candidate)
                    case_round.deferred += 1
                    continue
                if pending is None:
                    case_round.defended += 1
                    continue
                response = policy.generate(
                    builder.build_prompt(pending.observation), 256
                )
                reward = reward_model.evaluate(
                    response, case_round.flow.reward_context(pending)
                )
                if not reward["reward_available"]:
                    case_round.flow.defer_question(pending.candidate)
                    case_round.deferred += 1
                    continue
                if (
                    not reward["commit_valid"]
                    or reward["score"] <= case_round.flow.commit_threshold
                ):
                    case_round.flow.discard(pending)
                    case_round.discarded += 1
                    continue
                content = builder.parse_content(response)
                if case_round.flow.commit(pending, content, reward):
                    case_round.committed += 1
                else:
                    case_round.discarded += 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic graph-route GRPO training")
    parser.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data/longmemeval/memory_graph_fullgraph5.json",
    )
    parser.add_argument(
        "--longmemeval",
        type=Path,
        default=ROOT / "data/longmemeval/longmemeval_s.json",
    )
    parser.add_argument("--graph-version", default="fullgraph5")
    parser.add_argument(
        "--model",
        default=os.getenv(
            "TRAIN_MODEL", os.getenv("ANSWER_AGENT_MODEL", "Qwen/Qwen3-1.7B")
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--selector-candidates", type=int, default=2)
    parser.add_argument("--candidates-per-case", type=int, default=8)
    parser.add_argument("--routes-per-case", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--policy-port", type=int, default=8002)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        RunConfig(
            args.rounds,
            args.candidates_per_case,
            args.selector_candidates,
            args.work_dir,
            args.policy_port,
            args.gpu_memory_utilization,
        ),
        args,
    )


if __name__ == "__main__":
    main()
