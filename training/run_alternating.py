import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

from attacker.answer_agent import QwenAnswerAgent
from attacker.oracle import DeepSeekOracle
from attacker.probe import FixedProbeQuestionGenerator, ProbeFactory
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.selector import RouteSelector
from defender.memory_builder import MemoryBuilder
from defender.reward import MemoryBuilderReward
from defender.reward_judge import DeepSeekMemoryJudge
from memory.store import MemoryStore
from training.alternating import (
    MemoryTrainingFlow,
    PendingMemoryEdit,
    QuestionCandidate,
)
from training.dataset_builder import (
    RouteProposalBuilder,
    RouteSelectorDatasetBuilder,
    memory_builder_records,
    write_verl_dataset,
)
from training.policy_server import VLLMPolicyServer
from training.run_state import RunState
from training.stop_condition import (
    CompactionAuditor,
    StopCondition,
    StopConfig,
)
from training.verl_runner import VerlConfig, VerlRunner
from utils.longmemeval_graph_reader import LongMemEvalGraphReader
from utils.memory_retrieval import HybridMemoryRetriever


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class RunConfig:
    rounds: int
    routes_per_case: int
    candidates_per_case: int
    work_dir: Path
    policy_port: int
    gpu_memory_utilization: float
    selector_candidates: int


@dataclass
class CaseRound:
    store: MemoryStore
    flow: MemoryTrainingFlow
    candidates: tuple[QuestionCandidate, ...]
    fresh: tuple[QuestionCandidate, ...]
    pending: tuple[PendingMemoryEdit, ...]
    defended: int
    committed: int = 0
    discarded: int = 0
    saturated: bool = False
    compaction_attempts: int = 0


def run(config: RunConfig, args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    state_path = config.work_dir / "run_state.json"
    reader = LongMemEvalGraphReader(args.graph, args.longmemeval, args.graph_version)
    state = RunState.load(state_path, args.model, reader.available_cases())
    retriever = HybridMemoryRetriever.from_env()
    selector = RouteSelector()
    builder = MemoryBuilder()
    answer_agent = QwenAnswerAgent.from_env()
    answer_judge = DeepSeekRewardJudge.from_env()
    oracle = DeepSeekOracle()
    probe_factory = ProbeFactory(
        FixedProbeQuestionGenerator.from_env(),
        oracle,
        answer_agent,
        answer_judge,
    )
    stop_config = StopConfig(
        patience=args.stop_patience,
        min_valid_questions=args.stop_min_valid,
        max_neighborhoods=args.compaction_neighborhoods,
    )
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
        active_cases = {
            case_index: case_state
            for case_index, case_state in state.cases.items()
            if not case_state.stopped
        }
        if not active_cases:
            print("All cases have reached the stop condition")
            break

        round_dir = config.work_dir / f"round_{round_index:03d}"
        proposal_builder = RouteProposalBuilder(
            reader,
            seed=args.seed + round_index,
        )
        selector_data_builder = RouteSelectorDatasetBuilder(
            retriever,
            selector,
            seed=args.seed + round_index,
            tokenizer=tokenizer,
            max_prompt_tokens=args.max_prompt_length,
        )
        training_probes = {}
        attacker_records = []
        for case_index, case_state in active_cases.items():
            routes = proposal_builder.routes(case_index, config.routes_per_case)
            probes = probe_factory.build_many(
                routes,
                tuple(case_state.questions.values()),
            )
            training_probes[case_index] = probes
            case_state.questions.update(
                {probe.question_id: probe for probe in probes}
            )
            records = selector_data_builder.records(probes, case_state.memory)
            attacker_records.extend(records)
            print(
                f"Route data, case {case_index}: routes={len(routes)} "
                f"valid_probes={len(probes)} selector_records={len(records)}"
            )
        if attacker_records:
            attacker_data = write_verl_dataset(
                attacker_records,
                round_dir / "attacker_data",
                seed=args.seed + round_index,
            )
            print(
                f"Round {round_index}: training Route Selector on "
                f"{attacker_data.train_size} contrast prompts from "
                f"{sum(len(items) for items in training_probes.values())} probes"
            )
            state.attacker_model = runner.train(
                "attacker",
                state.attacker_model,
                attacker_data,
                round_dir / "attacker",
            )
        else:
            print(
                f"Round {round_index}: no trainable Route Selector prompts"
            )

        audit_proposal_builder = RouteProposalBuilder(
            reader,
            seed=args.seed + 100_000 + round_index,
        )
        with VLLMPolicyServer(
            runner.verl_dir,
            state.attacker_model,
            config.policy_port,
            round_dir / "attacker_server.log",
            config.gpu_memory_utilization,
        ) as policy:
            candidates_by_case = {}
            fresh_by_case = {}
            for case_index, case_state in active_cases.items():
                max_priority = max(0, config.candidates_per_case // 2)
                high_priority = tuple(
                    case_state.questions[question_id]
                    for question_id in case_state.memory.high_priority_buffer
                    if question_id in case_state.questions
                )[:max_priority]
                fresh_count = config.candidates_per_case - len(high_priority)
                if case_state.memory.active_nodes:
                    routes = audit_proposal_builder.routes(
                        case_index,
                        max(config.routes_per_case, fresh_count * 2),
                    )
                    pool = probe_factory.build_many(
                        routes,
                        tuple(case_state.questions.values()),
                    )
                else:
                    pool = training_probes[case_index]
                case_state.questions.update(
                    {probe.question_id: probe for probe in pool}
                )
                unseen = [
                    probe
                    for probe in pool
                    if probe.question_id
                    not in case_state.memory.capability_ledger
                ]
                random.Random(
                    args.seed + 200_000 + round_index * 10_000 + case_index
                ).shuffle(unseen)
                fresh = selector.select_many(
                    policy,
                    tuple(unseen),
                    case_state.memory,
                    retriever,
                    fresh_count,
                    config.selector_candidates,
                )
                candidates_by_case[case_index] = high_priority + fresh
                fresh_by_case[case_index] = fresh
        case_rounds = {}
        builder_records = []
        for case_index, case_state in active_cases.items():
            store = MemoryStore(case_state.memory)
            flow = MemoryTrainingFlow(
                store,
                retriever,
                answer_agent,
                answer_judge,
                builder,
            )
            candidates = candidates_by_case[case_index]
            evaluations = tuple(
                (candidate, *flow.try_process_question(candidate))
                for candidate in candidates
            )
            pending = tuple(
                item
                for _, available, item in evaluations
                if available and item is not None
            )
            case_rounds[case_index] = CaseRound(
                store,
                flow,
                candidates,
                fresh_by_case[case_index],
                pending,
                sum(
                    available and item is None
                    for _, available, item in evaluations
                ),
            )
            builder_records.extend(
                memory_builder_records(pending, store.state, builder)
            )

        if builder_records:
            builder_data = write_verl_dataset(
                builder_records,
                round_dir / "memory_builder_data",
                seed=args.seed + round_index,
            )
            print(
                f"Round {round_index}: training Memory Builder on "
                f"{builder_data.train_size} prompts"
            )
            state.builder_model = runner.train(
                "memory_builder",
                state.builder_model,
                builder_data,
                round_dir / "memory_builder",
            )
            reward_model = MemoryBuilderReward(
                builder,
                answer_agent,
                retriever,
                answer_judge,
                DeepSeekMemoryJudge.from_env(),
            )
            with VLLMPolicyServer(
                runner.verl_dir,
                state.builder_model,
                config.policy_port,
                round_dir / "memory_builder_server.log",
                config.gpu_memory_utilization,
            ) as policy:
                for case_round in case_rounds.values():
                    for old_pending in case_round.pending:
                        available, current = case_round.flow.try_process_question(
                            old_pending.candidate
                        )
                        if not available:
                            case_round.flow.defer_question(old_pending.candidate)
                            continue
                        if current is None:
                            case_round.defended += 1
                            continue
                        response = policy.generate(
                            builder.build_prompt(current.observation),
                            256,
                        )
                        reward = reward_model.evaluate(
                            response,
                            case_round.flow.reward_context(current),
                        )
                        if not reward.get("reward_available", 1.0):
                            case_round.flow.defer_question(current.candidate)
                            continue
                        if (
                            not reward.get("commit_valid", 0.0)
                            or reward["score"] <= case_round.flow.commit_threshold
                        ):
                            case_round.flow.discard(current)
                            case_round.discarded += 1
                            continue
                        action = builder.parse_action(
                            response,
                            current.observation.memory_neighborhood,
                        )
                        if case_round.flow.commit(current, action, reward):
                            case_round.committed += 1
                        else:
                            case_round.discarded += 1

        compact_cases = []
        for case_index, case_round in case_rounds.items():
            evaluations = tuple(
                (candidate, *case_round.flow.try_process_question(candidate))
                for candidate in case_round.candidates
            )
            unresolved = tuple(
                item
                for _, available, item in evaluations
                if available and item is not None
            )
            for candidate, available, item in evaluations:
                if not available:
                    case_round.flow.defer_question(candidate)
                elif item is not None:
                    case_round.flow.discard(item)
            fresh_ids = {
                candidate.question_id for candidate in case_round.fresh
            }
            case_round.saturated = (
                len(case_round.fresh) >= stop_config.min_valid_questions
                and all(available for _, available, _ in evaluations)
                and not any(
                    item.candidate.question_id in fresh_ids
                    for item in unresolved
                )
                and not case_round.store.state.high_priority_buffer
            )
            if case_round.saturated and stop_config.max_neighborhoods > 0:
                # Compaction is an optional audited optimization, not evidence that
                # the repair policy has or has not converged.
                if case_round.store.state.active_nodes:
                    compact_cases.append(case_index)

        if compact_cases:
            with VLLMPolicyServer(
                runner.verl_dir,
                state.builder_model,
                config.policy_port,
                round_dir / "compaction_server.log",
                config.gpu_memory_utilization,
            ) as policy:
                for case_index in compact_cases:
                    case_state = active_cases[case_index]
                    case_round = case_rounds[case_index]
                    auditor = CompactionAuditor(
                        builder,
                        retriever,
                        answer_agent,
                        answer_judge,
                        DeepSeekMemoryJudge.from_env(),
                        stop_config,
                    )
                    audit = auditor.audit(policy, case_round.store.state)
                    case_round.store.state = audit.memory
                    case_round.compaction_attempts = audit.attempts

        for case_index, case_round in case_rounds.items():
            case_state = active_cases[case_index]
            case_state.stopped = StopCondition(
                stop_config,
                case_state.stop_state,
            ).update(case_round.saturated)
            case_round.store.advance_iteration()
            case_state.memory = case_round.store.state
            print(
                f"Round {round_index}, case {case_index}: "
                f"defended={case_round.defended}, "
                f"committed={case_round.committed}, "
                f"discarded={case_round.discarded}, "
                f"memory_nodes={len(case_state.memory.active_nodes)}, "
                f"saturated={case_round.saturated}, "
                f"compaction_attempts={case_round.compaction_attempts}"
            )

        state.next_round = round_index + 1
        state.save(state_path)
        if all(case_state.stopped for case_state in state.cases.values()):
            print("All cases have reached the stop condition")
            break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Alternating GRPO training")
    parser.add_argument(
        "--graph",
        default=ROOT / "data/longmemeval/memory_graph_fullgraph5.json",
    )
    parser.add_argument(
        "--longmemeval",
        default=ROOT / "data/longmemeval/longmemeval_s.json",
    )
    parser.add_argument("--graph-version", default="fullgraph5")
    parser.add_argument(
        "--model",
        default=os.getenv(
            "TRAIN_MODEL",
            os.getenv("ANSWER_AGENT_MODEL", "Qwen/Qwen3-0.6B"),
        ),
    )
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--routes-per-case", type=int, default=16)
    parser.add_argument("--selector-candidates", type=int, default=2)
    parser.add_argument("--candidates-per-case", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--policy-port", type=int, default=8002)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-patience", type=int, default=2)
    parser.add_argument("--stop-min-valid", type=int, default=4)
    parser.add_argument("--compaction-neighborhoods", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        rounds=args.rounds,
        routes_per_case=args.routes_per_case,
        candidates_per_case=args.candidates_per_case,
        work_dir=args.work_dir,
        policy_port=args.policy_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        selector_candidates=args.selector_candidates,
    )
    run(config, args)


if __name__ == "__main__":
    main()
