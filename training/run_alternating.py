import argparse
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path

from attacker.answer_agent import QwenAnswerAgent
from attacker.attacker import Attacker
from attacker.models import GraphRouteBundle
from attacker.oracle import DeepSeekOracle
from attacker.reward_judge import DeepSeekRewardJudge
from defender.memory_builder import MemoryBuilder
from defender.reward import MemoryBuilderReward
from defender.reward_judge import DeepSeekMemoryJudge
from memory.models import MemoryState
from memory.store import MemoryStore
from training.alternating import MemoryTrainingFlow, QuestionCandidate
from training.dataset_builder import (
    AttackerDatasetBuilder,
    memory_builder_records,
    write_verl_dataset,
)
from training.policy_server import ChatPolicy, VLLMPolicyServer
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
    candidates_per_round: int
    work_dir: Path
    policy_port: int
    gpu_memory_utilization: float


class QuestionCollector:
    """Use the trained Attacker, then apply Oracle and golden-corpus validation."""

    def __init__(
        self,
        attacker: Attacker,
        oracle: DeepSeekOracle,
        answer_agent: QwenAnswerAgent,
        judge: DeepSeekRewardJudge,
        value_threshold: float = 0.5,
    ):
        self.attacker = attacker
        self.oracle = oracle
        self.answer_agent = answer_agent
        self.judge = judge
        self.value_threshold = value_threshold

    def collect(
        self,
        policy: ChatPolicy,
        routes: tuple[GraphRouteBundle, ...],
        memory: MemoryState,
        retriever: HybridMemoryRetriever,
        count: int,
        prior_questions: set[str],
    ) -> tuple[QuestionCandidate, ...]:
        if count == 0:
            return ()
        candidates = []
        for route in routes:
            observation = self.attacker.observe(route, memory, retriever)
            response = policy.generate(self.attacker.build_prompt(observation), 256)
            try:
                question = self.attacker.parse_question(response)
            except (ValueError, KeyError, TypeError):
                continue
            if not question or question in prior_questions:
                continue

            oracle = self.oracle.evaluate(question, route)
            if not oracle.valid:
                continue
            golden_answer = self.answer_agent.answer_sources(
                question,
                route.source_records,
            )
            judged = self.judge.evaluate(
                oracle,
                golden_answer,
                "INSUFFICIENT_INFORMATION",
            )
            if (
                judged.gold_correctness < 0.8
                or judged.value < self.value_threshold
            ):
                continue

            question_id = hashlib.sha256(
                f"{route.route_id}\n{question}".encode()
            ).hexdigest()[:20]
            candidates.append(
                QuestionCandidate(question_id, route, oracle, golden_answer)
            )
            prior_questions.add(question)
            if len(candidates) == count:
                break
        return tuple(candidates)


def run(config: RunConfig, args: argparse.Namespace) -> None:
    state_path = config.work_dir / "run_state.json"
    state = RunState.load(state_path, args.model)
    reader = LongMemEvalGraphReader(args.graph, args.longmemeval, args.graph_version)
    retriever = HybridMemoryRetriever.from_env()
    attacker = Attacker()
    builder = MemoryBuilder()
    answer_agent = QwenAnswerAgent.from_env()
    answer_judge = DeepSeekRewardJudge.from_env()
    oracle = DeepSeekOracle()
    collector = QuestionCollector(attacker, oracle, answer_agent, answer_judge)
    stop_config = StopConfig(
        patience=args.stop_patience,
        min_valid_questions=args.stop_min_valid,
        max_neighborhoods=args.compaction_neighborhoods,
    )
    stop_condition = StopCondition(stop_config, state.stop_state)
    runner = VerlRunner(
        ROOT,
        VerlConfig(args.epochs, args.batch_size, args.gpus),
    )

    for round_index in range(state.next_round, state.next_round + config.rounds):
        round_dir = config.work_dir / f"round_{round_index:03d}"
        route_builder = AttackerDatasetBuilder(
            reader,
            retriever,
            attacker,
            seed=args.seed + round_index,
        )
        attacker_records = route_builder.records(
            state.memory,
            config.routes_per_case,
        )
        attacker_data = write_verl_dataset(
            attacker_records,
            round_dir / "attacker_data",
            seed=args.seed + round_index,
        )
        print(f"Round {round_index}: training Attacker on {attacker_data.train_size} prompts")
        state.attacker_model = runner.train(
            "attacker",
            state.attacker_model,
            attacker_data,
            round_dir / "attacker",
        )

        high_priority = [
            state.questions[question_id]
            for question_id in state.memory.high_priority_buffer
            if question_id in state.questions
        ][: config.candidates_per_round]
        fresh_count = config.candidates_per_round - len(high_priority)
        attempts = max(fresh_count * 5, fresh_count)
        route_count = max(1, math.ceil(attempts / len(reader)))
        audit_route_builder = AttackerDatasetBuilder(
            reader,
            retriever,
            attacker,
            seed=args.seed + 100_000 + round_index,
        )
        routes = audit_route_builder.routes(route_count)
        with VLLMPolicyServer(
            runner.verl_dir,
            state.attacker_model,
            config.policy_port,
            round_dir / "attacker_server.log",
            config.gpu_memory_utilization,
        ) as policy:
            fresh = collector.collect(
                policy,
                routes,
                state.memory,
                retriever,
                fresh_count,
                {item.oracle.question for item in state.questions.values()},
            )
        candidates = tuple(high_priority) + fresh
        for item in fresh:
            state.questions[item.question_id] = item

        store = MemoryStore(state.memory)
        flow = MemoryTrainingFlow(
            store,
            retriever,
            answer_agent,
            answer_judge,
            builder,
        )
        pending = tuple(
            item
            for candidate in candidates
            if (item := flow.process_question(candidate)) is not None
        )
        defended = len(candidates) - len(pending)
        committed = 0
        discarded = 0

        if pending:
            records = memory_builder_records(pending, store.state, builder)
            builder_data = write_verl_dataset(
                records,
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
                DeepSeekMemoryJudge.from_env(),
            )
            with VLLMPolicyServer(
                runner.verl_dir,
                state.builder_model,
                config.policy_port,
                round_dir / "memory_builder_server.log",
                config.gpu_memory_utilization,
            ) as policy:
                for old_pending in pending:
                    current = flow.process_question(old_pending.candidate)
                    if current is None:
                        defended += 1
                        continue
                    response = policy.generate(
                        builder.build_prompt(current.observation),
                        512,
                    )
                    reward = reward_model.evaluate(
                        response,
                        flow.reward_context(current),
                    )
                    if reward["score"] <= flow.commit_threshold:
                        flow.discard(current)
                        discarded += 1
                        continue
                    action = builder.parse_action(response)
                    if flow.commit(current, action, reward):
                        committed += 1
                    else:
                        discarded += 1

        unresolved = tuple(
            item
            for candidate in candidates
            if (item := flow.process_question(candidate)) is not None
        )
        for item in unresolved:
            flow.discard(item)
        fresh_ids = {candidate.question_id for candidate in fresh}
        attacker_saturated = (
            len(fresh) >= stop_config.min_valid_questions
            and not any(item.candidate.question_id in fresh_ids for item in unresolved)
            and not store.state.high_priority_buffer
        )
        builder_saturated = False
        compaction_attempts = 0
        if attacker_saturated and store.state.active_nodes:
            auditor = CompactionAuditor(
                builder,
                retriever,
                answer_agent,
                answer_judge,
                state.questions,
                stop_config,
            )
            with VLLMPolicyServer(
                runner.verl_dir,
                state.builder_model,
                config.policy_port,
                round_dir / "compaction_server.log",
                config.gpu_memory_utilization,
            ) as policy:
                audit = auditor.audit(policy, store.state)
            store.state = audit.memory
            builder_saturated = not audit.compressed
            compaction_attempts = audit.attempts
        elif attacker_saturated:
            builder_saturated = True

        stopped = stop_condition.update(attacker_saturated, builder_saturated)

        store.advance_iteration()
        state.memory = store.state
        state.next_round = round_index + 1
        state.save(state_path)
        print(
            f"Round {round_index} complete: defended={defended}, "
            f"committed={committed}, discarded={discarded}, "
            f"memory_nodes={len(state.memory.active_nodes)}, "
            f"attack_saturated={attacker_saturated}, "
            f"builder_saturated={builder_saturated}, "
            f"compaction_attempts={compaction_attempts}"
        )
        if stopped:
            print("Stop condition reached")
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
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--work-dir", type=Path, default=ROOT / "data/training")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--routes-per-case", type=int, default=16)
    parser.add_argument("--candidates-per-round", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--policy-port", type=int, default=8002)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stop-patience", type=int, default=2)
    parser.add_argument("--stop-min-valid", type=int, default=4)
    parser.add_argument("--compaction-neighborhoods", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        rounds=args.rounds,
        routes_per_case=args.routes_per_case,
        candidates_per_round=args.candidates_per_round,
        work_dir=args.work_dir,
        policy_port=args.policy_port,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    run(config, args)


if __name__ == "__main__":
    main()
