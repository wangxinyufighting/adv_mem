import json
import math
from dataclasses import dataclass
from typing import Any

from attacker.answer_agent import QwenAnswerAgent
from attacker.models import (
    AttackMode,
    AttackerRewardContext,
    GraphRouteBundle,
    OracleResult,
)
from attacker.oracle import DeepSeekOracle
from attacker.reward_judge import DeepSeekRewardJudge
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class AttackerRewardConfig:
    gold_threshold: float = 0.8
    memory_top_k: int = 5
    novelty_low: float = 0.75
    novelty_high: float = 0.92


class AttackerReward:
    """Outcome reward for one attacker rollout."""

    def __init__(
        self,
        oracle: Any,
        answer_agent: Any,
        retriever: Any,
        judge: Any,
        embedder: Any,
        config: AttackerRewardConfig | None = None,
    ):
        self.oracle = oracle
        self.answer_agent = answer_agent
        self.retriever = retriever
        self.judge = judge
        self.embedder = embedder
        self.config = config or AttackerRewardConfig()

    @classmethod
    def from_env(cls) -> "AttackerReward":
        retriever = HybridMemoryRetriever.from_env()
        return cls(
            oracle=DeepSeekOracle(),
            answer_agent=QwenAnswerAgent.from_env(),
            retriever=retriever,
            judge=DeepSeekRewardJudge.from_env(),
            embedder=retriever.embedder,
        )

    def evaluate(
        self,
        response: str,
        context: AttackerRewardContext,
    ) -> dict[str, float]:
        try:
            question = json.loads(response)["question"].strip()
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
            return self._invalid(format_valid=0.0)

        if not question:
            return self._invalid(format_valid=0.0)

        oracle = self.oracle.evaluate(question, context.route)
        if not oracle.valid:
            return self._invalid(format_valid=1.0)

        # Golden corpus is copied verbatim from the Full Memory Graph.
        golden_answer = self.answer_agent.answer_sources(
            question,
            context.route.source_records,
        )
        # The same frozen agent now answers from fixed retrieval over M_t.
        memory_results = self.retriever.retrieve(
            question,
            context.memory_state(),
            top_k=self.config.memory_top_k,
        )
        memory_answer = self.answer_agent.answer_memories(
            question,
            tuple(result.node for result in memory_results),
        )
        judged = self.judge.evaluate(oracle, golden_answer, memory_answer)

        if judged.gold_correctness < self.config.gold_threshold:
            return {
                **self._invalid(format_valid=1.0, oracle_valid=1.0),
                "score": judged.gold_correctness / self.config.gold_threshold - 1.0,
                "gold_correctness": judged.gold_correctness,
            }

        uncovered = max(
            0.0,
            judged.gold_correctness - judged.memory_correctness,
        )
        novelty = self._novelty(question, oracle, context)
        fidelity = self._route_fidelity(context.route, oracle)
        score = (judged.value * uncovered * novelty * fidelity) ** 0.25
        return {
            "score": score,
            "format_valid": 1.0,
            "oracle_valid": 1.0,
            "gold_correctness": judged.gold_correctness,
            "memory_correctness": judged.memory_correctness,
            "value": judged.value,
            "uncovered": uncovered,
            "novelty": novelty,
            "route_fidelity": fidelity,
        }

    def _novelty(
        self,
        question: str,
        oracle: OracleResult,
        context: AttackerRewardContext,
    ) -> float:
        if not context.prior_questions:
            return 1.0

        current = f"{question}\nAnswer: {oracle.answer}"
        previous = [
            f"{item.question}\nAnswer: {item.answer}"
            for item in context.prior_questions
        ]
        vectors = self.embedder.embed([current, *previous])
        similarity = max(self._cosine(vectors[0], vector) for vector in vectors[1:])
        return max(
            0.0,
            min(
                1.0,
                (self.config.novelty_high - similarity)
                / (self.config.novelty_high - self.config.novelty_low),
            ),
        )

    @staticmethod
    def _route_fidelity(route: GraphRouteBundle, oracle: OracleResult) -> float:
        used = {item.node_id for item in oracle.supporting_evidence}
        intended = {node.id for node in route.evidence_nodes}
        required = 1 if route.attack_mode == AttackMode.SINGLE_FACT else 2
        return min(1.0, len(used & intended) / required)

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            math.sqrt(sum(value * value for value in left))
            * math.sqrt(sum(value * value for value in right))
        )

    @staticmethod
    def _invalid(
        format_valid: float,
        oracle_valid: float = 0.0,
    ) -> dict[str, float]:
        return {
            "score": -1.0,
            "format_valid": format_valid,
            "oracle_valid": oracle_valid,
            "gold_correctness": 0.0,
            "memory_correctness": 0.0,
            "value": 0.0,
            "uncovered": 0.0,
            "novelty": 0.0,
            "route_fidelity": 0.0,
        }
