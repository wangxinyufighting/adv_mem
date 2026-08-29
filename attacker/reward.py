import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

from attacker.answer_agent import QwenAnswerAgent, is_insufficient_answer
from attacker.attacker import Attacker
from attacker.models import (
    AttackerRewardContext,
    GraphRouteBundle,
    OracleResult,
)
from attacker.oracle import DeepSeekOracle
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.validation import (
    answer_is_leaked,
    has_terminal_question_mark,
    normalize_question,
    question_constraint_error,
    route_fidelity,
)
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class AttackerRewardConfig:
    gold_threshold: float = 0.8
    parametric_threshold: float = 0.8
    memory_threshold: float = 0.8
    value_threshold: float = 0.5
    novelty_threshold: float = 0.2
    route_threshold: float = 0.8
    memory_top_k: int = 5
    novelty_low: float = 0.75
    novelty_high: float = 0.92
    uncovered_weight: float = 0.4
    value_weight: float = 0.25
    novelty_weight: float = 0.2
    route_weight: float = 0.15
    question_mark_penalty: float = 0.05


@dataclass(frozen=True)
class _PreparedAttack:
    question: str
    context: AttackerRewardContext
    trace: dict[str, Any]
    question_mark: bool


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
        trace_path: str | Path | None = None,
    ):
        self.oracle = oracle
        self.answer_agent = answer_agent
        self.retriever = retriever
        self.judge = judge
        self.embedder = embedder
        self.config = config or AttackerRewardConfig()
        self.trace_path = Path(trace_path) if trace_path else None
        self._trace_lock = Lock()
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "AttackerReward":
        retriever = HybridMemoryRetriever.from_env()
        return cls(
            oracle=DeepSeekOracle(),
            answer_agent=QwenAnswerAgent.from_env(),
            retriever=retriever,
            judge=DeepSeekRewardJudge.from_env(),
            embedder=retriever.embedder,
            trace_path=os.getenv("ATTACKER_REWARD_TRACE"),
        )

    def evaluate(
        self,
        response: str,
        context: AttackerRewardContext,
    ) -> dict[str, float]:
        prepared = self._prepare(response, context)
        if isinstance(prepared, dict):
            return prepared
        try:
            oracle = self.oracle.evaluate(prepared.question, context.route)
        except StructuredOutputError as error:
            return self._oracle_unavailable(prepared, error)
        return self._score(prepared, oracle)

    def evaluate_batch(
        self,
        responses: list[str],
        contexts: list[AttackerRewardContext],
        group_ids: list[str],
    ) -> list[dict[str, float]]:
        results: list[dict[str, float] | None] = [None] * len(responses)
        groups: dict[tuple[str, str], list[tuple[int, _PreparedAttack]]] = {}
        for index, (response, context, group_id) in enumerate(
            zip(responses, contexts, group_ids, strict=True)
        ):
            prepared = self._prepare(response, context)
            if isinstance(prepared, dict):
                results[index] = prepared
                continue
            key = (str(group_id), context.route.route_id)
            groups.setdefault(key, []).append((index, prepared))

        for items in groups.values():
            route = items[0][1].context.route
            try:
                oracle_results = self.oracle.evaluate_many(
                    route,
                    tuple((index, item.question) for index, item in items),
                )
            except StructuredOutputError as error:
                for index, item in items:
                    results[index] = self._oracle_unavailable(item, error)
                continue
            for index, item in items:
                results[index] = self._score(item, oracle_results[index])
        return cast(list[dict[str, float]], results)

    def _prepare(
        self,
        response: str,
        context: AttackerRewardContext,
    ) -> _PreparedAttack | dict[str, float]:
        trace = {
            "route_id": context.route.route_id,
            "attack_mode": context.route.attack_mode.value,
            "response": response,
        }
        try:
            question = Attacker.parse_question(response)
        except ValueError:
            return self._finish(
                self._result(-1.0),
                trace,
                stage="schema_invalid",
            )
        question_mark = has_terminal_question_mark(question)
        trace["question"] = question
        trace["question_mark"] = question_mark
        constraint_error = question_constraint_error(question, context.route)
        if constraint_error:
            return self._finish(
                self._result(
                    -0.9,
                    schema_valid=1.0,
                    question_mark=float(question_mark),
                ),
                trace,
                stage=constraint_error,
            )

        normalized = normalize_question(question)
        if normalized in {
            normalize_question(item.question)
            for item in context.prior_questions
        }:
            return self._finish(
                self._result(
                    -1.0,
                    schema_valid=1.0,
                    question_valid=1.0,
                    question_mark=float(question_mark),
                ),
                trace,
                stage="duplicate",
            )
        return _PreparedAttack(question, context, trace, question_mark)

    def _oracle_unavailable(
        self,
        prepared: _PreparedAttack,
        error: StructuredOutputError,
    ) -> dict[str, float]:
        return self._finish(
            self._result(
                0.0,
                reward_available=0.0,
                **self._base_metrics(prepared),
            ),
            prepared.trace,
            stage="oracle_unavailable",
            error=str(error),
        )

    def _score(
        self,
        prepared: _PreparedAttack,
        oracle: OracleResult,
    ) -> dict[str, float]:
        question = prepared.question
        context = prepared.context
        trace = prepared.trace
        base_metrics = self._base_metrics(prepared)
        if not oracle.valid:
            relevance = self._route_relevance(question, context.route)
            return self._complete(
                self._result(
                    -0.8 + 0.4 * relevance,
                    **base_metrics,
                ),
                prepared,
                trace,
                stage="oracle_invalid",
                oracle_invalid_reason=oracle.invalid_reason,
                route_relevance=relevance,
            )

        trace.update(
            {
                "oracle_answer": oracle.answer,
                "supporting_source_ids": [
                    item.source_id for item in oracle.supporting_evidence
                ],
            }
        )
        fidelity = route_fidelity(context.route, oracle)
        hard_metrics = {
            **base_metrics,
            "oracle_valid": 1.0,
            "route_fidelity": fidelity,
            "route_pass": float(fidelity >= self.config.route_threshold),
        }
        if answer_is_leaked(question, oracle.answer):
            return self._complete(
                self._result(-1.0, **hard_metrics),
                prepared,
                trace,
                stage="answer_leak",
            )
        if fidelity < self.config.route_threshold:
            return self._complete(
                self._result(-1.0, **hard_metrics),
                prepared,
                trace,
                stage="route_unfaithful",
            )

        # Golden corpus is copied verbatim from the Full Memory Graph.
        golden_answer = self.answer_agent.answer_sources(
            question,
            context.route.source_records,
        )
        trace["golden_answer"] = golden_answer
        if is_insufficient_answer(golden_answer):
            return self._complete(
                self._result(-1.0, **hard_metrics),
                prepared,
                trace,
                stage="gold_insufficient",
            )
        parametric_answer = self.answer_agent.answer_question(question)
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
        trace.update(
            {
                "parametric_answer": parametric_answer,
                "memory_answer": memory_answer,
            }
        )
        try:
            judged = self.judge.evaluate(
                oracle,
                golden_answer,
                memory_answer,
                parametric_answer,
            )
        except StructuredOutputError as error:
            return self._complete(
                self._result(
                    0.0,
                    reward_available=0.0,
                    **hard_metrics,
                ),
                prepared,
                trace,
                stage="judge_unavailable",
                error=str(error),
            )

        gold_correctness = (
            0.0 if is_insufficient_answer(golden_answer) else judged.gold_correctness
        )
        memory_correctness = (
            0.0 if is_insufficient_answer(memory_answer) else judged.memory_correctness
        )
        parametric_correctness = (
            0.0
            if is_insufficient_answer(parametric_answer)
            else judged.parametric_correctness
        )
        uncovered = 1.0 - memory_correctness
        novelty = self._novelty(question, oracle, context)
        gold_pass = gold_correctness >= self.config.gold_threshold
        parametric_pass = parametric_correctness < self.config.parametric_threshold
        uncovered_pass = memory_correctness < self.config.memory_threshold
        value_pass = judged.value >= self.config.value_threshold
        novelty_pass = novelty >= self.config.novelty_threshold
        metrics = {
            **base_metrics,
            "oracle_valid": 1.0,
            "gold_correctness": gold_correctness,
            "gold_pass": float(gold_pass),
            "memory_correctness": memory_correctness,
            "uncovered": uncovered,
            "uncovered_pass": float(uncovered_pass),
            "parametric_correctness": parametric_correctness,
            "parametric_pass": float(parametric_pass),
            "value": judged.value,
            "value_pass": float(value_pass),
            "novelty": novelty,
            "novelty_pass": float(novelty_pass),
            "route_fidelity": fidelity,
            "route_pass": 1.0,
        }
        failures = []
        if not gold_pass:
            failures.append(
                (
                    "gold_invalid",
                    gold_correctness / self.config.gold_threshold - 1.0,
                )
            )
        if not parametric_pass:
            failures.append(("parametric_answerable", -parametric_correctness))
        if not uncovered_pass:
            failures.append(("memory_answerable", -memory_correctness))
        if not value_pass:
            failures.append(
                ("low_value", judged.value / self.config.value_threshold - 1.0)
            )
        if not novelty_pass:
            failures.append(
                ("redundant", novelty / self.config.novelty_threshold - 1.0)
            )
        if failures:
            stage, score = min(failures, key=lambda item: item[1])
            return self._complete(
                self._result(score, **metrics),
                prepared,
                trace,
                stage=stage,
            )

        score = min(
            1.0,
            self.config.uncovered_weight * uncovered
            + self.config.value_weight * judged.value
            + self.config.novelty_weight * novelty
            + self.config.route_weight * fidelity,
        )
        return self._complete(
            self._result(score, attack_valid=1.0, **metrics),
            prepared,
            trace,
            stage="scored",
        )

    @staticmethod
    def _base_metrics(prepared: _PreparedAttack) -> dict[str, float]:
        return {
            "schema_valid": 1.0,
            "question_valid": 1.0,
            "question_mark": float(prepared.question_mark),
        }

    def _complete(
        self,
        result: dict[str, float],
        prepared: _PreparedAttack,
        trace: dict[str, Any],
        **details: Any,
    ) -> dict[str, float]:
        if not prepared.question_mark and result["reward_available"]:
            result["score"] = max(
                -1.0,
                result["score"] - self.config.question_mark_penalty,
            )
        return self._finish(result, trace, **details)

    def _finish(
        self,
        result: dict[str, float],
        trace: dict[str, Any],
        **details: Any,
    ) -> dict[str, float]:
        if self.trace_path:
            record = {**trace, **details, "reward": result}
            line = json.dumps(record, ensure_ascii=False)
            with self._trace_lock, self.trace_path.open("a") as stream:
                stream.write(line + "\n")
        return result

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

    def _route_relevance(self, question: str, route: GraphRouteBundle) -> float:
        vectors = self.embedder.embed(
            [question, *(node.memory for node in route.evidence_nodes)]
        )
        similarity = max(self._cosine(vectors[0], vector) for vector in vectors[1:])
        return max(0.0, min(1.0, similarity))

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=True)) / (
            math.sqrt(sum(value * value for value in left))
            * math.sqrt(sum(value * value for value in right))
        )

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "schema_valid": 0.0,
            "question_valid": 0.0,
            "question_mark": 0.0,
            "oracle_valid": 0.0,
            "gold_correctness": 0.0,
            "gold_pass": 0.0,
            "memory_correctness": 0.0,
            "uncovered": 0.0,
            "uncovered_pass": 0.0,
            "parametric_correctness": 0.0,
            "parametric_pass": 0.0,
            "value": 0.0,
            "value_pass": 0.0,
            "novelty": 0.0,
            "novelty_pass": 0.0,
            "route_fidelity": 0.0,
            "route_pass": 0.0,
            "attack_valid": 0.0,
        }
        result.update(values)
        result["format_valid"] = result["schema_valid"]
        return result
