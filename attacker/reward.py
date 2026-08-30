import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

from openai import APIError

from attacker.answer_agent import QwenAnswerAgent
from attacker.gap import GapEvaluation, GapEvaluator, GapType
from attacker.models import RouteSelectorRewardContext
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.selector import RouteSelector
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class RouteSelectorRewardConfig:
    storage_gap_reward: float = 1.0
    retrieval_gap_reward: float = 0.75
    reasoning_gap_reward: float = 0.5
    no_gap_reward: float = 0.0
    exploration_weight: float = 0.1
    memory_top_k: int = 5
    correctness_threshold: float = 0.8


class RouteSelectorReward:
    """Reward route choice, not question wording."""

    def __init__(
        self,
        evaluator: GapEvaluator,
        config: RouteSelectorRewardConfig | None = None,
        trace_path: str | Path | None = None,
    ):
        self.evaluator = evaluator
        self.config = config or RouteSelectorRewardConfig()
        self.trace_path = Path(trace_path) if trace_path else None
        self._trace_lock = Lock()
        self._gap_cache: dict[tuple[str, int], GapEvaluation] = {}
        self._gap_lock = Lock()
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "RouteSelectorReward":
        config = RouteSelectorRewardConfig()
        retriever = HybridMemoryRetriever.from_env()
        evaluator = GapEvaluator(
            retriever,
            QwenAnswerAgent.from_env(),
            DeepSeekRewardJudge.from_env(),
            top_k=config.memory_top_k,
            correctness_threshold=config.correctness_threshold,
        )
        return cls(
            evaluator,
            config,
            trace_path=os.getenv("ATTACKER_REWARD_TRACE"),
        )

    def evaluate(
        self,
        response: str,
        context: RouteSelectorRewardContext,
    ) -> dict[str, float]:
        trace: dict[str, Any] = {"response": response}
        try:
            choice = RouteSelector.parse_choice(response, len(context.probes))
        except (KeyError, TypeError, ValueError) as error:
            return self._finish(
                self._result(-1.0),
                trace,
                stage="invalid_choice",
                error=str(error),
            )

        probe = context.probes[choice]
        memory = context.memory_state()
        trace.update(
            {
                "choice": choice,
                "route_id": probe.route.route_id,
                "question": probe.oracle.question,
            }
        )
        cache_key = (probe.route.route_id, context.memory_version)
        with self._gap_lock:
            evaluation = self._gap_cache.get(cache_key)
        if evaluation is None:
            try:
                evaluation = self.evaluator.evaluate(probe, memory)
            except (APIError, StructuredOutputError) as error:
                return self._finish(
                    self._result(0.0, reward_available=0.0, choice_valid=1.0),
                    trace,
                    stage="judge_unavailable",
                    error=str(error),
                )
            with self._gap_lock:
                self._gap_cache[cache_key] = evaluation

        history = memory.attack_history.get(probe.route.route_id)
        attempts = history.attempts if history else 0
        exploration = 1.0 / math.sqrt(1.0 + attempts)
        storage_value = (
            context.storage_values[choice]
            if len(context.storage_values) == len(context.probes)
            else 1.0
        )
        gap_reward = {
            GapType.STORAGE: self.config.storage_gap_reward * storage_value,
            GapType.RETRIEVAL: self.config.retrieval_gap_reward,
            GapType.REASONING: self.config.reasoning_gap_reward,
            GapType.NONE: self.config.no_gap_reward,
        }[evaluation.gap_type]
        score = (
            (1.0 - self.config.exploration_weight) * gap_reward
            + self.config.exploration_weight * exploration
        )
        result = self._result(
            score,
            choice=float(choice),
            choice_valid=1.0,
            gap_found=float(evaluation.gap_type != GapType.NONE),
            storage_gap=float(evaluation.gap_type == GapType.STORAGE),
            retrieval_gap=float(evaluation.gap_type == GapType.RETRIEVAL),
            reasoning_gap=float(evaluation.gap_type == GapType.REASONING),
            memory_correctness=evaluation.correctness,
            structural_coverage=evaluation.structural_coverage,
            retrieved_coverage=evaluation.retrieved_coverage,
            exploration=exploration,
            storage_value=storage_value,
        )
        return self._finish(
            result,
            trace,
            stage=evaluation.gap_type.value,
            memory_answer=evaluation.memory_answer,
            storage_value=storage_value,
        )

    def evaluate_batch(
        self,
        responses: list[str],
        contexts: list[RouteSelectorRewardContext],
        group_ids: list[str] | None = None,
    ) -> list[dict[str, float]]:
        results = [
            self.evaluate(response, context)
            for response, context in zip(responses, contexts, strict=True)
        ]
        groups: dict[str, list[dict[str, float]]] = {}
        if group_ids is not None:
            for group_id, result in zip(group_ids, results, strict=True):
                groups.setdefault(str(group_id), []).append(result)
        fields = {
            "samples": len(results),
            "groups": len(groups),
            "informative_groups": sum(
                len({item["score"] for item in items}) > 1
                for items in groups.values()
            ),
            "unique_choices": sum(
                len({item["choice"] for item in items if item["choice_valid"]})
                for items in groups.values()
            ),
            "positive": sum(item["score"] > 0 for item in results),
            "unavailable": sum(not item["reward_available"] for item in results),
            "storage": sum(item["storage_gap"] > 0 for item in results),
            "retrieval": sum(item["retrieval_gap"] > 0 for item in results),
            "reasoning": sum(item["reasoning_gap"] > 0 for item in results),
            "storage_value": round(
                sum(item["storage_value"] for item in results)
                / max(1, len(results)),
                3,
            ),
        }
        print(
            "Route Selector Reward: "
            + " ".join(f"{key}={value}" for key, value in fields.items()),
            flush=True,
        )
        return results

    def _finish(
        self,
        result: dict[str, float],
        trace: dict[str, Any],
        **details: Any,
    ) -> dict[str, float]:
        if self.trace_path:
            record = {**trace, **details, "reward": result}
            with self._trace_lock, self.trace_path.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return result

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "choice": -1.0,
            "choice_valid": 0.0,
            "gap_found": 0.0,
            "storage_gap": 0.0,
            "retrieval_gap": 0.0,
            "reasoning_gap": 0.0,
            "memory_correctness": 0.0,
            "structural_coverage": 0.0,
            "retrieved_coverage": 0.0,
            "exploration": 0.0,
            "storage_value": 0.0,
        }
        result.update(values)
        result["format_valid"] = result["choice_valid"]
        return result


# Compatibility names for existing scripts and external imports.
AttackerRewardConfig = RouteSelectorRewardConfig
AttackerReward = RouteSelectorReward
