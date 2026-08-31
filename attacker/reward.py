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
from attacker.models import GraphRouteBundle, RouteProbe, RouteSelectorRewardContext
from attacker.oracle import DeepSeekOracle
from attacker.probe import FixedProbeQuestionGenerator, ProbeFactory
from attacker.probe_cache import ProbeCache
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.selector import RouteSelector
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class RouteSelectorRewardConfig:
    novelty_weight: float = 0.1
    repeat_weight: float = 0.1
    memory_top_k: int = 5
    correctness_threshold: float = 0.8


class RouteSelectorReward:
    """R = memory failure + evidence novelty - repeated selection."""

    def __init__(
        self,
        evaluator: GapEvaluator,
        config: RouteSelectorRewardConfig | None = None,
        trace_path: str | Path | None = None,
        probe_factory: ProbeFactory | None = None,
        probe_cache_path: str | Path | None = None,
    ):
        self.evaluator = evaluator
        self.config = config or RouteSelectorRewardConfig()
        self.trace_path = Path(trace_path) if trace_path else None
        self.probe_factory = probe_factory
        self.probe_cache_path = probe_cache_path
        self._trace_lock = Lock()
        self._gap_cache: dict[tuple[str, int], GapEvaluation] = {}
        self._probe_cache: dict[str, RouteProbe | None] = {}
        self._persistent_caches: dict[str, ProbeCache] = {}
        self._gap_lock = Lock()
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "RouteSelectorReward":
        config = RouteSelectorRewardConfig()
        answer_agent = QwenAnswerAgent.from_env()
        judge = DeepSeekRewardJudge.from_env()
        evaluator = GapEvaluator(
            HybridMemoryRetriever.from_env(),
            answer_agent,
            judge,
            top_k=config.memory_top_k,
            correctness_threshold=config.correctness_threshold,
        )
        factory = ProbeFactory(
            FixedProbeQuestionGenerator.from_env(),
            DeepSeekOracle(),
            answer_agent,
            judge,
        )
        return cls(
            evaluator,
            config,
            os.getenv("ATTACKER_REWARD_TRACE"),
            factory,
            os.getenv("ATTACKER_PROBE_CACHE"),
        )

    def evaluate(
        self,
        response: str,
        context: RouteSelectorRewardContext,
    ) -> dict[str, float]:
        trace: dict[str, Any] = {"response": response}
        try:
            choice = RouteSelector.parse_choice(response, len(context.routes))
        except (KeyError, TypeError, ValueError) as error:
            return self._finish(
                self._result(-1.0), trace, stage="invalid_choice", error=str(error)
            )

        route = context.routes[choice]
        probe = self._probe(route, context)
        if probe is None:
            return self._finish(
                self._result(0.0, reward_available=0.0, choice_valid=1.0),
                trace,
                stage="probe_unavailable",
                route_id=route.route_id,
            )
        memory = context.memory_state()
        trace.update(
            choice=choice,
            route_id=route.route_id,
            question=probe.oracle.question,
        )
        cache_key = (route.route_id, context.memory_version)
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

        history = memory.attack_history.get(route.route_id)
        attempts = history.attempts if history else 0
        novelty = (
            context.novelty_values[choice]
            if len(context.novelty_values) == len(context.routes)
            else 1.0 - evaluation.structural_coverage
        )
        repeat = 1.0 - 1.0 / math.sqrt(1.0 + attempts)
        failure = 1.0 - evaluation.correctness
        score = (
            failure
            + self.config.novelty_weight * novelty
            - self.config.repeat_weight * repeat
        )
        result = self._result(
            score,
            choice=float(choice),
            choice_valid=1.0,
            gap_found=float(evaluation.gap_type != GapType.NONE),
            memory_correctness=evaluation.correctness,
            structural_coverage=evaluation.structural_coverage,
            retrieved_coverage=evaluation.retrieved_coverage,
            failure=failure,
            novelty=novelty,
            repeat=repeat,
        )
        return self._finish(
            result,
            trace,
            stage=evaluation.gap_type.value,
            memory_answer=evaluation.memory_answer,
        )

    def _probe(
        self,
        route: GraphRouteBundle,
        context: RouteSelectorRewardContext,
    ) -> RouteProbe | None:
        if route.route_id in self._probe_cache:
            return self._probe_cache[route.route_id]
        cached = next(
            (
                probe
                for probe in context.cached_probes
                if probe.route.route_id == route.route_id
            ),
            None,
        )
        persistent = None
        if self.probe_cache_path:
            persistent = self._persistent_caches.get(route.graph_version)
            if persistent is None:
                persistent = ProbeCache(route.graph_version, self.probe_cache_path)
                self._persistent_caches[route.graph_version] = persistent
        probe = cached or (persistent.get(route) if persistent else None)
        if probe is None and self.probe_factory:
            probe = self.probe_factory.build(route)
            if probe is not None and persistent:
                persistent.put(probe)
        self._probe_cache[route.route_id] = probe
        return probe

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
        for group_id, result in zip(group_ids or (), results):
            groups.setdefault(str(group_id), []).append(result)
        informative = sum(
            len({item["score"] for item in group}) > 1
            for group in groups.values()
        )
        choices = sum(
            len({item["choice"] for item in group if item["choice_valid"]})
            for group in groups.values()
        )
        print(
            "Route Selector Reward: "
            f"samples={len(results)} groups={len(groups)} "
            f"informative_groups={informative} unique_choices={choices} "
            f"unavailable={sum(not item['reward_available'] for item in results)}",
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
            with self._trace_lock, self.trace_path.open("a") as stream:
                stream.write(
                    json.dumps({**trace, **details, "reward": result}, ensure_ascii=False)
                    + "\n"
                )
        return result

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "choice": -1.0,
            "choice_valid": 0.0,
            "gap_found": 0.0,
            "memory_correctness": 0.0,
            "structural_coverage": 0.0,
            "retrieved_coverage": 0.0,
            "failure": 0.0,
            "novelty": 0.0,
            "repeat": 0.0,
        }
        result.update(values)
        result["format_valid"] = result["choice_valid"]
        return result


AttackerRewardConfig = RouteSelectorRewardConfig
AttackerReward = RouteSelectorReward
