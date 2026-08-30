import hashlib
import json
from typing import Any

from attacker.models import (
    RouteCandidateObservation,
    RouteProbe,
    RouteSelectorObservation,
    RouteSelectorRewardContext,
)
from memory.models import MemoryState, RouteAttackStats
from utils.json_output import clean_model_output, parse_json_object


SYSTEM_PROMPT = """Select the one candidate route most likely to expose a real gap in
the current long-term memory.

Prefer missing substantive user facts, state changes, decisions, preferences, and
prior assistant responses. A useful gap may be absent from memory, hard to retrieve,
or present but difficult to answer from. Use target, probe_question, known memory,
and history together. Do not repeatedly select a route that has already produced no
gap unless the memory version or known evidence has materially changed.

Return exactly one JSON object and nothing else: {"choice": 0}. The choice must be
one of the integer candidate choices shown in the input."""


class RouteSelector:
    """Learned policy that chooses where to attack before wording is considered."""

    def __init__(self, neighborhood_size: int = 5):
        self.neighborhood_size = neighborhood_size

    def observe(
        self,
        probes: tuple[RouteProbe, ...],
        memory: MemoryState,
        retriever: Any,
    ) -> RouteSelectorObservation:
        candidates = []
        for choice, probe in enumerate(probes):
            results = retriever.retrieve(
                probe.oracle.question,
                memory,
                top_k=self.neighborhood_size,
            )
            history = memory.attack_history.get(
                probe.route.route_id,
                RouteAttackStats(probe.route.route_id),
            )
            candidates.append(
                RouteCandidateObservation(
                    choice=choice,
                    probe=probe,
                    memory_neighborhood=tuple(result.node for result in results),
                    history=history,
                )
            )
        return RouteSelectorObservation(memory.version, tuple(candidates))

    @staticmethod
    def build_prompt(
        observation: RouteSelectorObservation,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation.to_dict(), ensure_ascii=False)
                + "\n\n/no_think",
            },
        ]

    @staticmethod
    def parse_choice(response: str, candidate_count: int) -> int:
        payload = parse_json_object(clean_model_output(response), ("choice",))
        if set(payload) != {"choice"}:
            raise ValueError("Route selector returned unexpected fields")
        choice = payload["choice"]
        if type(choice) is not int or not 0 <= choice < candidate_count:
            raise ValueError("Route selector choice is out of range")
        return choice

    def to_verl_record(
        self,
        observation: RouteSelectorObservation,
        probes: tuple[RouteProbe, ...],
        memory: MemoryState,
    ) -> dict[str, Any]:
        signature = "|".join(probe.route.route_id for probe in probes)
        group_id = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:20]
        context = RouteSelectorRewardContext.from_state(probes, memory)
        return {
            "data_source": "route_selector",
            "prompt": self.build_prompt(observation),
            "ability": "memory_route_selection",
            "reward_model": {
                "style": "rule",
                "ground_truth": group_id,
            },
            "extra_info": context.to_dict(),
        }

    def select_many(
        self,
        policy: Any,
        probes: tuple[RouteProbe, ...],
        memory: MemoryState,
        retriever: Any,
        count: int,
        candidates_per_prompt: int,
    ) -> tuple[RouteProbe, ...]:
        remaining = list(probes)
        selected = []
        while remaining and len(selected) < count:
            window = tuple(remaining[:candidates_per_prompt])
            observation = self.observe(window, memory, retriever)
            response = policy.generate(self.build_prompt(observation), 64)
            try:
                choice = self.parse_choice(response, len(window))
            except (KeyError, TypeError, ValueError):
                remaining = remaining[len(window) :]
                continue
            probe = window[choice]
            selected.append(probe)
            remaining.remove(probe)
        return tuple(selected)
