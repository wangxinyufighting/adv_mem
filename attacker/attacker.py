import json
from typing import Any

from attacker.models import (
    AttackerObservation,
    AttackerRewardContext,
    GraphRouteBundle,
)
from memory.models import MemoryState
from utils.memory_retrieval import HybridMemoryRetriever


SYSTEM_PROMPT = """You generate one question that exposes a missing memory capability.

The question must be valuable, answerable from the route, not answerable from the
memory neighborhood, and different from prior questions. Follow the attack mode:
- single_fact: ask about one evidence fact.
- same_topic: combine at least two facts under the topic.
- temporal_evolution: ask about the change from the archived fact to the active fact.
- comparison: compare the two evidence facts.

Write a natural standalone question. Do not include the answer, evidence, node IDs,
or explanations. Return JSON only: {"question":"..."}"""


class Attacker:
    """Question policy interface used by Qwen rollouts in verl."""

    def __init__(self, neighborhood_size: int = 5, history_size: int = 10):
        self.neighborhood_size = neighborhood_size
        self.history_size = history_size

    def observe(
        self,
        route: GraphRouteBundle,
        memory: MemoryState,
        retriever: HybridMemoryRetriever,
    ) -> AttackerObservation:
        route_query = "\n".join(node.memory for node in route.evidence_nodes)
        results = retriever.retrieve(
            route_query,
            memory,
            top_k=self.neighborhood_size,
        )
        history = tuple(
            record.question for record in memory.capability_ledger.values()
        )[-self.history_size :]
        return AttackerObservation(
            route=route,
            memory_neighborhood=tuple(result.node for result in results),
            prior_questions=history,
        )

    def build_prompt(
        self,
        observation: AttackerObservation,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(observation.to_dict(), ensure_ascii=False),
            },
        ]

    @staticmethod
    def parse_question(response: str) -> str:
        return json.loads(response)["question"].strip()

    def to_verl_record(
        self,
        observation: AttackerObservation,
        memory: MemoryState,
    ) -> dict[str, Any]:
        reward_context = AttackerRewardContext.from_state(
            observation.route,
            memory,
        )
        return {
            "data_source": "attacker",
            "prompt": self.build_prompt(observation),
            "ability": "memory_attack",
            "reward_model": {
                "style": "rule",
                "ground_truth": observation.route.route_id,
            },
            "extra_info": reward_context.to_dict(),
        }
