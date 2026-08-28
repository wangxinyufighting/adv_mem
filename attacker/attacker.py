import json
from typing import Any

from attacker.models import (
    AttackerObservation,
    AttackerRewardContext,
    GraphRouteBundle,
)
from memory.models import MemoryState
from utils.json_output import is_clean_json_object, parse_json_object
from utils.memory_retrieval import HybridMemoryRetriever


SYSTEM_PROMPT = """Generate one natural question about only the supplied route.

The question must be valuable, answerable from the route, not answerable from the
memory neighborhood, and faithful to the attack mode:
- single_fact: ask for one detail without stating that detail in the question.
- same_topic: ask a question that needs at least two related memories.
- temporal_evolution: ask about an explicit change from an earlier event to a
  later event.
- comparison: ask for a meaningful comparison between two memories.

Do not import people, events, or facts that are absent from this route. The question
must be standalone and end with a question mark. Never mention routes,
evidence, facts, nodes, memory, context, IDs, or these instructions. Do not state
the answer inside the question. Return exactly one JSON object:
{"question":"..."}"""


class Attacker:
    """Question policy interface used by Qwen rollouts in verl."""

    def __init__(self, neighborhood_size: int = 5):
        self.neighborhood_size = neighborhood_size

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
        return AttackerObservation(
            route=route,
            memory_neighborhood=tuple(result.node for result in results),
        )

    def build_prompt(
        self,
        observation: AttackerObservation,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    json.dumps(observation.to_dict(), ensure_ascii=False)
                    + "\n\n/no_think"
                ),
            },
        ]

    @staticmethod
    def parse_question(response: str) -> str:
        payload = parse_json_object(response, ("question",))
        if (
            set(payload) != {"question"}
            or not isinstance(payload["question"], str)
            or not is_clean_json_object(response)
        ):
            raise ValueError("Invalid question schema")
        return payload["question"].strip()

    @staticmethod
    def normalize_question(question: str) -> str:
        return " ".join(question.casefold().split())

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
