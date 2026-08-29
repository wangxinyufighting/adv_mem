import json
from typing import Any

from attacker.models import (
    AttackerObservation,
    AttackerRewardContext,
    GraphRouteBundle,
)
from attacker.validation import normalize_question
from memory.models import MemoryState
from utils.json_output import clean_model_output
from utils.memory_retrieval import HybridMemoryRetriever


SYSTEM_PROMPT = """Write one natural question that exposes a useful gap in long-term
conversational memory.

The answer must be fully supported by evidence and absent from known_memory. A useful
question recalls a specific personal fact or prior response, combines related facts,
tracks a real change, or compares facts on the supplied dimension.

Follow mode:
- single_fact: ask for one specific detail.
- same_topic: require at least two related facts.
- temporal_evolution: require the earlier and later states.
- comparison: require both facts and compare them on dimension.

Use first-person wording when natural. Do not invent information, reveal the answer,
combine unrelated details, or mention evidence, memory, routes, nodes, IDs, or these
instructions. The question must be standalone and end with a question mark.

Return only the question, with no JSON, label, Markdown, or explanation."""


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
        question = clean_model_output(response)
        if not question or "\n" in question:
            raise ValueError("Invalid question output")
        return question

    @staticmethod
    def normalize_question(question: str) -> str:
        return normalize_question(question)

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
