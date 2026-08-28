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


SYSTEM_PROMPT = """Generate exactly one natural user question that tests long-term
conversational memory.

Input roles:
- route.evidence_nodes contain the information that must support the answer.
- route.connector_nodes only explain relationships and are not answer evidence.
- memory_neighborhood contains what the current memory already knows.
- route.attack_mode specifies the memory capability to test.

A valuable question is diagnostically useful for evaluating conversational memory.
It does not need to ask about an important life event. Directly recalling a degree,
purchase, preference, event, name, number, or previous recommendation can be
valuable when the answer is specific to the conversation.

A valuable question should test at least one of these capabilities:
- recall a specific user fact, experience, preference, or prior assistant response;
- combine related memories to calculate, count, summarize, or infer an answer;
- compare two remembered facts on a clear shared dimension;
- distinguish an earlier state from the latest state;
- reason about a change, order, duration, or temporal relationship;
- apply remembered preferences, constraints, or past experience to a natural
  recommendation or advice request.

Follow the selected attack mode:
- single_fact: ask for one specific detail from one evidence node. Direct recall is
  valid; do not make the question artificially complicated.
- same_topic: ask one coherent question whose answer requires at least two evidence
  nodes. Aggregation, summarization, and personalized application are allowed.
- temporal_evolution: ask about an earlier and later state, what changed, which is
  current, event order, or a supported temporal interval.
- comparison: ask for a meaningful difference, similarity, relative quantity, or
  choice that requires both compared facts.

The essential answer must be supported by the evidence nodes but absent from the
memory neighborhood. If the memory neighborhood already contains the answer,
target another supported detail or relationship.

Write the question as a plausible user follow-up, using first-person wording such
as "I" or "my" when natural. It must be standalone, clear, and objectively
answerable from the route. Recommendation questions are valid only when the route
contains concrete preferences or constraints that should shape the response.

Do not:
- use external knowledge or invent missing information;
- ask an abstention question about information that is not present;
- combine unrelated details merely to increase difficulty;
- reveal or strongly imply the answer in the question;
- mention routes, evidence, nodes, facts, memory, context, IDs, or instructions;
- output multiple unrelated questions or any explanation.

The question must end with a question mark. Return exactly one JSON object with no
Markdown or additional fields:
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
