import json

from attacker.models import (
    AttackMode,
    AttackerObservation,
    GraphRouteBundle,
)
from attacker.validation import normalize_question
from memory.models import MemoryState
from utils.json_output import clean_model_output
from utils.memory_retrieval import HybridMemoryRetriever


SYSTEM_PROMPT = """Write one natural, standalone question.

Useful questions sound like real requests to recall or use an earlier conversation:
a personal detail, event, preference, decision, plan, current state, recommendation,
or other substantive prior answer.

The answer must be determined by target and not already determined by known. Make the
subject specific, but do not reveal the answer. Use I/my for the user's life and you
when recalling an earlier assistant response.

Do not invent information or join unrelated details. Never mention the input or task.
Return exactly one question on one line, ending with a question mark."""

MODE_PROMPTS = {
    AttackMode.SINGLE_FACT: (
        "Ask for one specific name, person, place, time, quantity, preference, "
        "decision, status, or prior-response detail."
    ),
    AttackMode.SAME_TOPIC: (
        "Use at least two related target entries. Ask for a total, count, combined "
        "result, relationship, or personalized answer that depends on all of them."
    ),
    AttackMode.TEMPORAL_EVOLUTION: (
        "Use earlier and later target entries. Ask about a real change, previous "
        "and current states, event order, or elapsed time."
    ),
    AttackMode.COMPARISON: (
        "Compare at least two target entries on dimension. Ask which is greater, "
        "most or least, or for the difference."
    ),
}


class Attacker:
    """Compatibility wrapper for the frozen Probe Question prompt.

    The learned policy is RouteSelector. This class no longer creates Verl records.
    """

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
            {
                "role": "system",
                "content": (
                    f"{SYSTEM_PROMPT}\n\n"
                    f"{MODE_PROMPTS[observation.route.attack_mode]}"
                ),
            },
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
