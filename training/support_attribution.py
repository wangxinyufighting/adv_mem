from typing import Any

from attacker.answer_agent import is_insufficient_answer
from attacker.models import OracleResult
from memory.models import MemoryNode
from utils.json_output import StructuredOutputError


class SupportAttributor:
    """Find an answer-preserving subset of retrieved memories."""

    def __init__(self, answer_agent: Any, answer_judge: Any, threshold: float = 0.8):
        self.answer_agent = answer_agent
        self.answer_judge = answer_judge
        self.threshold = threshold

    def select(
        self,
        oracle: OracleResult,
        golden_answer: str,
        memories: tuple[MemoryNode, ...],
    ) -> tuple[str, ...]:
        selected = list(memories)
        # Keep a removal only when the remaining memories still answer correctly.
        for memory in reversed(memories):
            trial = tuple(node for node in selected if node.id != memory.id)
            if not trial:
                continue
            answer = self.answer_agent.answer_memories(oracle.question, trial)
            if is_insufficient_answer(answer):
                continue
            try:
                correctness = self.answer_judge.evaluate(
                    oracle,
                    None,
                    answer,
                ).memory_correctness
            except StructuredOutputError:
                continue
            if correctness >= self.threshold:
                selected = list(trial)
        return tuple(node.id for node in selected)
