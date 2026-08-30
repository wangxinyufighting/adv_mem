from dataclasses import dataclass
from enum import Enum
from typing import Any

from attacker.answer_agent import is_insufficient_answer
from attacker.models import RouteProbe
from memory.models import MemoryNode, MemoryState


class GapType(str, Enum):
    STORAGE = "storage_gap"
    RETRIEVAL = "retrieval_gap"
    REASONING = "reasoning_gap"
    NONE = "none"


@dataclass(frozen=True)
class GapEvaluation:
    gap_type: GapType
    correctness: float
    memories: tuple[MemoryNode, ...]
    memory_answer: str | None
    structural_coverage: float
    retrieved_coverage: float


class GapEvaluator:
    """Separate storage, retrieval and answer failures for one frozen probe."""

    def __init__(
        self,
        retriever: Any,
        answer_agent: Any,
        judge: Any,
        top_k: int = 5,
        correctness_threshold: float = 0.8,
    ):
        self.retriever = retriever
        self.answer_agent = answer_agent
        self.judge = judge
        self.top_k = top_k
        self.correctness_threshold = correctness_threshold

    def evaluate(self, probe: RouteProbe, memory: MemoryState) -> GapEvaluation:
        results = self.retriever.retrieve(
            probe.oracle.question,
            memory,
            top_k=self.top_k,
        )
        memories = tuple(result.node for result in results)
        structural = support_coverage(probe, memory.active_nodes)
        retrieved = support_coverage(probe, memories)

        if not memories:
            gap = GapType.STORAGE if structural < 1.0 else GapType.RETRIEVAL
            return GapEvaluation(gap, 0.0, (), None, structural, retrieved)

        answer = self.answer_agent.answer_memories(
            probe.oracle.question,
            memories,
        )
        if is_insufficient_answer(answer):
            return GapEvaluation(
                _failed_gap(structural, retrieved),
                0.0,
                memories,
                answer,
                structural,
                retrieved,
            )

        correctness = self.judge.evaluate(
            probe.oracle,
            None,
            answer,
        ).memory_correctness
        gap = (
            GapType.NONE
            if correctness >= self.correctness_threshold
            else _failed_gap(structural, retrieved)
        )
        return GapEvaluation(
            gap,
            correctness,
            memories,
            answer,
            structural,
            retrieved,
        )


def support_coverage(
    probe: RouteProbe,
    memories: tuple[MemoryNode, ...],
) -> float:
    required_nodes = {
        item.node_id for item in probe.oracle.supporting_evidence
    }
    required_sources = {
        item.source_id for item in probe.oracle.supporting_evidence
    }
    if not required_nodes and not required_sources:
        return 0.0

    covered_nodes = {
        node_id
        for memory in memories
        for node_id in memory.provenance_node_ids
    }
    covered_sources = {
        source_id
        for memory in memories
        for source_id in memory.source_ids
    }
    node_score = (
        len(required_nodes & covered_nodes) / len(required_nodes)
        if required_nodes
        else 0.0
    )
    source_score = (
        len(required_sources & covered_sources) / len(required_sources)
        if required_sources
        else 0.0
    )
    return max(node_score, source_score)


def _failed_gap(structural: float, retrieved: float) -> GapType:
    if structural < 1.0:
        return GapType.STORAGE
    if retrieved < 1.0:
        return GapType.RETRIEVAL
    return GapType.REASONING
