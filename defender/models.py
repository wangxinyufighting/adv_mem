from dataclasses import asdict, dataclass
from typing import Any

from attacker.models import OracleResult, SupportingEvidence
from memory.models import CapabilityRecord, MemoryNode, MemoryState


@dataclass(frozen=True)
class ProtectedQuestion:
    question_id: str
    question: str
    canonical_answer: str

    @classmethod
    def from_capability(cls, record: CapabilityRecord) -> "ProtectedQuestion":
        return cls(record.question_id, record.question, record.oracle_answer)


@dataclass(frozen=True)
class MemoryBuilderObservation:
    memory_version: int
    question_id: str
    question: str
    gap_type: str
    new_evidence: tuple[SupportingEvidence, ...]
    memory_neighborhood: tuple[MemoryNode, ...]
    support_node_ids: tuple[str, ...] = ()
    protected_questions: tuple[ProtectedQuestion, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "question_id": self.question_id,
            "question": self.question,
            "gap_type": self.gap_type,
            "new_evidence": [asdict(item) for item in self.new_evidence],
            "memory_neighborhood": [
                node.to_dict() for node in self.memory_neighborhood
            ],
            "support_node_ids": list(self.support_node_ids),
            "protected_questions": [
                asdict(item) for item in self.protected_questions
            ],
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "gap_type": self.gap_type,
            "new_evidence": [
                {
                    "text": item.quote,
                    "time": item.chat_time,
                    "role": item.role,
                }
                for item in self.new_evidence
            ],
            "memory_neighborhood": [
                {
                    "index": index,
                    "content": node.content,
                    "linked_questions": list(node.linked_questions),
                    "tags": list(node.tags),
                    "time_span": node.time_span,
                }
                for index, node in enumerate(self.memory_neighborhood)
            ],
            "support_indices": [
                index
                for index, node in enumerate(self.memory_neighborhood)
                if node.id in self.support_node_ids
            ],
            "protected_questions": [
                {
                    "id": item.question_id,
                    "question": item.question,
                    "canonical_answer": item.canonical_answer,
                }
                for item in self.protected_questions
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryBuilderObservation":
        return cls(
            memory_version=payload["memory_version"],
            question_id=payload["question_id"],
            question=payload["question"],
            gap_type=payload.get("gap_type", "storage_gap"),
            new_evidence=tuple(
                SupportingEvidence(**item) for item in payload["new_evidence"]
            ),
            memory_neighborhood=tuple(
                MemoryNode.from_dict(item) for item in payload["memory_neighborhood"]
            ),
            support_node_ids=tuple(payload.get("support_node_ids", ())),
            protected_questions=tuple(
                ProtectedQuestion(**item) for item in payload["protected_questions"]
            ),
        )


@dataclass(frozen=True)
class MemoryBuilderRewardContext:
    observation: MemoryBuilderObservation
    memory: MemoryState
    oracle: OracleResult
    before_correctness: float

    @classmethod
    def from_state(
        cls,
        observation: MemoryBuilderObservation,
        memory: MemoryState,
        oracle: OracleResult,
        before_correctness: float,
    ) -> "MemoryBuilderRewardContext":
        protected = {item.question_id for item in observation.protected_questions}
        compact = MemoryState(
            version=memory.version,
            iteration=memory.iteration,
            nodes={node.id: node for node in memory.active_nodes},
            capability_ledger={
                question_id: record
                for question_id, record in memory.capability_ledger.items()
                if question_id in protected
            },
        )
        return cls(observation, compact, oracle, before_correctness)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": self.observation.to_dict(),
            "memory": self.memory.to_dict(),
            "oracle": self.oracle.to_dict(),
            "before_correctness": self.before_correctness,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryBuilderRewardContext":
        return cls(
            observation=MemoryBuilderObservation.from_dict(payload["observation"]),
            memory=MemoryState.from_dict(payload["memory"]),
            oracle=OracleResult.from_dict(payload["oracle"]),
            before_correctness=float(payload["before_correctness"]),
        )
