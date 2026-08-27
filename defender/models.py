from dataclasses import asdict, dataclass
from typing import Any

from attacker.models import OracleResult, SupportingEvidence
from memory.models import MemoryNode, MemoryState


@dataclass(frozen=True)
class MemoryBuilderObservation:
    memory_version: int
    question_id: str
    question: str
    new_evidence: tuple[SupportingEvidence, ...]
    memory_neighborhood: tuple[MemoryNode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "question_id": self.question_id,
            "question": self.question,
            "new_evidence": [asdict(item) for item in self.new_evidence],
            "memory_neighborhood": [
                node.to_dict() for node in self.memory_neighborhood
            ],
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
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
                    "memory_node_id": node.id,
                    "content": node.content,
                    "linked_questions": list(node.linked_questions),
                    "tags": list(node.tags),
                    "time_span": node.time_span,
                }
                for node in self.memory_neighborhood
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryBuilderObservation":
        return cls(
            memory_version=payload["memory_version"],
            question_id=payload["question_id"],
            question=payload["question"],
            new_evidence=tuple(
                SupportingEvidence(**item) for item in payload["new_evidence"]
            ),
            memory_neighborhood=tuple(
                MemoryNode.from_dict(item) for item in payload["memory_neighborhood"]
            ),
        )


@dataclass(frozen=True)
class MemoryBuilderRewardContext:
    observation: MemoryBuilderObservation
    memory: MemoryState
    oracle: OracleResult
    before_correctness: float

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
