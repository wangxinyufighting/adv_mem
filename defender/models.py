from dataclasses import asdict, dataclass
from typing import Any

from attacker.models import OracleResult, SupportingEvidence
from memory.models import MemoryNode, MemoryOperation, MemoryState


@dataclass(frozen=True)
class RepairPlan:
    operation: MemoryOperation
    target_node_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "target_node_ids": list(self.target_node_ids),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RepairPlan":
        return cls(
            MemoryOperation(payload["operation"]),
            tuple(payload.get("target_node_ids", ())),
        )


@dataclass(frozen=True)
class MemoryBuilderObservation:
    memory_version: int
    question_id: str
    question: str
    new_evidence: tuple[SupportingEvidence, ...]
    target_memories: tuple[MemoryNode, ...]
    plan: RepairPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "question_id": self.question_id,
            "question": self.question,
            "new_evidence": [asdict(item) for item in self.new_evidence],
            "target_memories": [node.to_dict() for node in self.target_memories],
            "plan": self.plan.to_dict(),
        }

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "operation": self.plan.operation.value,
            "new_evidence": [
                {"text": item.quote, "time": item.chat_time, "role": item.role}
                for item in self.new_evidence
            ],
            "target_memories": [node.content for node in self.target_memories],
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
            target_memories=tuple(
                MemoryNode.from_dict(item) for item in payload["target_memories"]
            ),
            plan=RepairPlan.from_dict(payload["plan"]),
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
        linked = {
            question_id
            for node in observation.target_memories
            for question_id in node.linked_questions
        }
        compact = MemoryState(
            version=memory.version,
            iteration=memory.iteration,
            nodes={node.id: node for node in memory.active_nodes},
            capability_ledger={
                question_id: record
                for question_id, record in memory.capability_ledger.items()
                if question_id in linked and record.passed
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
            MemoryBuilderObservation.from_dict(payload["observation"]),
            MemoryState.from_dict(payload["memory"]),
            OracleResult.from_dict(payload["oracle"]),
            float(payload["before_correctness"]),
        )
