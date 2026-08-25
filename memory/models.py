from copy import deepcopy
from dataclasses import asdict, dataclass, field
from enum import Enum


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"


class MemoryOperation(str, Enum):
    ADD = "add"
    MERGE = "merge"
    DELETE = "delete"
    NOOP = "noop"


@dataclass(frozen=True)
class MemoryDraft:
    content: str
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class MemoryEditAction:
    operation: MemoryOperation
    target_node_ids: tuple[str, ...] = ()
    new_memory: MemoryDraft | None = None


@dataclass(frozen=True)
class MemoryEvidence:
    source_id: str
    node_id: str
    quote: str
    chat_time: str | None
    role: str | None


@dataclass(frozen=True)
class MemoryNode:
    """A memory in M_t; provenance IDs also define its lightweight graph links."""

    id: str
    content: str
    status: MemoryStatus = MemoryStatus.ACTIVE
    provenance_node_ids: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    linked_questions: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    time_span: tuple[str | None, str | None] | None = None
    confidence: float = 1.0
    token_count: int = 0
    created_version: int = 0
    updated_version: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "MemoryNode":
        payload = dict(payload)
        payload["status"] = MemoryStatus(payload["status"])
        for key in (
            "provenance_node_ids",
            "source_ids",
            "linked_questions",
            "tags",
        ):
            payload[key] = tuple(payload[key])
        if payload["time_span"] is not None:
            payload["time_span"] = tuple(payload["time_span"])
        return cls(**payload)


@dataclass(frozen=True)
class CapabilityRecord:
    question_id: str
    question: str
    route_id: str
    attack_mode: str
    oracle_answer: str
    oracle_source_ids: tuple[str, ...] = ()
    supporting_memory_node_ids: tuple[str, ...] = ()
    question_type: str | None = None
    difficulty: float | None = None
    passed: bool = False
    verified_version: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "CapabilityRecord":
        payload = dict(payload)
        payload["oracle_source_ids"] = tuple(payload["oracle_source_ids"])
        payload["supporting_memory_node_ids"] = tuple(
            payload["supporting_memory_node_ids"]
        )
        return cls(**payload)


@dataclass(frozen=True)
class MemoryEditRecord:
    version: int
    action: str
    target_node_ids: tuple[str, ...] = ()
    result_node_ids: tuple[str, ...] = ()


@dataclass
class MemoryState:
    version: int = 0
    iteration: int = 0
    nodes: dict[str, MemoryNode] = field(default_factory=dict)
    capability_ledger: dict[str, CapabilityRecord] = field(default_factory=dict)
    edit_history: list[MemoryEditRecord] = field(default_factory=list)
    evidence_ledger: dict[str, tuple[MemoryEvidence, ...]] = field(default_factory=dict)
    success_pool: list[str] = field(default_factory=list)
    high_priority_buffer: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "MemoryState":
        return cls()

    @property
    def active_nodes(self) -> tuple[MemoryNode, ...]:
        return tuple(
            node for node in self.nodes.values() if node.status == MemoryStatus.ACTIVE
        )

    @property
    def active_token_count(self) -> int:
        return sum(node.token_count for node in self.active_nodes)

    def snapshot(self) -> "MemoryState":
        return deepcopy(self)

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "iteration": self.iteration,
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "capability_ledger": [
                record.to_dict() for record in self.capability_ledger.values()
            ],
            "edit_history": [asdict(record) for record in self.edit_history],
            # A list keeps the Parquet schema stable across different question IDs.
            "evidence_ledger": [
                {
                    "question_id": question_id,
                    "evidence": [asdict(item) for item in evidence],
                }
                for question_id, evidence in self.evidence_ledger.items()
            ],
            "success_pool": list(self.success_pool),
            "high_priority_buffer": list(self.high_priority_buffer),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MemoryState":
        nodes = [MemoryNode.from_dict(item) for item in payload["nodes"]]
        records = [
            CapabilityRecord.from_dict(item) for item in payload["capability_ledger"]
        ]
        edits = []
        for item in payload["edit_history"]:
            item = dict(item)
            item["target_node_ids"] = tuple(item["target_node_ids"])
            item["result_node_ids"] = tuple(item["result_node_ids"])
            edits.append(MemoryEditRecord(**item))
        evidence_records = payload["evidence_ledger"]
        if isinstance(evidence_records, dict):
            # Read run_state.json files written before the Parquet-safe format.
            evidence_records = [
                {"question_id": question_id, "evidence": evidence}
                for question_id, evidence in evidence_records.items()
                if evidence is not None
            ]
        return cls(
            version=payload["version"],
            iteration=payload["iteration"],
            nodes={node.id: node for node in nodes},
            capability_ledger={record.question_id: record for record in records},
            edit_history=edits,
            evidence_ledger={
                record["question_id"]: tuple(
                    MemoryEvidence(**item) for item in record["evidence"]
                )
                for record in evidence_records
            },
            success_pool=list(payload["success_pool"]),
            high_priority_buffer=list(payload["high_priority_buffer"]),
        )
