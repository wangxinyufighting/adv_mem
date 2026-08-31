import hashlib
import re
from dataclasses import replace

from memory.models import (
    CapabilityRecord,
    MemoryEvidence,
    MemoryEditAction,
    MemoryEditRecord,
    MemoryNode,
    MemoryOperation,
    MemoryState,
    MemoryStatus,
    RouteAttackStats,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def _node_id(version: int, action: MemoryEditAction) -> str:
    draft = action.new_memory.content if action.new_memory else ""
    value = "\0".join(
        (str(version), action.operation.value, *action.target_node_ids, draft)
    )
    return f"memory-{hashlib.sha256(value.encode()).hexdigest()[:24]}"


def _time_span(
    supplied: tuple[str | None, str | None] | None,
    targets: tuple[MemoryNode, ...],
) -> tuple[str | None, str | None] | None:
    values = [
        value
        for span in (supplied, *(node.time_span for node in targets))
        if span
        for value in span
        if value
    ]
    return (min(values), max(values)) if values else None


class MemoryStore:
    """Mutable owner of the current memory state M_t."""

    def __init__(self, state: MemoryState | None = None):
        self.state = state or MemoryState.empty()

    def add_node(self, node: MemoryNode) -> MemoryNode:
        if node.id in self.state.nodes:
            raise ValueError(f"Memory node already exists: {node.id}")

        version = self.state.version + 1
        stored = replace(
            node,
            status=MemoryStatus.ACTIVE,
            created_version=version,
            updated_version=version,
        )
        self.state.nodes[stored.id] = stored
        self.state.version = version
        self.state.edit_history.append(
            MemoryEditRecord(
                version=version,
                action="add",
                result_node_ids=(stored.id,),
            )
        )
        return stored

    def archive_node(self, node_id: str) -> MemoryNode:
        version = self.state.version + 1
        archived = replace(
            self.state.nodes[node_id],
            status=MemoryStatus.ARCHIVED,
            updated_version=version,
        )
        self.state.nodes[node_id] = archived
        self.state.version = version
        self.state.edit_history.append(
            MemoryEditRecord(
                version=version,
                action="archive",
                target_node_ids=(node_id,),
            )
        )
        return archived

    def apply_action(
        self,
        action: MemoryEditAction,
        *,
        question_id: str | None,
        provenance_node_ids: tuple[str, ...] = (),
        source_ids: tuple[str, ...] = (),
        time_span: tuple[str | None, str | None] | None = None,
        inherit_target_provenance: bool = False,
    ) -> MemoryState:
        """Apply one builder action as one memory version."""
        if action.new_memory is None:
            raise ValueError("ADD and MERGE require new memory content")
        if action.operation == MemoryOperation.ADD and action.target_node_ids:
            raise ValueError("ADD cannot target existing memories")
        if action.operation == MemoryOperation.MERGE and not action.target_node_ids:
            raise ValueError("MERGE requires at least one target")
        version = self.state.version + 1
        targets = tuple(self.state.nodes[node_id] for node_id in action.target_node_ids)

        if action.operation == MemoryOperation.MERGE:
            for node in targets:
                self.state.nodes[node.id] = replace(
                    node,
                    status=MemoryStatus.ARCHIVED,
                    updated_version=version,
                )

        draft = action.new_memory
        node = MemoryNode(
            id=_node_id(version, action),
            content=draft.content,
            provenance_node_ids=_unique(
                provenance_node_ids,
                *(
                    tuple(node.provenance_node_ids for node in targets)
                    if inherit_target_provenance
                    else ()
                ),
            ),
            source_ids=_unique(
                source_ids,
                *(
                    tuple(node.source_ids for node in targets)
                    if inherit_target_provenance
                    else ()
                ),
            ),
            linked_questions=_unique(
                *(node.linked_questions for node in targets),
                (question_id,) if question_id else (),
            ),
            tags=_unique(draft.tags, *(node.tags for node in targets)),
            time_span=_time_span(time_span, targets),
            token_count=estimate_token_count(draft.content),
            created_version=version,
            updated_version=version,
        )
        self.state.nodes[node.id] = node

        self.state.version = version
        self.state.edit_history.append(
            MemoryEditRecord(
                version=version,
                action=action.operation.value,
                target_node_ids=action.target_node_ids,
                result_node_ids=(node.id,),
            )
        )
        return self.state

    def record_capability(self, record: CapabilityRecord) -> None:
        self.state.capability_ledger[record.question_id] = record

    def record_route_attack(self, route_id: str, gap: str) -> None:
        """Record at most one identical outcome per memory version."""
        current = self.state.attack_history.get(route_id, RouteAttackStats(route_id))
        if (
            current.last_memory_version == self.state.version
            and current.last_gap == gap
        ):
            return
        counts = {
            "storage_gap": "storage_gaps",
            "retrieval_gap": "retrieval_gaps",
            "reasoning_gap": "reasoning_gaps",
            "none": "no_gaps",
        }
        field = counts[gap]
        values = current.to_dict()
        values["attempts"] += 1
        values[field] += 1
        values["last_memory_version"] = self.state.version
        values["last_gap"] = gap
        self.state.attack_history[route_id] = RouteAttackStats.from_dict(values)

    def mark_success(
        self,
        record: CapabilityRecord,
        node_ids: tuple[str, ...],
        evidence: tuple[MemoryEvidence, ...] = (),
    ) -> None:
        if not node_ids:
            raise ValueError("A defense success must be supported by memory")
        stored = replace(
            record,
            supporting_memory_node_ids=node_ids,
            passed=True,
            verified_version=self.state.version,
        )
        self.record_capability(stored)
        self.state.evidence_ledger[record.question_id] = evidence
        self.bind_question(node_ids, record.question_id)
        if record.question_id not in self.state.success_pool:
            self.state.success_pool.append(record.question_id)
        if record.question_id in self.state.high_priority_buffer:
            self.state.high_priority_buffer.remove(record.question_id)

    def mark_high_priority(
        self,
        record: CapabilityRecord,
        evidence: tuple[MemoryEvidence, ...] = (),
    ) -> None:
        self.record_capability(replace(record, passed=False))
        self.state.evidence_ledger[record.question_id] = evidence
        self.bind_question((), record.question_id)
        if record.question_id in self.state.success_pool:
            self.state.success_pool.remove(record.question_id)
        if record.question_id in self.state.high_priority_buffer:
            self.state.high_priority_buffer.remove(record.question_id)
        self.state.high_priority_buffer.append(record.question_id)

    def link_question(self, node_ids: tuple[str, ...], question_id: str) -> None:
        for node_id in node_ids:
            node = self.state.nodes[node_id]
            if question_id not in node.linked_questions:
                self.state.nodes[node_id] = replace(
                    node,
                    linked_questions=(*node.linked_questions, question_id),
                )

    def bind_question(self, node_ids: tuple[str, ...], question_id: str) -> None:
        """Bind a capability to only its currently attributed support."""
        for node_id, node in self.state.nodes.items():
            if question_id in node.linked_questions and node_id not in node_ids:
                self.state.nodes[node_id] = replace(
                    node,
                    linked_questions=tuple(
                        item for item in node.linked_questions if item != question_id
                    ),
                )
        self.link_question(node_ids, question_id)

    def advance_iteration(self) -> int:
        self.state.iteration += 1
        return self.state.iteration

    def snapshot(self) -> MemoryState:
        return self.state.snapshot()
