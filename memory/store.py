import re
from dataclasses import replace
from uuid import uuid4

from memory.models import (
    CapabilityRecord,
    MemoryEvidence,
    MemoryEditAction,
    MemoryEditRecord,
    MemoryNode,
    MemoryOperation,
    MemoryState,
    MemoryStatus,
)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]")


def _unique(*groups: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(item for group in groups for item in group))


def estimate_token_count(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


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
    ) -> MemoryState:
        """Apply one builder action as one memory version."""
        if action.operation == MemoryOperation.NOOP:
            return self.state

        version = self.state.version + 1
        targets = tuple(self.state.nodes[node_id] for node_id in action.target_node_ids)

        if action.operation in (MemoryOperation.MERGE, MemoryOperation.DELETE):
            for node in targets:
                self.state.nodes[node.id] = replace(
                    node,
                    status=MemoryStatus.ARCHIVED,
                    updated_version=version,
                )

        result_node_ids: tuple[str, ...] = ()
        if action.operation in (MemoryOperation.ADD, MemoryOperation.MERGE):
            draft = action.new_memory
            node = MemoryNode(
                id=str(uuid4()),
                content=draft.content,
                provenance_node_ids=_unique(
                    provenance_node_ids,
                    *(node.provenance_node_ids for node in targets),
                ),
                source_ids=_unique(
                    source_ids,
                    *(node.source_ids for node in targets),
                ),
                linked_questions=_unique(
                    *(node.linked_questions for node in targets),
                    (question_id,) if question_id else (),
                ),
                tags=_unique(
                    draft.tags,
                    *(node.tags for node in targets),
                ),
                time_span=time_span,
                token_count=estimate_token_count(draft.content),
                created_version=version,
                updated_version=version,
            )
            self.state.nodes[node.id] = node
            result_node_ids = (node.id,)

        self.state.version = version
        self.state.edit_history.append(
            MemoryEditRecord(
                version=version,
                action=action.operation.value,
                target_node_ids=action.target_node_ids,
                result_node_ids=result_node_ids,
            )
        )
        return self.state

    def record_capability(self, record: CapabilityRecord) -> None:
        self.state.capability_ledger[record.question_id] = record

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
        if record.question_id not in self.state.high_priority_buffer:
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
