import json
from typing import Any

from attacker.models import OracleResult
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from memory.models import (
    MemoryDraft,
    MemoryEditAction,
    MemoryNode,
    MemoryOperation,
    MemoryState,
)
from memory.store import MemoryStore
from utils.json_output import parse_json_object


class ActionSchemaError(ValueError):
    pass


class ActionConstraintError(ValueError):
    pass


SYSTEM_PROMPT = """You are a long-term memory editor.
Choose exactly one operation using only the new evidence and memory neighborhood.

ADD: create a new memory when the evidence contains useful knowledge not stored nearby.
MERGE: replace one or more nearby memories with a single better memory.
DELETE: archive nearby memories that are wrong, obsolete, or no longer useful.
NOOP: make no change when the evidence adds no useful memory.

Write at most two concise sentences of durable factual memory, not a transcript or
question-answer pair. Each nearby memory has an integer index. Targets may contain
only unique indices from memory_neighborhood. MERGE may target one memory when
revising it. When memory_neighborhood is empty, only ADD or NOOP is valid.

Return JSON only with exactly these three fields:
{"operation":"add","targets":[],"content":"..."}
operation must be add, merge, delete, or noop.
ADD: empty targets and non-empty content.
MERGE: non-empty targets and non-empty content.
DELETE: non-empty targets and empty content.
NOOP: empty targets and empty content."""

COMPACTION_PROMPT = """Compress a long-term memory without losing information.
Use MERGE to replace at least two memories with one shorter complete memory. Use
DELETE only when a memory is redundant. Otherwise use NOOP. Preserve every fact
needed by linked_questions and target only unique neighborhood indices.

Return JSON only with exactly these three fields:
{"operation":"noop","targets":[],"content":""}
operation must be merge, delete, or noop.
MERGE requires at least two targets and non-empty content. DELETE requires non-empty
targets and empty content. NOOP requires empty targets and empty content."""


class MemoryBuilder:
    """Prompt, parse, and execute the editing policy used by verl rollouts."""

    def build_prompt(
        self, observation: MemoryBuilderObservation
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    json.dumps(observation.to_prompt_dict(), ensure_ascii=False)
                    + "\n\n/no_think"
                ),
            },
        ]

    def build_compaction_prompt(
        self,
        neighborhood: tuple,
    ) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": COMPACTION_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "memory_neighborhood": [
                            {
                                "index": index,
                                "content": node.content,
                                "linked_questions": list(node.linked_questions),
                                "tags": list(node.tags),
                                "time_span": node.time_span,
                            }
                            for index, node in enumerate(neighborhood)
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n\n/no_think",
            },
        ]

    def parse_action(
        self,
        response: str,
        neighborhood: tuple[MemoryNode, ...],
    ) -> MemoryEditAction:
        try:
            payload = parse_json_object(
                response,
                ("operation", "targets", "content"),
            )
        except ValueError as error:
            raise ActionSchemaError from error
        if set(payload) != {"operation", "targets", "content"}:
            raise ActionSchemaError(
                "Action must contain exactly operation, targets, and content"
            )
        if not isinstance(payload["operation"], str):
            raise ActionSchemaError("operation must be a string")
        if not isinstance(payload["targets"], list) or any(
            type(index) is not int for index in payload["targets"]
        ):
            raise ActionSchemaError("targets must be a list of integer indices")
        if not isinstance(payload["content"], str):
            raise ActionSchemaError("content must be a string")

        try:
            operation = MemoryOperation(payload["operation"])
        except ValueError as error:
            raise ActionConstraintError from error
        indices = payload["targets"]
        content = payload["content"]
        if len(indices) != len(set(indices)):
            raise ActionConstraintError("targets must be unique")
        if any(index < 0 or index >= len(neighborhood) for index in indices):
            raise ActionConstraintError("target index is outside memory_neighborhood")

        has_content = bool(content.strip())
        valid_shape = {
            MemoryOperation.ADD: not indices and has_content,
            MemoryOperation.MERGE: bool(indices) and has_content,
            MemoryOperation.DELETE: bool(indices) and content == "",
            MemoryOperation.NOOP: not indices and content == "",
        }
        if not valid_shape[operation]:
            raise ActionConstraintError(f"Invalid fields for {operation.value}")

        return MemoryEditAction(
            operation=operation,
            target_node_ids=tuple(neighborhood[index].id for index in indices),
            new_memory=MemoryDraft(content.strip()) if has_content else None,
        )

    def execute(
        self,
        state: MemoryState,
        observation: MemoryBuilderObservation,
        action: MemoryEditAction,
    ) -> MemoryState:
        """Execute an action on a snapshot and return M_temp."""
        evidence_times = tuple(
            item.chat_time for item in observation.new_evidence if item.chat_time
        )
        time_span = (
            (min(evidence_times), max(evidence_times)) if evidence_times else None
        )
        store = MemoryStore(state.snapshot())
        return store.apply_action(
            action,
            question_id=observation.question_id,
            provenance_node_ids=tuple(
                dict.fromkeys(item.node_id for item in observation.new_evidence)
            ),
            source_ids=tuple(
                dict.fromkeys(item.source_id for item in observation.new_evidence)
            ),
            time_span=time_span,
        )

    @staticmethod
    def compact(state: MemoryState, action: MemoryEditAction) -> MemoryState:
        store = MemoryStore(state.snapshot())
        return store.apply_action(action, question_id=None)

    def to_verl_record(
        self,
        observation: MemoryBuilderObservation,
        memory: MemoryState,
        oracle: OracleResult,
        before_correctness: float,
    ) -> dict[str, Any]:
        context = MemoryBuilderRewardContext(
            observation=observation,
            memory=memory.snapshot(),
            oracle=oracle,
            before_correctness=before_correctness,
        )
        return {
            "data_source": "memory_builder",
            "prompt": self.build_prompt(observation),
            "ability": "memory_edit",
            "reward_model": {
                "style": "rule",
                "ground_truth": observation.question_id,
            },
            "extra_info": context.to_dict(),
        }
