import json
from typing import Any

from attacker.models import OracleResult
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from memory.models import (
    MemoryDraft,
    MemoryEditAction,
    MemoryOperation,
    MemoryState,
)
from memory.store import MemoryStore
from utils.json_output import parse_json_object


SYSTEM_PROMPT = """You are a long-term memory editor.
Choose exactly one operation using only the new evidence and memory neighborhood.

ADD: create a new memory when the evidence contains useful knowledge not stored nearby.
MERGE: replace one or more nearby memories with a single better memory.
DELETE: archive nearby memories that are wrong, obsolete, or no longer useful.
NOOP: make no change when the evidence adds no useful memory.

Write durable factual memory, not a question-answer pair. MERGE may target one node when
revising it. Target only node IDs from memory_neighborhood.

Return JSON only:
{"operation":"add|merge|delete|noop","target_node_ids":[],"new_memory":{"content":"...","tags":["..."]}|null}
new_memory is required for ADD and MERGE and must be null for DELETE and NOOP."""

COMPACTION_PROMPT = """Compress a long-term memory without losing information.
Use MERGE to replace at least two memories with one shorter complete memory. Use
DELETE only when a memory is redundant. Otherwise use NOOP. Preserve every fact
needed by linked_questions and target only IDs from memory_neighborhood.

Return JSON only:
{"operation":"merge|delete|noop","target_node_ids":[],"new_memory":{"content":"...","tags":["..."]}|null}
"""


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
                    json.dumps(observation.to_dict(), ensure_ascii=False)
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
                            node.to_dict() for node in neighborhood
                        ]
                    },
                    ensure_ascii=False,
                )
                + "\n\n/no_think",
            },
        ]

    def parse_action(self, response: str) -> MemoryEditAction:
        payload = parse_json_object(
            response,
            ("operation", "target_node_ids", "new_memory"),
        )
        new_memory = payload["new_memory"]
        draft = (
            MemoryDraft(
                content=new_memory["content"],
                tags=tuple(new_memory["tags"]),
            )
            if new_memory is not None
            else None
        )
        return MemoryEditAction(
            operation=MemoryOperation(payload["operation"]),
            target_node_ids=tuple(payload["target_node_ids"]),
            new_memory=draft,
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
