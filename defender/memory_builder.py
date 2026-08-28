import json
from typing import Any

from attacker.models import OracleResult
from defender.models import (
    MemoryBuilderObservation,
    MemoryBuilderRewardContext,
    ProtectedQuestion,
)
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


SYSTEM_PROMPT = """Edit long-term conversational memory so the current question
becomes answerable without losing previously supported information.

Priorities, in order:
1. Grounded: every new claim must be supported by new_evidence or targeted memories.
2. Effective: the resulting memory must support the current question.
3. Retentive: preserve valid facts that may support protected_questions.
4. Minimal: use the smallest justified edit and concise memory content.

The question indicates relevance but is not evidence. Durable means reusable later,
not necessarily timeless; preserve dates and earlier/current states when they matter.

Choose exactly one operation:
- ADD: store useful evidence not represented by a nearby memory.
- MERGE: revise or combine nearby memories when evidence concerns the same person,
  event, topic, preference, or state. One target is allowed for revision.
- DELETE: remove a target only when it is demonstrably false or fully redundant and
  no replacement content is needed.
- NOOP: use only when no grounded edit can improve the memory.

Prefer MERGE over ADD when new evidence updates or extends a nearby memory. Do not
target unrelated memories. MERGE must preserve every valid answer-relevant fact from
its targets while integrating only supported new information. If evidence corrects
a false statement, remove it. If the user genuinely changed, preserve both states
and their order.

Write one or two concise, self-contained sentences. Preserve necessary names,
numbers, dates, negation, preferences, constraints, and relationships. Identify
assistant content as a prior assistant response, not as a user fact. Do not write a
transcript, question-answer pair, IDs, analysis, or unsupported inference.

Each nearby memory has an integer index. Targets may contain only unique indices
from memory_neighborhood. When memory_neighborhood is empty, only ADD or NOOP is
valid.

Return exactly one JSON object with no additional fields:
{"operation":"add","targets":[],"content":"..."}

operation must be add, merge, delete, or noop.
ADD: empty targets and non-empty content.
MERGE: non-empty targets and non-empty content.
DELETE: non-empty targets and empty content.
NOOP: empty targets and empty content."""

COMPACTION_PROMPT = """Compress the supplied long-term memory neighborhood only
when a smaller representation is lossless.

Priorities:
1. Preserve every supported fact and the answer to every protected question.
2. Reduce active memory size.
3. Use NOOP whenever lossless compression is uncertain.

Choose exactly one operation:
- MERGE: combine at least two related memories into one shorter memory while
  preserving every factual distinction from the targets.
- DELETE: remove a target only when all its information remains explicitly available
  in a non-targeted memory. Never delete every copy.
- NOOP: use when memories are unrelated, not redundant, or cannot be shortened
  without information loss.

Preserve names, quantities, dates, event order, negation, preferences, constraints,
and earlier/current states. An older fact is not redundant when it records a real
earlier state. Do not combine unrelated facts, invent information, or resolve
conflicts by guessing. Target only unique indices from memory_neighborhood.

Return exactly one JSON object with no additional fields:
{"operation":"noop","targets":[],"content":""}

operation must be merge, delete, or noop.
MERGE: at least two targets and non-empty content.
DELETE: non-empty targets and empty content.
NOOP: empty targets and empty content."""


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
        neighborhood: tuple[MemoryNode, ...],
        protected_questions: tuple[ProtectedQuestion, ...] = (),
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
                        ],
                        "protected_questions": [
                            {
                                "id": item.question_id,
                                "question": item.question,
                                "canonical_answer": item.canonical_answer,
                            }
                            for item in protected_questions
                        ],
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
