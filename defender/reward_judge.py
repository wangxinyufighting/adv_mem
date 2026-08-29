import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from attacker.answer_agent import is_insufficient_answer
from attacker.models import OracleResult, SupportingEvidence
from memory.models import CapabilityRecord, MemoryEditAction, MemoryNode
from utils.json_output import retry_json_object


SYSTEM_PROMPT = """Evaluate one proposed long-term conversational-memory edit.
Score every field independently from 0 to 1.

Correctness:
- after_correctness compares after_answer with canonical_answer.
- retention_correctness compares each protected after_answer with its canonical
  answer, preserving the supplied order.
- Use 1.0 for fully correct, 0.5 for partially correct or materially incomplete,
  and 0.0 for incorrect, contradictory, irrelevant, or unable to answer.
- Judge semantic equivalence rather than exact wording.

Groundedness:
- ADD content may use only new_evidence.
- MERGE content may use new_evidence and targeted memories.
- Score 1.0 only when every claim is supported and roles are represented correctly.
- For DELETE and NOOP, groundedness is 1.0.

Action quality:
- ADD is appropriate for separate useful evidence not represented nearby.
- MERGE is appropriate for extending, correcting, or combining related memories;
  targets must be relevant and minimal.
- DELETE is appropriate only for a demonstrably false or fully redundant memory
  when no replacement content is needed.
- NOOP is appropriate only when no grounded edit can improve memory.
- Preserve real temporal evolution; an older true state is not obsolete merely
  because a newer state exists.

Memory quality:
- New memory should be self-contained, concise, complete, and reusable.
- It must preserve necessary names, numbers, dates, negation, preferences,
  constraints, temporal order, and valid details from targeted memories.
- It must not be a transcript, question-answer pair, ID, unsupported inference, or
  assistant statement misrepresented as a user fact.
- For DELETE and NOOP, memory_quality is 1.0.

Return exactly one JSON object with no additional fields:
{"after_correctness":0.0,"groundedness":0.0,"action_quality":0.0,"memory_quality":0.0,"retention_correctness":[]}"""


@dataclass(frozen=True)
class ProtectedAnswer:
    capability: CapabilityRecord
    answer: str


@dataclass(frozen=True)
class MemoryJudgeResult:
    after_correctness: float
    groundedness: float
    action_quality: float
    memory_quality: float
    retention_correctness: tuple[float, ...]


class DeepSeekMemoryJudge:
    def __init__(self, client: Any, model: str, max_tokens: int = 1536):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls) -> "DeepSeekMemoryJudge":
        return cls(
            client=OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            ),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            max_tokens=int(os.getenv("MEMORY_JUDGE_MAX_TOKENS", "1536")),
        )

    def evaluate(
        self,
        action: MemoryEditAction,
        evidence: tuple[SupportingEvidence, ...],
        neighborhood: tuple[MemoryNode, ...],
        oracle: OracleResult,
        after_answer: str,
        protected_answers: tuple[ProtectedAnswer, ...],
    ) -> MemoryJudgeResult:
        payload = {
            "action": {
                "operation": action.operation.value,
                "target_node_ids": list(action.target_node_ids),
                "new_memory": asdict(action.new_memory) if action.new_memory else None,
            },
            "new_evidence": [
                {
                    "quote": item.quote,
                    "chat_time": item.chat_time,
                    "role": item.role,
                }
                for item in evidence
            ],
            "memory_neighborhood": [
                {
                    "id": node.id,
                    "content": node.content,
                    "linked_questions": list(node.linked_questions),
                }
                for node in neighborhood
            ],
            "question": oracle.question,
            "canonical_answer": oracle.answer,
            "after_answer": after_answer,
            "protected_answers": [
                {
                    "question": item.capability.question,
                    "canonical_answer": item.capability.oracle_answer,
                    "after_answer": item.answer,
                }
                for item in protected_answers
            ],
        }

        def request():
            return self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=self.max_tokens,
            )

        required = (
            "after_correctness",
            "groundedness",
            "action_quality",
            "memory_quality",
            "retention_correctness",
        )

        def transform(result: dict[str, Any]) -> MemoryJudgeResult:
            names = required[:-1]
            if any(
                isinstance(result[name], bool)
                or not isinstance(result[name], (int, float))
                for name in names
            ):
                raise TypeError("Judge scores must be numeric")
            scores = tuple(float(result[name]) for name in names)
            retention = result["retention_correctness"]
            if not isinstance(retention, list) or any(
                isinstance(score, bool) or not isinstance(score, (int, float))
                for score in retention
            ):
                raise TypeError("retention_correctness must be a numeric list")
            retention_scores = tuple(float(score) for score in retention)
            if any(not 0.0 <= score <= 1.0 for score in (*scores, *retention_scores)):
                raise ValueError("Judge scores must be between 0 and 1")
            if len(retention_scores) != len(protected_answers):
                raise ValueError("Judge returned the wrong number of retention scores")
            if is_insufficient_answer(after_answer):
                scores = (0.0, *scores[1:])
            retention_scores = tuple(
                0.0 if is_insufficient_answer(item.answer) else score
                for item, score in zip(
                    protected_answers,
                    retention_scores,
                    strict=True,
                )
            )
            return MemoryJudgeResult(*scores, retention_scores)

        return retry_json_object(
            request,
            required,
            transform,
        )
