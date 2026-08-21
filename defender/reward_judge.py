import json
import os
from dataclasses import asdict, dataclass
from typing import Any

from openai import OpenAI

from attacker.models import OracleResult, SupportingEvidence
from memory.models import CapabilityRecord, MemoryEditAction, MemoryNode


SYSTEM_PROMPT = """Evaluate a proposed long-term memory edit.

Score from 0 to 1:
- after_correctness: the new answer against the canonical answer.
- groundedness: whether new_memory is fully supported by new_evidence.
- action_quality: whether ADD, MERGE, DELETE, or NOOP and its targets are appropriate.
- memory_quality: whether new_memory is durable, concise, complete, and not a QA pair.
- retention_correctness: each protected answer against its canonical answer.

For DELETE or NOOP, groundedness and memory_quality are 1. Return JSON only:
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
    def __init__(self, client: Any, model: str, max_tokens: int = 500):
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
            "new_evidence": [asdict(item) for item in evidence],
            "memory_neighborhood": [node.to_dict() for node in neighborhood],
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
        response = self.client.chat.completions.create(
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
        result = json.loads(response.choices[0].message.content)
        return MemoryJudgeResult(
            after_correctness=float(result["after_correctness"]),
            groundedness=float(result["groundedness"]),
            action_quality=float(result["action_quality"]),
            memory_quality=float(result["memory_quality"]),
            retention_correctness=tuple(
                float(score) for score in result["retention_correctness"]
            ),
        )
