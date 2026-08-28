import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from attacker.models import OracleResult
from utils.json_output import retry_json_object


SYSTEM_PROMPT = """Judge only semantic answer equivalence and long-term-memory value.
All format, grounding, route, novelty, and answer-leak constraints were checked by
code and must not affect this judgment.

For each candidate answer, compare it independently with canonical_answer:
- correct: all essential information is semantically correct with no contradiction;
- partial: some essential information is correct but incomplete or underspecified;
- incorrect: wrong, contradictory, irrelevant, or unable to answer.
Treat NOT_PROVIDED as incorrect.

Judge question value independently from the candidate answers:
- high: meaningfully tests integration, comparison, temporal change, state updates,
  personalized application, or recall of a substantive prior response;
- medium: clearly tests one specific conversational fact, preference, decision,
  experience, name, number, or response;
- low: vague, generic, incidental, mechanically combined, or weakly diagnostic.

Complexity alone does not create value. Return exactly one JSON object with no
additional fields:
{"gold_correctness":"correct","memory_correctness":"incorrect","parametric_correctness":"incorrect","value":"medium"}"""


@dataclass(frozen=True)
class RewardJudgeResult:
    gold_correctness: float
    memory_correctness: float
    value: float
    parametric_correctness: float = 0.0


class DeepSeekRewardJudge:
    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls) -> "DeepSeekRewardJudge":
        return cls(
            client=OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            ),
            model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            max_tokens=int(os.getenv("ATTACKER_JUDGE_MAX_TOKENS", "1024")),
        )

    def evaluate(
        self,
        oracle: OracleResult,
        golden_answer: str,
        memory_answer: str,
        parametric_answer: str | None = None,
    ) -> RewardJudgeResult:
        payload = {
            "question": oracle.question,
            "canonical_answer": oracle.answer,
            "golden_answer": golden_answer,
            "memory_answer": memory_answer,
            "parametric_answer": parametric_answer or "NOT_PROVIDED",
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

        required = [
            "gold_correctness",
            "memory_correctness",
            "parametric_correctness",
            "value",
        ]

        def transform(result: dict[str, Any]) -> RewardJudgeResult:
            if set(result) != set(required):
                raise ValueError("Judge returned an invalid schema")
            equivalence = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}
            value = {"low": 0.0, "medium": 0.5, "high": 1.0}
            scores = {
                name: (value if name == "value" else equivalence)[result[name]]
                for name in required
            }
            return RewardJudgeResult(
                gold_correctness=scores["gold_correctness"],
                memory_correctness=scores["memory_correctness"],
                value=scores["value"],
                parametric_correctness=scores.get("parametric_correctness", 0.0),
            )

        return retry_json_object(
            request,
            required,
            transform,
        )
