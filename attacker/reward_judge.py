import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from attacker.models import OracleResult
from utils.json_output import retry_json_object


SYSTEM_PROMPT = """Evaluate a memory question and candidate answers.

Score each supplied answer's correctness from 0 to 1 against the canonical answer.
Score value from 0 to 1 based on whether the question captures useful, substantive,
natural, user-relevant information rather than artificial trivia. Judge all scores
independently. Return JSON only:
{"gold_correctness":0.0,"memory_correctness":0.0,"parametric_correctness":0.0,"value":0.0}"""


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
            "supporting_evidence": [
                {
                    "quote": item.quote,
                    "chat_time": item.chat_time,
                    "role": item.role,
                }
                for item in oracle.supporting_evidence
            ],
            "golden_answer": golden_answer,
            "memory_answer": memory_answer,
        }
        if parametric_answer is not None:
            payload["parametric_answer"] = parametric_answer

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

        required = ["gold_correctness", "memory_correctness", "value"]
        if parametric_answer is not None:
            required.append("parametric_correctness")
        return retry_json_object(
            request,
            required,
            lambda result: RewardJudgeResult(
                gold_correctness=float(result["gold_correctness"]),
                memory_correctness=float(result["memory_correctness"]),
                value=float(result["value"]),
                parametric_correctness=float(
                    result.get("parametric_correctness", 0.0)
                ),
            ),
        )
