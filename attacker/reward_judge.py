import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from attacker.answer_agent import is_insufficient_answer
from attacker.models import OracleResult
from utils.json_output import retry_json_object


SYSTEM_PROMPT = """Judge only semantic answer equivalence. All question validity,
grounding, route, novelty, and answer-leak constraints are handled elsewhere.

For each candidate answer, compare it independently with canonical_answer:
- correct: it answers the exact same fact with all essential information;
- partial: some essential information is correct but incomplete or underspecified;
- incorrect: wrong, contradictory, irrelevant, or unable to answer.
An adjacent fact, a different valid fact, or a list replacing one canonical item is
not correct merely because it is plausible or related.
Treat INSUFFICIENT_INFORMATION as incorrect. Return exactly one JSON object. Its
keys must exactly match the supplied candidate_answers keys and each value must be
correct, partial, or incorrect."""


@dataclass(frozen=True)
class RewardJudgeResult:
    gold_correctness: float
    memory_correctness: float
    parametric_correctness: float = 0.0
    # Kept for readers of old traces; the learned selector does not use it.
    value: float = 0.0


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
        golden_answer: str | None,
        memory_answer: str | None,
        parametric_answer: str | None = None,
    ) -> RewardJudgeResult:
        candidate_answers = {}
        if golden_answer is not None:
            candidate_answers["gold_correctness"] = golden_answer
        if memory_answer is not None:
            candidate_answers["memory_correctness"] = memory_answer
        if parametric_answer is not None:
            candidate_answers["parametric_correctness"] = parametric_answer
        payload = {
            "question": oracle.question,
            "canonical_answer": oracle.answer,
            "candidate_answers": candidate_answers,
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

        required = list(candidate_answers)

        def transform(result: dict[str, Any]) -> RewardJudgeResult:
            if set(result) != set(required):
                raise ValueError("Judge returned an invalid schema")
            equivalence = {"incorrect": 0.0, "partial": 0.5, "correct": 1.0}
            scores = {
                name: equivalence[result[name]]
                for name in required
            }
            for name, answer in candidate_answers.items():
                if is_insufficient_answer(answer):
                    scores[name] = 0.0
            return RewardJudgeResult(
                gold_correctness=scores.get("gold_correctness", 0.0),
                memory_correctness=scores.get("memory_correctness", 0.0),
                parametric_correctness=scores.get("parametric_correctness", 0.0),
            )

        return retry_json_object(
            request,
            required,
            transform,
        )
