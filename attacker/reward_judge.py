import json
import os
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from attacker.models import OracleResult
from utils.json_output import retry_json_object


SYSTEM_PROMPT = """Evaluate one conversational-memory question and the candidate
answers supplied in the input.

Correctness:
- Score each candidate independently against the canonical answer.
- 1.0 means the essential answer is fully correct and has no contradiction.
- 0.5 means partially correct, incomplete, or materially underspecified.
- 0.0 means incorrect, contradictory, irrelevant, or unable to answer.
- Judge semantic equivalence rather than exact wording.
- If parametric_answer is absent, set parametric_correctness to 0.0.

Value measures the question's diagnostic value for long-term conversational memory,
not how important the subject is in everyday life. Judge value independently from
candidate correctness, novelty, and whether the current memory already answers it.

Use this value rubric:
- 0.90-1.00: strongly tests integration, comparison, temporal reasoning, state
  updates, personalized application, or another substantive memory capability.
- 0.70-0.89: clearly requires multiple related memories or a specific previous
  assistant response, decision, preference, or constraint.
- 0.50-0.69: clearly recalls one specific user or conversation fact. Direct recall
  remains valuable even when the fact is ordinary or the question is easy.
- 0.20-0.49: user-related but vague, weakly diagnostic, mechanically combined, or
  focused on incidental metadata without a meaningful memory target.
- 0.00-0.19: generic knowledge, answer leakage, unsupported content, graph
  metadata, or an artificial question with no clear memory capability.

Do not reward complexity by itself. A multi-fact question is valuable only when the
facts jointly determine a clear answer. A recommendation or advice question is
valuable when remembered preferences, constraints, or prior experience should
materially shape the response.

Return exactly one JSON object with numeric values from 0 to 1 and no additional
fields:
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

        def transform(result: dict[str, Any]) -> RewardJudgeResult:
            if any(
                isinstance(result[name], bool)
                or not isinstance(result[name], (int, float))
                for name in required
            ):
                raise TypeError("Judge scores must be numeric")
            scores = {name: float(result[name]) for name in required}
            if any(not 0.0 <= score <= 1.0 for score in scores.values()):
                raise ValueError("Judge scores must be between 0 and 1")
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
