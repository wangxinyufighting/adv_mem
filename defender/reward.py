from dataclasses import dataclass
from typing import Any

from openai import APIError

from attacker.answer_agent import QwenAnswerAgent, is_insufficient_answer
from attacker.models import OracleResult
from attacker.reward_judge import DeepSeekRewardJudge
from defender.memory_builder import ContentSchemaError, MemoryBuilder
from defender.models import MemoryBuilderRewardContext
from defender.reward_judge import DeepSeekMemoryJudge
from memory.models import CapabilityRecord
from memory.store import estimate_token_count
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class MemoryBuilderRewardConfig:
    answer_threshold: float = 0.8
    retention_threshold: float = 0.8
    regression_weight: float = 1.0
    length_weight: float = 0.05
    retrieval_top_k: int = 5
    max_memory_tokens: int = 128


class MemoryBuilderReward:
    """R = correctness gain - regression - normalized memory length."""

    def __init__(
        self,
        builder: MemoryBuilder,
        answer_agent: Any,
        retriever: Any,
        answer_judge: Any,
        edit_judge: Any,
        config: MemoryBuilderRewardConfig | None = None,
    ):
        self.builder = builder
        self.answer_agent = answer_agent
        self.retriever = retriever
        self.answer_judge = answer_judge
        self.edit_judge = edit_judge
        self.config = config or MemoryBuilderRewardConfig()

    @classmethod
    def from_env(cls) -> "MemoryBuilderReward":
        return cls(
            MemoryBuilder(),
            QwenAnswerAgent.from_env(),
            HybridMemoryRetriever.from_env(),
            DeepSeekRewardJudge.from_env(),
            DeepSeekMemoryJudge.from_env(),
        )

    def evaluate(
        self,
        response: str,
        context: MemoryBuilderRewardContext,
    ) -> dict[str, float]:
        try:
            content = self.builder.parse_content(response)
        except ContentSchemaError:
            return self._result(-1.0)
        tokens = estimate_token_count(content)
        if not 0 < tokens <= self.config.max_memory_tokens:
            return self._result(-1.0, format_valid=1.0)

        action = self.builder.action(content, context.observation)
        temp = self.builder.execute(context.memory, context.observation, content)
        try:
            valid = self.edit_judge.evaluate(
                action,
                context.observation.new_evidence,
                context.observation.target_memories,
            ).valid
        except (APIError, StructuredOutputError) as error:
            return self._unavailable("grounding", error, format_valid=1.0)
        if not valid:
            return self._result(
                -1.0,
                format_valid=1.0,
                grounding_available=1.0,
            )

        try:
            after = self._correctness(context.oracle, temp)
            retention = tuple(
                self._correctness(self._oracle(record), temp)
                for record in context.memory.capability_ledger.values()
            )
        except (APIError, StructuredOutputError) as error:
            return self._unavailable(
                "answer", error, format_valid=1.0, grounding_available=1.0
            )

        retention_min = min(retention, default=1.0)
        gain = after - context.before_correctness
        regression = 1.0 - retention_min
        length = tokens / self.config.max_memory_tokens
        score = (
            gain
            - self.config.regression_weight * regression
            - self.config.length_weight * length
        )
        commit_valid = (
            after >= self.config.answer_threshold
            and retention_min >= self.config.retention_threshold
        )
        return self._result(
            score,
            format_valid=1.0,
            grounded=1.0,
            grounding_available=1.0,
            answer_available=1.0,
            after_correctness=after,
            gain=gain,
            regression=regression,
            length=length,
            retention_min=retention_min,
            commit_valid=float(commit_valid),
        )

    def _correctness(self, oracle: OracleResult, memory) -> float:
        results = self.retriever.retrieve(
            oracle.question, memory, top_k=self.config.retrieval_top_k
        )
        answer = self.answer_agent.answer_memories(
            oracle.question, tuple(result.node for result in results)
        )
        if is_insufficient_answer(answer):
            return 0.0
        return self.answer_judge.evaluate(oracle, None, answer).memory_correctness

    @staticmethod
    def _oracle(record: CapabilityRecord) -> OracleResult:
        return OracleResult(
            route_id=record.route_id,
            question=record.question,
            valid=True,
            answer=record.oracle_answer,
            supporting_evidence=(),
            invalid_reason=None,
            confidence=1.0,
        )

    def _unavailable(
        self,
        stage: str,
        error: Exception,
        **values: float,
    ) -> dict[str, float]:
        print(
            f"Memory Builder Reward unavailable: stage={stage} "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        return self._result(0.0, reward_available=0.0, **values)

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "format_valid": 0.0,
            "grounded": 0.0,
            "grounding_available": 0.0,
            "answer_available": 0.0,
            "after_correctness": 0.0,
            "gain": 0.0,
            "regression": 0.0,
            "length": 0.0,
            "retention_min": 0.0,
            "commit_valid": 0.0,
        }
        result.update(values)
        return result
