from dataclasses import dataclass
from typing import Any

from attacker.answer_agent import QwenAnswerAgent, is_insufficient_answer
from defender.memory_builder import (
    ActionConstraintError,
    ActionSchemaError,
    MemoryBuilder,
)
from defender.models import MemoryBuilderRewardContext
from defender.reward_judge import DeepSeekMemoryJudge, ProtectedAnswer
from memory.models import CapabilityRecord, MemoryEditAction, MemoryOperation
from memory.store import estimate_token_count
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class MemoryBuilderRewardConfig:
    grounding_threshold: float = 0.8
    answer_threshold: float = 0.8
    retention_threshold: float = 0.8
    quality_threshold: float = 0.6
    retrieval_top_k: int = 5
    max_protected_questions: int = 3
    max_memory_tokens: int = 128
    gain_weight: float = 0.7
    quality_weight: float = 0.2
    growth_weight: float = 0.1
    shrink_weight: float = 0.1


class MemoryBuilderReward:
    """Behavioral reward for one temporary memory edit."""

    def __init__(
        self,
        builder: MemoryBuilder,
        answer_agent: Any,
        retriever: Any,
        judge: Any,
        config: MemoryBuilderRewardConfig | None = None,
    ):
        self.builder = builder
        self.answer_agent = answer_agent
        self.retriever = retriever
        self.judge = judge
        self.config = config or MemoryBuilderRewardConfig()

    @classmethod
    def from_env(cls) -> "MemoryBuilderReward":
        return cls(
            builder=MemoryBuilder(),
            answer_agent=QwenAnswerAgent.from_env(),
            retriever=HybridMemoryRetriever.from_env(),
            judge=DeepSeekMemoryJudge.from_env(),
        )

    def evaluate(
        self,
        response: str,
        context: MemoryBuilderRewardContext,
    ) -> dict[str, float]:
        try:
            action = self.builder.parse_action(
                response,
                context.observation.memory_neighborhood,
            )
        except ActionSchemaError:
            return self._result(-1.0)
        except ActionConstraintError:
            return self._result(-0.8, schema_valid=1.0)
        if not self._valid_action(action, context):
            return self._result(-0.7, schema_valid=1.0)
        temp = self.builder.execute(
            context.memory,
            context.observation,
            action,
        )
        growth, shrink = self._size_change(context, temp)

        after_results = self.retriever.retrieve(
            context.observation.question,
            temp,
            top_k=self.config.retrieval_top_k,
        )
        after_answer = self.answer_agent.answer_memories(
            context.observation.question,
            tuple(result.node for result in after_results),
        )

        protected = self._protected_capabilities(context)
        protected_answers = tuple(
            ProtectedAnswer(
                capability=record,
                answer=self._answer_capability(record, temp),
            )
            for record in protected
        )
        try:
            judged = self.judge.evaluate(
                action=action,
                evidence=context.observation.new_evidence,
                neighborhood=context.observation.memory_neighborhood,
                oracle=context.oracle,
                after_answer=after_answer,
                protected_answers=protected_answers,
            )
        except StructuredOutputError:
            return self._result(
                0.0,
                reward_available=0.0,
                schema_valid=1.0,
                action_valid=1.0,
                growth=growth,
                shrink=shrink,
            )

        after_correctness = (
            0.0 if is_insufficient_answer(after_answer) else judged.after_correctness
        )
        retention_scores = tuple(
            0.0 if is_insufficient_answer(item.answer) else score
            for item, score in zip(
                protected_answers,
                judged.retention_correctness,
                strict=True,
            )
        )
        gain = max(0.0, after_correctness - context.before_correctness)
        edit_quality = (judged.action_quality + judged.memory_quality) / 2
        retention_min = min(retention_scores, default=1.0)
        answer_pass = after_correctness >= self.config.answer_threshold
        retention_pass = retention_min >= self.config.retention_threshold
        failures = [
            value / threshold - 1.0
            for value, threshold in (
                (judged.groundedness, self.config.grounding_threshold),
                (after_correctness, self.config.answer_threshold),
                (judged.action_quality, self.config.quality_threshold),
                (judged.memory_quality, self.config.quality_threshold),
                (retention_min, self.config.retention_threshold),
            )
            if value < threshold
        ]
        commit_valid = not failures
        score = (
            min(failures)
            if failures
            else max(
                0.0,
                min(
                    1.0,
                    self.config.gain_weight * gain
                    + self.config.quality_weight * edit_quality
                    + self.config.shrink_weight * shrink
                    - self.config.growth_weight * growth,
                ),
            )
        )
        return self._result(
            score,
            schema_valid=1.0,
            action_valid=1.0,
            after_correctness=after_correctness,
            answer_pass=float(answer_pass),
            groundedness=judged.groundedness,
            action_quality=judged.action_quality,
            memory_quality=judged.memory_quality,
            retention_min=retention_min,
            retention_pass=float(retention_pass),
            growth=growth,
            shrink=shrink,
            commit_valid=float(commit_valid),
            gain=gain,
        )

    def _valid_action(
        self,
        action: MemoryEditAction,
        context: MemoryBuilderRewardContext,
    ) -> bool:
        neighborhood_ids = {node.id for node in context.observation.memory_neighborhood}
        targets = set(action.target_node_ids)
        if not targets <= neighborhood_ids:
            return False

        memory_tokens = (
            estimate_token_count(action.new_memory.content) if action.new_memory else 0
        )
        has_memory = 0 < memory_tokens <= self.config.max_memory_tokens
        if action.operation == MemoryOperation.ADD:
            return not targets and has_memory
        if action.operation == MemoryOperation.MERGE:
            return bool(targets) and has_memory
        if action.operation == MemoryOperation.DELETE:
            return bool(targets) and action.new_memory is None
        return not targets and action.new_memory is None

    def _protected_capabilities(
        self,
        context: MemoryBuilderRewardContext,
    ) -> tuple[CapabilityRecord, ...]:
        ledger = context.memory.capability_ledger
        return tuple(
            ledger[item.question_id]
            for item in context.observation.protected_questions[
                : self.config.max_protected_questions
            ]
        )

    def _answer_capability(
        self,
        capability: CapabilityRecord,
        memory,
    ) -> str:
        results = self.retriever.retrieve(
            capability.question,
            memory,
            top_k=self.config.retrieval_top_k,
        )
        return self.answer_agent.answer_memories(
            capability.question,
            tuple(result.node for result in results),
        )

    @staticmethod
    def _size_change(context: MemoryBuilderRewardContext, temp) -> tuple[float, float]:
        before = context.memory.active_token_count
        after = temp.active_token_count
        evidence_tokens = sum(
            estimate_token_count(item.quote)
            for item in context.observation.new_evidence
        )
        growth = min(1.0, max(0, after - before) / max(1, evidence_tokens))
        shrink = min(1.0, max(0, before - after) / max(1, before))
        return growth, shrink

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "schema_valid": 0.0,
            "action_valid": 0.0,
            "groundedness": 0.0,
            "after_correctness": 0.0,
            "answer_pass": 0.0,
            "action_quality": 0.0,
            "memory_quality": 0.0,
            "retention_min": 0.0,
            "retention_pass": 0.0,
            "growth": 0.0,
            "shrink": 0.0,
            "commit_valid": 0.0,
            "gain": 0.0,
        }
        result.update(values)
        result["retention"] = result["retention_min"]
        result["memory_cost"] = result["growth"]
        return result
