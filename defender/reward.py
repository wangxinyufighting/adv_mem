import json
from dataclasses import dataclass
from typing import Any

from attacker.answer_agent import QwenAnswerAgent
from defender.memory_builder import MemoryBuilder
from defender.models import MemoryBuilderRewardContext
from defender.reward_judge import DeepSeekMemoryJudge, ProtectedAnswer
from memory.models import CapabilityRecord, MemoryEditAction, MemoryOperation
from memory.store import estimate_token_count
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class MemoryBuilderRewardConfig:
    grounding_threshold: float = 0.8
    retrieval_top_k: int = 5
    max_protected_questions: int = 3
    memory_cost_weight: float = 0.05
    max_memory_tokens: int = 128


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
        except (json.JSONDecodeError, KeyError, TypeError, AttributeError, ValueError):
            return self._invalid()
        if not self._valid_action(action, context):
            return self._invalid(-0.7)
        temp = self.builder.execute(
            context.memory,
            context.observation,
            action,
        )

        after_results = self.retriever.retrieve(
            context.observation.question,
            temp,
            top_k=self.config.retrieval_top_k,
        )
        after_answer = self.answer_agent.answer_memories(
            context.observation.question,
            tuple(result.node for result in after_results),
        )

        protected = self._protected_capabilities(action, context)
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
            return self._neutral()

        if judged.groundedness < self.config.grounding_threshold:
            return {
                **self._invalid(),
                "score": judged.groundedness / self.config.grounding_threshold - 1.0,
                "groundedness": judged.groundedness,
            }

        gain = max(0.0, judged.after_correctness - context.before_correctness)
        edit_quality = (judged.action_quality + judged.memory_quality) / 2
        retention = (
            sum(judged.retention_correctness) / len(judged.retention_correctness)
            if judged.retention_correctness
            else 1.0
        )
        cost = self._memory_cost(context, temp)
        score = gain * edit_quality * retention - self.config.memory_cost_weight * cost
        return {
            "score": score,
            "action_valid": 1.0,
            "after_correctness": judged.after_correctness,
            "gain": gain,
            "groundedness": judged.groundedness,
            "action_quality": judged.action_quality,
            "memory_quality": judged.memory_quality,
            "retention": retention,
            "memory_cost": cost,
            "reward_available": 1.0,
        }

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
            estimate_token_count(action.new_memory.content)
            if action.new_memory
            else 0
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
        action: MemoryEditAction,
        context: MemoryBuilderRewardContext,
    ) -> tuple[CapabilityRecord, ...]:
        passed = [
            record
            for record in context.memory.capability_ledger.values()
            if record.passed
        ]
        targets = set(action.target_node_ids)
        related = [
            record
            for record in passed
            if targets & set(record.supporting_memory_node_ids)
        ]
        ordered = [*related, *reversed(passed)]
        unique = {record.question_id: record for record in ordered}
        return tuple(unique.values())[: self.config.max_protected_questions]

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
    def _memory_cost(context: MemoryBuilderRewardContext, temp) -> float:
        growth = max(
            0,
            temp.active_token_count - context.memory.active_token_count,
        )
        evidence_tokens = sum(
            estimate_token_count(item.quote)
            for item in context.observation.new_evidence
        )
        return min(1.0, growth / max(1, evidence_tokens))

    @staticmethod
    def _invalid(score: float = -1.0) -> dict[str, float]:
        return {
            "score": score,
            "action_valid": 0.0,
            "after_correctness": 0.0,
            "gain": 0.0,
            "groundedness": 0.0,
            "action_quality": 0.0,
            "memory_quality": 0.0,
            "retention": 0.0,
            "memory_cost": 0.0,
            "reward_available": 1.0,
        }

    @staticmethod
    def _neutral() -> dict[str, float]:
        return {
            "score": 0.0,
            "action_valid": 1.0,
            "after_correctness": 0.0,
            "gain": 0.0,
            "groundedness": 0.0,
            "action_quality": 0.0,
            "memory_quality": 0.0,
            "retention": 0.0,
            "memory_cost": 0.0,
            "reward_available": 0.0,
        }
