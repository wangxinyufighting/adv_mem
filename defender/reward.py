from dataclasses import dataclass
from typing import Any

from openai import APIError

from attacker.answer_agent import QwenAnswerAgent, is_insufficient_answer
from attacker.models import OracleResult
from attacker.reward_judge import DeepSeekRewardJudge
from defender.memory_builder import (
    ActionConstraintError,
    ActionSchemaError,
    MemoryBuilder,
)
from defender.models import MemoryBuilderRewardContext
from defender.reward_judge import DeepSeekMemoryJudge
from memory.models import CapabilityRecord, MemoryEditAction, MemoryOperation
from memory.store import estimate_token_count
from utils.json_output import StructuredOutputError
from utils.memory_retrieval import HybridMemoryRetriever


@dataclass(frozen=True)
class MemoryBuilderRewardConfig:
    answer_threshold: float = 0.8
    retention_threshold: float = 0.8
    retrieval_top_k: int = 5
    max_memory_tokens: int = 128


class MemoryBuilderReward:
    """Reward one grounded repair that fixes the gap without regression."""

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
            action = self.builder.parse_action(
                response,
                context.observation.memory_neighborhood,
            )
        except ActionSchemaError:
            return self._result(-1.0)
        except ActionConstraintError:
            return self._result(-0.9, schema_valid=1.0)
        if not self._valid_action(action, context):
            return self._result(-0.8, schema_valid=1.0)

        temp = self.builder.execute(context.memory, context.observation, action)
        growth, shrink = self._size_change(context, temp)
        try:
            structure = self.edit_judge.evaluate(
                action,
                context.observation.new_evidence,
                context.observation.memory_neighborhood,
            )
        except (APIError, StructuredOutputError) as error:
            return self._unavailable("edit_judge", error, growth, shrink)

        try:
            after = self._correctness(context.oracle, temp)
        except (APIError, StructuredOutputError) as error:
            return self._unavailable(
                "answer", error, growth, shrink, edit_judge_available=1.0
            )

        try:
            retention = tuple(
                self._correctness(self._oracle(record), temp)
                for record in self._protected(context)
            )
        except (APIError, StructuredOutputError) as error:
            return self._unavailable(
                "retention",
                error,
                growth,
                shrink,
                edit_judge_available=1.0,
                answer_available=1.0,
            )

        grounded = float(structure.grounded)
        evidence_covered = float(structure.evidence_covered)
        targets_preserved = float(structure.targets_preserved)
        retention_min = min(retention, default=1.0)
        answer_pass = after >= self.config.answer_threshold
        retention_pass = retention_min >= self.config.retention_threshold
        structure_pass = bool(
            structure.grounded
            and structure.evidence_covered
            and structure.targets_preserved
        )
        commit_valid = structure_pass and answer_pass and retention_pass
        gain = max(0.0, after - context.before_correctness)
        if not structure_pass:
            score = -1.0
        elif not retention_pass:
            score = -0.8
        elif not answer_pass:
            score = after - self.config.answer_threshold
        else:
            score = min(1.0, 0.8 * gain + 0.1 * (1.0 - growth) + 0.1 * shrink)

        return self._result(
            score,
            schema_valid=1.0,
            action_valid=1.0,
            after_correctness=after,
            answer_pass=float(answer_pass),
            grounded=grounded,
            evidence_covered=evidence_covered,
            targets_preserved=targets_preserved,
            retention_min=retention_min,
            retention_pass=float(retention_pass),
            growth=growth,
            shrink=shrink,
            gain=gain,
            commit_valid=float(commit_valid),
            edit_judge_available=1.0,
            answer_available=1.0,
            retention_available=1.0,
        )

    def _unavailable(
        self,
        stage: str,
        error: Exception,
        growth: float,
        shrink: float,
        **values: float,
    ) -> dict[str, float]:
        print(
            f"Memory Builder Reward unavailable: stage={stage} "
            f"error={type(error).__name__}: {error}",
            flush=True,
        )
        return self._result(
            0.0,
            reward_available=0.0,
            schema_valid=1.0,
            action_valid=1.0,
            growth=growth,
            shrink=shrink,
            **values,
        )

    def _valid_action(
        self,
        action: MemoryEditAction,
        context: MemoryBuilderRewardContext,
    ) -> bool:
        observation = context.observation
        targets = set(action.target_node_ids)
        neighborhood = {node.id for node in observation.memory_neighborhood}
        support = set(observation.support_node_ids)
        tokens = (
            estimate_token_count(action.new_memory.content)
            if action.new_memory
            else 0
        )
        if not targets <= neighborhood or not 0 < tokens <= self.config.max_memory_tokens:
            return False
        if action.operation == MemoryOperation.ADD:
            return (
                observation.gap_type == "storage_gap"
                and not support
                and not targets
            )
        if action.operation != MemoryOperation.MERGE:
            return False
        return bool(support) and support <= targets

    def _correctness(self, oracle: OracleResult, memory) -> float:
        results = self.retriever.retrieve(
            oracle.question,
            memory,
            top_k=self.config.retrieval_top_k,
        )
        answer = self.answer_agent.answer_memories(
            oracle.question,
            tuple(result.node for result in results),
        )
        if is_insufficient_answer(answer):
            return 0.0
        return self.answer_judge.evaluate(oracle, None, answer).memory_correctness

    @staticmethod
    def _protected(
        context: MemoryBuilderRewardContext,
    ) -> tuple[CapabilityRecord, ...]:
        ledger = context.memory.capability_ledger
        return tuple(
            ledger[item.question_id]
            for item in context.observation.protected_questions
            if item.question_id in ledger
        )

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

    @staticmethod
    def _size_change(context: MemoryBuilderRewardContext, temp) -> tuple[float, float]:
        before = context.memory.active_token_count
        after = temp.active_token_count
        evidence = sum(
            estimate_token_count(item.quote)
            for item in context.observation.new_evidence
        )
        growth = min(1.0, max(0, after - before) / max(1, evidence))
        shrink = min(1.0, max(0, before - after) / max(1, before))
        return growth, shrink

    @staticmethod
    def _result(score: float, **values: float) -> dict[str, float]:
        result = {
            "score": score,
            "reward_available": 1.0,
            "schema_valid": 0.0,
            "action_valid": 0.0,
            "after_correctness": 0.0,
            "answer_pass": 0.0,
            "grounded": 0.0,
            "evidence_covered": 0.0,
            "targets_preserved": 0.0,
            "retention_min": 0.0,
            "retention_pass": 0.0,
            "growth": 0.0,
            "shrink": 0.0,
            "gain": 0.0,
            "commit_valid": 0.0,
            "edit_judge_available": 0.0,
            "answer_available": 0.0,
            "retention_available": 0.0,
        }
        result.update(values)
        return result
