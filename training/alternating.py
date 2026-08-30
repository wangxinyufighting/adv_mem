from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from openai import APIError

from attacker.gap import GapEvaluator
from attacker.attacker import Attacker
from attacker.models import RouteProbe
from defender.memory_builder import MemoryBuilder
from defender.models import (
    MemoryBuilderObservation,
    MemoryBuilderRewardContext,
    ProtectedQuestion,
)
from memory.models import (
    CapabilityRecord,
    MemoryEditAction,
    MemoryEvidence,
    MemoryNode,
    MemoryState,
)
from memory.store import MemoryStore
from training.support_attribution import SupportAttributor
from utils.json_output import StructuredOutputError


# Compatibility name retained for the memory-builder and old run-state files.
QuestionCandidate = RouteProbe


@dataclass(frozen=True)
class PendingMemoryEdit:
    candidate: QuestionCandidate
    capability: CapabilityRecord
    observation: MemoryBuilderObservation
    before_correctness: float
    gap_type: str


@dataclass(frozen=True)
class AlternatingRoundResult:
    attacker_checkpoint: str
    builder_checkpoint: str | None
    defended: int
    committed: int
    discarded: int


class MemoryTrainingFlow:
    """State transitions shared by collection, GRPO reward, and commit."""

    def __init__(
        self,
        store: MemoryStore,
        retriever: Any,
        answer_agent: Any,
        answer_judge: Any,
        builder: MemoryBuilder | None = None,
        top_k: int = 5,
        defense_threshold: float = 0.8,
        commit_threshold: float = 0.0,
        max_protected_questions: int = 3,
    ):
        self.store = store
        self.retriever = retriever
        self.answer_agent = answer_agent
        self.answer_judge = answer_judge
        self.builder = builder or MemoryBuilder()
        self.top_k = top_k
        self.defense_threshold = defense_threshold
        self.commit_threshold = commit_threshold
        self.max_protected_questions = max_protected_questions
        self.support_attributor = SupportAttributor(
            answer_agent,
            answer_judge,
            defense_threshold,
        )
        self.gap_evaluator = GapEvaluator(
            retriever,
            answer_agent,
            answer_judge,
            top_k=top_k,
            correctness_threshold=defense_threshold,
        )

    def process_question(
        self,
        candidate: QuestionCandidate,
    ) -> PendingMemoryEdit | None:
        evaluation = self.gap_evaluator.evaluate(candidate, self.store.state)
        memories = evaluation.memories
        correctness = evaluation.correctness
        self.store.record_route_attack(
            candidate.route.route_id,
            evaluation.gap_type.value,
        )
        capability = self._capability(candidate, evaluation.gap_type.value)
        evidence = self._evidence(candidate)

        # Correct answers are defense successes; no memory reconstruction is needed.
        if correctness >= self.defense_threshold:
            supporting_ids = self.support_attributor.select(
                candidate.oracle,
                candidate.golden_answer,
                memories,
            )
            self.store.mark_success(
                capability,
                supporting_ids,
                evidence,
            )
            return None

        observation = MemoryBuilderObservation(
            memory_version=self.store.state.version,
            question_id=candidate.question_id,
            question=candidate.oracle.question,
            new_evidence=candidate.oracle.supporting_evidence,
            memory_neighborhood=memories,
            protected_questions=self._protected_questions(memories),
        )
        return PendingMemoryEdit(
            candidate=candidate,
            capability=capability,
            observation=observation,
            before_correctness=correctness,
            gap_type=evaluation.gap_type.value,
        )

    def try_process_question(
        self,
        candidate: QuestionCandidate,
    ) -> tuple[bool, PendingMemoryEdit | None]:
        """Return availability separately so Judge failure is not a defense result."""
        try:
            return True, self.process_question(candidate)
        except (APIError, StructuredOutputError):
            return False, None

    def reward_context(
        self,
        pending: PendingMemoryEdit,
    ) -> MemoryBuilderRewardContext:
        return MemoryBuilderRewardContext(
            observation=pending.observation,
            memory=self.store.snapshot(),
            oracle=pending.candidate.oracle,
            before_correctness=pending.before_correctness,
        )

    def commit(
        self,
        pending: PendingMemoryEdit,
        action: MemoryEditAction,
        reward: dict[str, float],
    ) -> bool:
        evidence = self._evidence(pending.candidate)
        if (
            not reward.get("commit_valid", 0.0)
            or reward["score"] <= self.commit_threshold
        ):
            self.store.mark_high_priority(pending.capability, evidence)
            return False

        temp = self.builder.execute(
            self.store.state,
            pending.observation,
            action,
        )
        results = self.retriever.retrieve(
            pending.capability.question,
            temp,
            top_k=self.top_k,
        )
        if not results:
            self.store.mark_high_priority(pending.capability, evidence)
            return False
        supporting_ids = self.support_attributor.select(
            pending.candidate.oracle,
            pending.candidate.golden_answer,
            tuple(result.node for result in results),
        )
        committed = MemoryStore(temp)
        committed.mark_success(
            pending.capability,
            supporting_ids,
            evidence,
        )
        self.store.state = committed.state
        return True

    def discard(self, pending: PendingMemoryEdit) -> None:
        self.store.mark_high_priority(
            pending.capability,
            self._evidence(pending.candidate),
        )

    def defer_question(self, candidate: QuestionCandidate) -> None:
        self.store.mark_high_priority(
            self._capability(candidate),
            self._evidence(candidate),
        )

    def high_priority_questions(self) -> tuple[CapabilityRecord, ...]:
        return tuple(
            self.store.state.capability_ledger[question_id]
            for question_id in self.store.state.high_priority_buffer
        )

    def _protected_questions(
        self,
        memories: tuple[MemoryNode, ...],
    ) -> tuple[ProtectedQuestion, ...]:
        linked = {
            question_id
            for memory in memories
            for question_id in memory.linked_questions
        }
        passed = [
            record
            for record in self.store.state.capability_ledger.values()
            if record.passed
        ]
        related = [
            record for record in reversed(passed) if record.question_id in linked
        ]
        selected = (related or list(reversed(passed)))[
            : self.max_protected_questions
        ]
        return tuple(ProtectedQuestion.from_capability(record) for record in selected)

    @staticmethod
    def _capability(
        candidate: QuestionCandidate,
        discovered_gap: str | None = None,
    ) -> CapabilityRecord:
        oracle = candidate.oracle
        return CapabilityRecord(
            question_id=candidate.question_id,
            question=oracle.question,
            route_id=candidate.route.route_id,
            attack_mode=candidate.route.attack_mode.value,
            oracle_answer=oracle.answer,
            oracle_source_ids=tuple(
                item.source_id for item in oracle.supporting_evidence
            ),
            discovered_gap=discovered_gap,
        )

    @staticmethod
    def _evidence(candidate: QuestionCandidate) -> tuple[MemoryEvidence, ...]:
        return tuple(
            MemoryEvidence(
                source_id=item.source_id,
                node_id=item.node_id,
                quote=item.quote,
                chat_time=item.chat_time,
                role=item.role,
            )
            for item in candidate.oracle.supporting_evidence
        )


class AlternatingTrainer:
    """One block-coordinate round: Attacker, Builder, then guarded commit."""

    def __init__(
        self,
        flow: MemoryTrainingFlow,
        attacker: Attacker | None = None,
        builder: MemoryBuilder | None = None,
    ):
        self.flow = flow
        self.attacker = attacker or Attacker()
        self.builder = builder or MemoryBuilder()

    def run_round(
        self,
        train_attacker: Callable[[MemoryState], str],
        collect_questions: Callable[
            [str, MemoryState, tuple[CapabilityRecord, ...]],
            tuple[QuestionCandidate, ...],
        ],
        train_builder: Callable[[tuple[dict[str, Any], ...]], str],
        generate_action: Callable[[str, MemoryBuilderObservation], str],
        score_action: Callable[[str, MemoryBuilderRewardContext], dict[str, float]],
    ) -> AlternatingRoundResult:
        attacker_checkpoint = train_attacker(self.flow.store.snapshot())
        candidates = collect_questions(
            attacker_checkpoint,
            self.flow.store.snapshot(),
            self.flow.high_priority_questions(),
        )

        evaluations = tuple(
            self.flow.try_process_question(candidate) for candidate in candidates
        )
        pending = tuple(
            item
            for available, item in evaluations
            if available and item is not None
        )
        for candidate, (available, _) in zip(
            candidates,
            evaluations,
            strict=True,
        ):
            if not available:
                self.flow.defer_question(candidate)
        builder_records = tuple(
            self.builder.to_verl_record(
                item.observation,
                self.flow.store.state,
                item.candidate.oracle,
                item.before_correctness,
            )
            for item in pending
        )
        builder_checkpoint = train_builder(builder_records) if builder_records else None

        defended = sum(available and item is None for available, item in evaluations)
        committed = 0
        discarded = 0
        if builder_checkpoint:
            for item in pending:
                # Earlier commits may already make this question answerable.
                available, refreshed = self.flow.try_process_question(item.candidate)
                if not available:
                    self.flow.defer_question(item.candidate)
                    continue
                if refreshed is None:
                    defended += 1
                    continue
                response = generate_action(
                    builder_checkpoint,
                    refreshed.observation,
                )
                context = self.flow.reward_context(refreshed)
                reward = score_action(response, context)
                if not reward.get("reward_available", 1.0):
                    self.flow.defer_question(refreshed.candidate)
                    continue
                if (
                    not reward.get("commit_valid", 0.0)
                    or reward["score"] <= self.flow.commit_threshold
                ):
                    self.flow.discard(refreshed)
                    discarded += 1
                    continue
                action = self.builder.parse_action(
                    response,
                    refreshed.observation.memory_neighborhood,
                )
                if self.flow.commit(refreshed, action, reward):
                    committed += 1
                else:
                    discarded += 1

        self.flow.store.advance_iteration()
        return AlternatingRoundResult(
            attacker_checkpoint=attacker_checkpoint,
            builder_checkpoint=builder_checkpoint,
            defended=defended,
            committed=committed,
            discarded=discarded,
        )
