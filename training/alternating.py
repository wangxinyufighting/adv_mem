from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from attacker.attacker import Attacker
from attacker.models import GraphRouteBundle, OracleResult
from defender.memory_builder import MemoryBuilder
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from memory.models import (
    CapabilityRecord,
    MemoryEditAction,
    MemoryEvidence,
    MemoryState,
)
from memory.store import MemoryStore


@dataclass(frozen=True)
class QuestionCandidate:
    question_id: str
    route: GraphRouteBundle
    oracle: OracleResult
    golden_answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "route": self.route.to_dict(),
            "oracle": self.oracle.to_dict(),
            "golden_answer": self.golden_answer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "QuestionCandidate":
        return cls(
            question_id=payload["question_id"],
            route=GraphRouteBundle.from_dict(payload["route"]),
            oracle=OracleResult.from_dict(payload["oracle"]),
            golden_answer=payload["golden_answer"],
        )


@dataclass(frozen=True)
class PendingMemoryEdit:
    candidate: QuestionCandidate
    capability: CapabilityRecord
    observation: MemoryBuilderObservation
    before_correctness: float


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
    ):
        self.store = store
        self.retriever = retriever
        self.answer_agent = answer_agent
        self.answer_judge = answer_judge
        self.builder = builder or MemoryBuilder()
        self.top_k = top_k
        self.defense_threshold = defense_threshold
        self.commit_threshold = commit_threshold

    def process_question(
        self,
        candidate: QuestionCandidate,
    ) -> PendingMemoryEdit | None:
        results = self.retriever.retrieve(
            candidate.oracle.question,
            self.store.state,
            top_k=self.top_k,
        )
        memories = tuple(result.node for result in results)
        memory_answer = self.answer_agent.answer_memories(
            candidate.oracle.question,
            memories,
        )
        judged = self.answer_judge.evaluate(
            candidate.oracle,
            candidate.golden_answer,
            memory_answer,
        )
        capability = self._capability(candidate)
        evidence = self._evidence(candidate)

        # Correct answers are defense successes; no memory reconstruction is needed.
        if judged.memory_correctness >= self.defense_threshold:
            self.store.mark_success(
                capability,
                tuple(node.id for node in memories),
                evidence,
            )
            return None

        observation = MemoryBuilderObservation(
            memory_version=self.store.state.version,
            question_id=candidate.question_id,
            question=candidate.oracle.question,
            new_evidence=candidate.oracle.supporting_evidence,
            memory_neighborhood=memories,
        )
        return PendingMemoryEdit(
            candidate=candidate,
            capability=capability,
            observation=observation,
            before_correctness=judged.memory_correctness,
        )

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
        if reward["score"] <= self.commit_threshold:
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
        committed = MemoryStore(temp)
        committed.mark_success(
            pending.capability,
            tuple(result.node.id for result in results),
            evidence,
        )
        self.store.state = committed.state
        return True

    def discard(self, pending: PendingMemoryEdit) -> None:
        self.store.mark_high_priority(
            pending.capability,
            self._evidence(pending.candidate),
        )

    def high_priority_questions(self) -> tuple[CapabilityRecord, ...]:
        return tuple(
            self.store.state.capability_ledger[question_id]
            for question_id in self.store.state.high_priority_buffer
        )

    @staticmethod
    def _capability(candidate: QuestionCandidate) -> CapabilityRecord:
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

        pending = tuple(
            item
            for candidate in candidates
            if (item := self.flow.process_question(candidate)) is not None
        )
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

        defended = len(candidates) - len(pending)
        committed = 0
        discarded = 0
        if builder_checkpoint:
            for item in pending:
                # Earlier commits may already make this question answerable.
                refreshed = self.flow.process_question(item.candidate)
                if refreshed is None:
                    defended += 1
                    continue
                response = generate_action(
                    builder_checkpoint,
                    refreshed.observation,
                )
                context = self.flow.reward_context(refreshed)
                reward = score_action(response, context)
                if reward["score"] <= self.flow.commit_threshold:
                    self.flow.discard(refreshed)
                    discarded += 1
                    continue
                action = self.builder.parse_action(response)
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
