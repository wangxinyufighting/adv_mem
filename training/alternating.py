from dataclasses import dataclass
from typing import Any

from openai import APIError

from attacker.gap import GapEvaluator
from attacker.models import OracleResult, RouteProbe
from defender.controller import RepairController
from defender.memory_builder import MemoryBuilder
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from memory.models import CapabilityRecord, MemoryEvidence, MemoryState
from memory.store import MemoryStore
from training.support_attribution import SupportAttributor
from utils.json_output import StructuredOutputError


QuestionCandidate = RouteProbe


@dataclass(frozen=True)
class PendingMemoryEdit:
    candidate: RouteProbe
    capability: CapabilityRecord
    observation: MemoryBuilderObservation
    before_correctness: float
    gap_type: str


class MemoryTrainingFlow:
    """Evaluate one fixed probe, plan one repair, and guard one commit."""

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
        self.builder = builder or MemoryBuilder()
        self.top_k = top_k
        self.defense_threshold = defense_threshold
        self.commit_threshold = commit_threshold
        self.support_attributor = SupportAttributor(
            answer_agent, answer_judge, defense_threshold
        )
        self.gap_evaluator = GapEvaluator(
            retriever,
            answer_agent,
            answer_judge,
            top_k=top_k,
            correctness_threshold=defense_threshold,
        )

    def process_question(self, candidate: RouteProbe) -> PendingMemoryEdit | None:
        evaluation = self.gap_evaluator.evaluate(candidate, self.store.state)
        gap = evaluation.gap_type.value
        self.store.record_route_attack(candidate.route.route_id, gap)
        capability = self._capability(candidate, gap)
        evidence = self._evidence(candidate)

        if evaluation.correctness >= self.defense_threshold:
            support = self.support_attributor.select(
                candidate.oracle, evaluation.memories
            )
            if not support:
                raise StructuredOutputError("Could not verify supporting memories")
            self.store.mark_success(capability, support, evidence)
            return None

        plan = RepairController.plan(candidate, self.store.state)
        observation = MemoryBuilderObservation(
            memory_version=self.store.state.version,
            question_id=candidate.question_id,
            question=candidate.oracle.question,
            new_evidence=candidate.oracle.supporting_evidence,
            target_memories=tuple(
                self.store.state.nodes[node_id] for node_id in plan.target_node_ids
            ),
            plan=plan,
        )
        return PendingMemoryEdit(
            candidate, capability, observation, evaluation.correctness, gap
        )

    def try_process_question(
        self,
        candidate: RouteProbe,
    ) -> tuple[bool, PendingMemoryEdit | None]:
        try:
            return True, self.process_question(candidate)
        except (APIError, StructuredOutputError):
            return False, None

    def reward_context(
        self,
        pending: PendingMemoryEdit,
    ) -> MemoryBuilderRewardContext:
        return MemoryBuilderRewardContext.from_state(
            pending.observation,
            self.store.state,
            pending.candidate.oracle,
            pending.before_correctness,
        )

    def commit(
        self,
        pending: PendingMemoryEdit,
        content: str,
        reward: dict[str, float],
    ) -> bool:
        evidence = self._evidence(pending.candidate)
        if (
            not reward.get("commit_valid", 0.0)
            or reward["score"] <= self.commit_threshold
        ):
            self.store.mark_failure(pending.capability, evidence)
            return False

        temp = self.builder.execute(
            self.store.state,
            pending.observation,
            content,
            trusted_provenance=True,
        )
        old = tuple(
            record
            for record in self.store.state.capability_ledger.values()
            if record.passed
        )
        support = {}
        for record in old:
            node_ids = self._verified_support(self._oracle(record), temp)
            if not node_ids:
                self.store.mark_failure(pending.capability, evidence)
                return False
            support[record.question_id] = node_ids

        current = self._verified_support(pending.candidate.oracle, temp)
        if not current:
            self.store.mark_failure(pending.capability, evidence)
            return False

        committed = MemoryStore(temp)
        for record in old:
            committed.mark_success(
                record,
                support[record.question_id],
                self.store.state.evidence_ledger.get(record.question_id, ()),
            )
        committed.mark_success(pending.capability, current, evidence)
        self.store.state = committed.state
        return True

    def discard(self, pending: PendingMemoryEdit) -> None:
        self.store.mark_failure(
            pending.capability, self._evidence(pending.candidate)
        )

    def defer_question(self, candidate: RouteProbe) -> None:
        self.store.mark_failure(
            self._capability(candidate), self._evidence(candidate)
        )

    def _verified_support(
        self,
        oracle: OracleResult,
        memory: MemoryState,
    ) -> tuple[str, ...]:
        results = self.retriever.retrieve(oracle.question, memory, top_k=self.top_k)
        if not results:
            return ()
        return self.support_attributor.select(
            oracle, tuple(result.node for result in results)
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
    def _capability(
        candidate: RouteProbe,
        gap: str | None = None,
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
            discovered_gap=gap,
        )

    @staticmethod
    def _evidence(candidate: RouteProbe) -> tuple[MemoryEvidence, ...]:
        return tuple(
            MemoryEvidence(
                item.source_id,
                item.node_id,
                item.quote,
                item.chat_time,
                item.role,
            )
            for item in candidate.oracle.supporting_evidence
        )
