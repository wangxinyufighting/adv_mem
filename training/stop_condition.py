from dataclasses import dataclass
from itertools import combinations
from typing import Any

from attacker.answer_agent import is_insufficient_answer
from defender.memory_builder import MemoryBuilder
from defender.models import ProtectedQuestion
from memory.models import MemoryOperation, MemoryState
from memory.store import MemoryStore
from training.alternating import QuestionCandidate
from training.support_attribution import SupportAttributor
from utils.json_output import StructuredOutputError


@dataclass(frozen=True)
class StopConfig:
    patience: int = 2
    min_valid_questions: int = 4
    max_neighborhoods: int = 8
    max_global_questions: int = 4
    defense_threshold: float = 0.8
    regression_tolerance: float = 0.05


@dataclass
class StopState:
    converged_rounds: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"converged_rounds": self.converged_rounds}

    @classmethod
    def from_dict(cls, payload: dict) -> "StopState":
        return cls(payload.get("converged_rounds", 0))


class StopCondition:
    """Stop after both policies are saturated for several consecutive rounds."""

    def __init__(
        self,
        config: StopConfig | None = None,
        state: StopState | None = None,
    ):
        self.config = config or StopConfig()
        self.state = state or StopState()

    def update(self, attacker_saturated: bool, builder_saturated: bool) -> bool:
        self.state.converged_rounds = (
            self.state.converged_rounds + 1
            if attacker_saturated and builder_saturated
            else 0
        )
        return self.state.converged_rounds >= self.config.patience


@dataclass(frozen=True)
class CompactionAudit:
    memory: MemoryState
    compressed: bool
    attempts: int


class CompactionAuditor:
    """Try one smaller memory and reject it if any protected answer regresses."""

    def __init__(
        self,
        builder: MemoryBuilder,
        retriever: Any,
        answer_agent: Any,
        answer_judge: Any,
        questions: dict[str, QuestionCandidate],
        config: StopConfig,
    ):
        self.builder = builder
        self.retriever = retriever
        self.answer_agent = answer_agent
        self.answer_judge = answer_judge
        self.questions = questions
        self.config = config
        self.support_attributor = SupportAttributor(
            answer_agent,
            answer_judge,
            config.defense_threshold,
        )

    def audit(self, policy: Any, memory: MemoryState) -> CompactionAudit:
        attempts = 0
        for neighborhood in self._neighborhoods(memory):
            attempts += 1
            visible_question_ids = self._protected_questions(
                memory,
                tuple(node.id for node in neighborhood),
            )
            response = policy.generate(
                self.builder.build_compaction_prompt(
                    neighborhood,
                    tuple(
                        ProtectedQuestion.from_capability(
                            memory.capability_ledger[question_id]
                        )
                        for question_id in visible_question_ids
                    ),
                ),
                512,
            )
            try:
                action = self.builder.parse_action(response, neighborhood)
                if not self._valid(action, neighborhood):
                    continue
                temp = self.builder.compact(memory, action)
            except (ValueError, KeyError, TypeError):
                continue

            if temp.active_token_count >= memory.active_token_count:
                continue
            protected = self._protected_questions(memory, action.target_node_ids)
            try:
                if self._no_regression(memory, temp, protected):
                    return CompactionAudit(
                        self._refresh_links(temp, protected),
                        compressed=True,
                        attempts=attempts,
                    )
            except StructuredOutputError:
                continue
        return CompactionAudit(memory, compressed=False, attempts=attempts)

    def _neighborhoods(self, memory: MemoryState) -> tuple[tuple, ...]:
        nodes = memory.active_nodes
        pairs = list(combinations(nodes, 2))
        pairs.sort(
            key=lambda pair: bool(set(pair[0].tags) & set(pair[1].tags)),
            reverse=True,
        )
        return tuple([*pairs, *((node,) for node in nodes)])[
            : self.config.max_neighborhoods
        ]

    def _protected_questions(
        self,
        memory: MemoryState,
        target_ids: tuple[str, ...],
    ) -> tuple[str, ...]:
        linked = (
            question_id
            for node_id in target_ids
            for question_id in memory.nodes[node_id].linked_questions
        )
        global_sample = memory.success_pool[: self.config.max_global_questions]
        return tuple(
            question_id
            for question_id in dict.fromkeys((*linked, *global_sample))
            if question_id in self.questions
        )

    def _no_regression(
        self,
        before: MemoryState,
        after: MemoryState,
        question_ids: tuple[str, ...],
    ) -> bool:
        for question_id in question_ids:
            candidate = self.questions[question_id]
            before_score, _ = self._answer(candidate, before)
            after_score, _ = self._answer(candidate, after)
            if (
                after_score < self.config.defense_threshold
                or before_score - after_score > self.config.regression_tolerance
            ):
                return False
        return True

    def _refresh_links(
        self,
        memory: MemoryState,
        question_ids: tuple[str, ...],
    ) -> MemoryState:
        store = MemoryStore(memory)
        for question_id in question_ids:
            candidate = self.questions[question_id]
            _, node_ids = self._answer(candidate, store.state, attribute=True)
            record = store.state.capability_ledger[question_id]
            evidence = store.state.evidence_ledger.get(question_id, ())
            store.mark_success(record, node_ids, evidence)
        return store.state

    def _answer(
        self,
        candidate: QuestionCandidate,
        memory: MemoryState,
        attribute: bool = False,
    ) -> tuple[float, tuple[str, ...]]:
        results = self.retriever.retrieve(
            candidate.oracle.question,
            memory,
            top_k=5,
        )
        if not results:
            return 0.0, ()
        answer = self.answer_agent.answer_memories(
            candidate.oracle.question,
            tuple(result.node for result in results),
        )
        if is_insufficient_answer(answer):
            return 0.0, ()
        judged = self.answer_judge.evaluate(
            candidate.oracle,
            None,
            answer,
        )
        memories = tuple(result.node for result in results)
        node_ids = tuple(node.id for node in memories)
        if attribute and judged.memory_correctness >= self.config.defense_threshold:
            node_ids = self.support_attributor.select(
                candidate.oracle,
                candidate.golden_answer,
                memories,
            )
        return judged.memory_correctness, node_ids

    @staticmethod
    def _valid(action, neighborhood: tuple) -> bool:
        targets = set(action.target_node_ids)
        if not targets or not targets <= {node.id for node in neighborhood}:
            return False
        if action.operation == MemoryOperation.DELETE:
            return action.new_memory is None
        return (
            action.operation == MemoryOperation.MERGE
            and len(targets) >= 2
            and action.new_memory is not None
        )
