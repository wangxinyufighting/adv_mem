import sys
import types
import unittest


if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIError = type("APIError", (Exception,), {})
    sys.modules["openai"] = openai

from attacker.models import OracleResult, SupportingEvidence
from defender.memory_builder import MemoryBuilder
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from defender.reward import MemoryBuilderReward
from defender.reward_judge import DeepSeekMemoryJudge
from memory.models import (
    CapabilityRecord,
    MemoryDraft,
    MemoryEditAction,
    MemoryNode,
    MemoryOperation,
    MemoryState,
    MemoryStatus,
)
from memory.store import MemoryStore
from training.alternating import MemoryTrainingFlow, PendingMemoryEdit
from training.stop_condition import CompactionAuditor, StopCondition, StopConfig
from training.support_attribution import SupportAttributor


class _Retriever:
    def retrieve(self, question, memory, top_k=5):
        return [
            types.SimpleNamespace(node=node)
            for node in memory.active_nodes[:top_k]
        ]


class _AnswerAgent:
    def answer_memories(self, question, memories):
        text = " ".join(node.content for node in memories)
        expected = "Paris" if "live" in question else "Kyoto"
        return expected if expected in text else "wrong"


class _AnswerJudge:
    def evaluate(self, oracle, golden_answer, memory_answer, parametric_answer=None):
        return types.SimpleNamespace(
            memory_correctness=float(memory_answer == oracle.answer)
        )


class _Policy:
    def generate(self, prompt, max_tokens):
        return '{"operation":"merge","targets":[0,1],"content":"short"}'


class _EditJudge:
    grounded = False
    evidence_covered = True
    targets_preserved = True

    def evaluate(self, action, evidence, neighborhood):
        return self


class _ValidEditJudge(_EditJudge):
    grounded = True


class _Completions:
    def create(self, **kwargs):
        content = (
            '{"grounded":true,"evidence_covered":true,'
            '"targets_preserved":true}'
        )
        message = types.SimpleNamespace(content=content)
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=message)]
        )


def _oracle(question: str, answer: str, source: str) -> OracleResult:
    return OracleResult(
        route_id=f"route-{source}",
        question=question,
        valid=True,
        answer=answer,
        supporting_evidence=(
            SupportingEvidence(source, f"fact-{source}", answer, "2025-01-01", "user"),
        ),
        invalid_reason=None,
        confidence=1.0,
    )


def _observation(
    oracle: OracleResult,
    neighborhood=(),
    gap="storage_gap",
    support=(),
) -> MemoryBuilderObservation:
    return MemoryBuilderObservation(
        memory_version=0,
        question_id=f"question-{oracle.route_id}",
        question=oracle.question,
        gap_type=gap,
        new_evidence=oracle.supporting_evidence,
        memory_neighborhood=tuple(neighborhood),
        support_node_ids=tuple(support),
    )


class MemoryBuilderTests(unittest.TestCase):
    def test_memory_judge_has_only_three_boolean_decisions(self):
        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=_Completions())
        )
        action = MemoryEditAction(
            MemoryOperation.ADD,
            new_memory=MemoryDraft("Kyoto"),
        )
        judged = DeepSeekMemoryJudge(client, "judge").evaluate(action, (), ())
        self.assertTrue(judged.grounded)
        self.assertTrue(judged.evidence_covered)
        self.assertTrue(judged.targets_preserved)

    def test_gap_type_constrains_repairs(self):
        node = MemoryNode("support", "Kyoto", source_ids=("new",))
        oracle = _oracle("Where will I visit?", "Kyoto", "new")
        reward = MemoryBuilderReward(
            MemoryBuilder(),
            _AnswerAgent(),
            _Retriever(),
            _AnswerJudge(),
            object(),
        )
        retrieval = _observation(
            oracle,
            (node,),
            "retrieval_gap",
            (node.id,),
        )
        context = MemoryBuilderRewardContext.from_state(
            retrieval,
            MemoryState(nodes={node.id: node}),
            oracle,
            0.0,
        )
        add = MemoryEditAction(
            MemoryOperation.ADD,
            new_memory=MemoryDraft("Kyoto"),
        )
        merge = MemoryEditAction(
            MemoryOperation.MERGE,
            (node.id,),
            MemoryDraft("Kyoto"),
        )
        self.assertFalse(reward._valid_action(add, context))
        self.assertTrue(reward._valid_action(merge, context))

    def test_provenance_is_deterministic_and_only_inherited_when_trusted(self):
        target = MemoryNode(
            "old",
            "The user lived in Paris.",
            source_ids=("old-source",),
            provenance_node_ids=("old-fact",),
            time_span=("2020-01-01", "2020-01-01"),
        )
        state = MemoryState(nodes={target.id: target})
        oracle = _oracle("Where will I visit?", "Kyoto", "new-source")
        observation = _observation(
            oracle,
            (target,),
            "retrieval_gap",
            (target.id,),
        )
        action = MemoryEditAction(
            MemoryOperation.MERGE,
            (target.id,),
            MemoryDraft("The user lived in Paris and will visit Kyoto."),
        )
        builder = MemoryBuilder()
        untrusted = builder.execute(state, observation, action)
        trusted = builder.execute(
            state,
            observation,
            action,
            trusted_provenance=True,
        )
        left = untrusted.active_nodes[0]
        right = trusted.active_nodes[0]
        self.assertEqual(left.id, right.id)
        self.assertEqual(left.source_ids, ())
        self.assertEqual(set(right.source_ids), {"old-source", "new-source"})
        self.assertEqual(right.time_span, ("2020-01-01", "2025-01-01"))

    def test_reward_context_keeps_only_local_capabilities(self):
        oracle = _oracle("Where will I visit?", "Kyoto", "new")
        observation = _observation(oracle)
        unrelated = CapabilityRecord(
            "old-question",
            "Where did I live?",
            "old-route",
            "single_fact",
            "Paris",
            passed=True,
        )
        memory = MemoryState(
            nodes={
                "archived": MemoryNode(
                    "archived",
                    "obsolete",
                    status=MemoryStatus.ARCHIVED,
                )
            },
            capability_ledger={unrelated.question_id: unrelated},
            success_pool=[unrelated.question_id],
        )
        compact = MemoryBuilderRewardContext.from_state(
            observation,
            memory,
            oracle,
            0.0,
        ).memory
        self.assertEqual(compact.capability_ledger, {})
        self.assertEqual(compact.success_pool, [])
        self.assertEqual(compact.nodes, {})

    def test_commit_rolls_back_when_any_known_capability_regresses(self):
        old_node = MemoryNode(
            "old",
            "The user lived in Paris.",
            linked_questions=("old-question",),
            token_count=5,
        )
        old_record = CapabilityRecord(
            question_id="old-question",
            question="Where did I live?",
            route_id="old-route",
            attack_mode="single_fact",
            oracle_answer="Paris",
            supporting_memory_node_ids=(old_node.id,),
            passed=True,
        )
        state = MemoryState(
            nodes={old_node.id: old_node},
            capability_ledger={old_record.question_id: old_record},
            success_pool=[old_record.question_id],
        )
        oracle = _oracle("Where will I visit?", "Kyoto", "new")
        observation = _observation(oracle, (old_node,))
        capability = CapabilityRecord(
            question_id="new-question",
            question=oracle.question,
            route_id=oracle.route_id,
            attack_mode="single_fact",
            oracle_answer="Kyoto",
        )
        candidate = types.SimpleNamespace(
            oracle=oracle,
            golden_answer="Kyoto",
            question_id=capability.question_id,
        )
        pending = PendingMemoryEdit(
            candidate,
            capability,
            observation,
            0.0,
            "storage_gap",
        )
        flow = MemoryTrainingFlow(
            MemoryStore(state),
            _Retriever(),
            _AnswerAgent(),
            _AnswerJudge(),
        )
        destructive = MemoryEditAction(
            MemoryOperation.MERGE,
            (old_node.id,),
            MemoryDraft("The user will visit Kyoto."),
        )
        committed = flow.commit(
            pending,
            destructive,
            {"commit_valid": 1.0, "score": 1.0},
        )
        self.assertFalse(committed)
        self.assertEqual(flow.store.state.nodes[old_node.id].status.value, "active")
        self.assertIn(capability.question_id, flow.store.state.high_priority_buffer)
        self.assertTrue(flow.store.state.capability_ledger[old_record.question_id].passed)

    def test_support_attribution_rejects_an_incorrect_full_set(self):
        oracle = _oracle("Where will I visit?", "Kyoto", "new")
        node = MemoryNode("wrong", "The user will visit Osaka.")
        selected = SupportAttributor(
            _AnswerAgent(),
            _AnswerJudge(),
        ).select(oracle, (node,))
        self.assertEqual(selected, ())

    def test_compaction_rejects_a_structurally_invalid_edit(self):
        left = MemoryNode("left", "The user lived in Paris.", token_count=5)
        right = MemoryNode("right", "The user visits Kyoto.", token_count=5)
        memory = MemoryState(nodes={left.id: left, right.id: right})
        audit = CompactionAuditor(
            MemoryBuilder(),
            _Retriever(),
            _AnswerAgent(),
            _AnswerJudge(),
            _EditJudge(),
            StopConfig(max_neighborhoods=8),
        ).audit(_Policy(), memory)
        self.assertFalse(audit.compressed)
        self.assertIs(audit.memory, memory)

    def test_compaction_checks_every_passed_capability(self):
        left = MemoryNode(
            "left",
            "The user lived in Paris.",
            linked_questions=("old-question",),
            token_count=5,
        )
        right = MemoryNode("right", "The user visits Kyoto.", token_count=5)
        record = CapabilityRecord(
            "old-question",
            "Where did I live?",
            "old-route",
            "single_fact",
            "Paris",
            passed=True,
        )
        memory = MemoryState(
            nodes={left.id: left, right.id: right},
            capability_ledger={record.question_id: record},
        )
        audit = CompactionAuditor(
            MemoryBuilder(),
            _Retriever(),
            _AnswerAgent(),
            _AnswerJudge(),
            _ValidEditJudge(),
            StopConfig(max_neighborhoods=8),
        ).audit(_Policy(), memory)
        self.assertFalse(audit.compressed)

    def test_stop_condition_depends_only_on_unresolved_gaps(self):
        stop = StopCondition(StopConfig(patience=2))
        self.assertFalse(stop.update(True))
        self.assertTrue(stop.update(True))
        self.assertFalse(stop.update(False))


if __name__ == "__main__":
    unittest.main()
