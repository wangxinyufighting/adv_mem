import sys
import types
import unittest


if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIError = type("APIError", (Exception,), {})
    sys.modules["openai"] = openai

from attacker.models import (
    AttackMode,
    GraphRouteBundle,
    OracleResult,
    RouteNode,
    RouteProbe,
    SupportingEvidence,
)
from defender.controller import RepairController
from defender.memory_builder import ContentSchemaError, MemoryBuilder
from defender.models import MemoryBuilderObservation, MemoryBuilderRewardContext
from defender.reward import MemoryBuilderReward
from memory.models import CapabilityRecord, MemoryNode, MemoryOperation, MemoryState
from memory.store import MemoryStore


def _probe() -> RouteProbe:
    route_node = RouteNode(
        id="fact-1",
        type="fact",
        status="activated",
        memory_type=None,
        key=None,
        memory="The user plans to visit Kyoto.",
        background=None,
        tags=(),
        confidence=1.0,
        version=1,
        created_at=None,
        updated_at=None,
        source_ids=("src-1",),
    )
    route = GraphRouteBundle(
        route_id="route-1",
        graph_version="test",
        case_index=0,
        user_name="user",
        attack_mode=AttackMode.SINGLE_FACT,
        walk_node_ids=("fact-1",),
        walk_steps=(),
        evidence_nodes=(route_node,),
        connector_nodes=(),
        source_records=(),
        route_signature="route-1",
        sampling_seed=0,
        sampling_attempt=1,
    )
    oracle = OracleResult(
        route_id="route-1",
        question="Where do I plan to visit?",
        valid=True,
        answer="Kyoto",
        supporting_evidence=(
            SupportingEvidence("src-1", "fact-1", "Kyoto", None, "user"),
        ),
        invalid_reason=None,
        confidence=1.0,
    )
    return RouteProbe("question-1", route, oracle, "Kyoto")


def _observation(memory: MemoryState) -> MemoryBuilderObservation:
    probe = _probe()
    plan = RepairController.plan(probe, memory)
    return MemoryBuilderObservation(
        memory.version,
        probe.question_id,
        probe.oracle.question,
        probe.oracle.supporting_evidence,
        tuple(memory.nodes[node_id] for node_id in plan.target_node_ids),
        plan,
    )


class _Retriever:
    def retrieve(self, question, memory, top_k=5):
        return [types.SimpleNamespace(node=node) for node in memory.active_nodes[:top_k]]


class _AnswerAgent:
    def answer_memories(self, question, memories):
        return "Kyoto" if any("Kyoto" in node.content for node in memories) else "wrong"


class _AnswerJudge:
    def evaluate(self, oracle, golden_answer, memory_answer, parametric_answer=None):
        return types.SimpleNamespace(
            memory_correctness=float(memory_answer == oracle.answer)
        )


class MemoryBuilderTests(unittest.TestCase):
    def test_success_and_high_priority_pools_are_state_guards(self):
        node = MemoryNode("memory-1", "Kyoto")
        store = MemoryStore(MemoryState(nodes={node.id: node}))

        def capability(question_id):
            return CapabilityRecord(
                question_id,
                "Where?",
                f"route-{question_id}",
                "single_fact",
                "Kyoto",
            )

        store.mark_high_priority(capability("q1"))
        store.mark_high_priority(capability("q2"))
        store.mark_high_priority(capability("q1"))
        self.assertEqual(store.state.high_priority_buffer, ["q2", "q1"])

        store.mark_success(capability("q2"), (node.id,))
        self.assertEqual(store.state.success_pool, ["q2"])
        self.assertEqual(store.state.high_priority_buffer, ["q1"])

        restored = MemoryState.from_dict(store.state.to_dict())
        self.assertEqual(restored.success_pool, ["q2"])
        self.assertEqual(restored.high_priority_buffer, ["q1"])

    def test_controller_adds_when_provenance_is_absent(self):
        plan = RepairController.plan(_probe(), MemoryState.empty())
        self.assertEqual(plan.operation, MemoryOperation.ADD)
        self.assertEqual(plan.target_node_ids, ())

    def test_controller_merges_all_provenance_matches(self):
        first = MemoryNode(
            "memory-1", "Kyoto", provenance_node_ids=("fact-1",)
        )
        second = MemoryNode("memory-2", "Kyoto", source_ids=("src-1",))
        unrelated = MemoryNode("memory-3", "Osaka")
        state = MemoryState(
            nodes={node.id: node for node in (first, second, unrelated)}
        )
        plan = RepairController.plan(_probe(), state)
        self.assertEqual(plan.operation, MemoryOperation.MERGE)
        self.assertEqual(plan.target_node_ids, ("memory-1", "memory-2"))

    def test_builder_accepts_only_one_content_field(self):
        builder = MemoryBuilder()
        self.assertEqual(builder.parse_content('{"content":"Kyoto"}'), "Kyoto")
        for response in (
            'prefix {"content":"Kyoto"}',
            '{"content":""}',
            '{"content":"Kyoto","operation":"add"}',
        ):
            with self.assertRaises(ContentSchemaError):
                builder.parse_content(response)

    def test_builder_executes_the_controller_plan(self):
        target = MemoryNode(
            "memory-1",
            "The user likes travel.",
            provenance_node_ids=("fact-0",),
            source_ids=("src-0",),
        )
        state = MemoryState(nodes={target.id: target})
        observation = _observation(state)
        temp = MemoryBuilder().execute(
            state,
            observation,
            "The user plans to visit Kyoto.",
            trusted_provenance=True,
        )
        node = temp.active_nodes[-1]
        self.assertEqual(observation.plan.operation, MemoryOperation.ADD)
        self.assertEqual(node.provenance_node_ids, ("fact-1",))
        self.assertEqual(state.version, 0)

    def test_observation_round_trip_keeps_plan(self):
        target = MemoryNode(
            "memory-1", "Kyoto", provenance_node_ids=("fact-1",)
        )
        observation = _observation(MemoryState(nodes={target.id: target}))
        restored = MemoryBuilderObservation.from_dict(observation.to_dict())
        self.assertEqual(restored, observation)
        self.assertEqual(
            restored.to_prompt_dict()["operation"], MemoryOperation.MERGE.value
        )

    def test_reward_is_gain_minus_length_when_grounded(self):
        observation = _observation(MemoryState.empty())
        context = MemoryBuilderRewardContext.from_state(
            observation,
            MemoryState.empty(),
            _probe().oracle,
            0.0,
        )
        edit_judge = types.SimpleNamespace(
            evaluate=lambda *args: types.SimpleNamespace(valid=True)
        )
        reward = MemoryBuilderReward(
            MemoryBuilder(),
            _AnswerAgent(),
            _Retriever(),
            _AnswerJudge(),
            edit_judge,
        ).evaluate('{"content":"The user plans to visit Kyoto."}', context)
        self.assertEqual(reward["gain"], 1.0)
        self.assertEqual(reward["regression"], 0.0)
        self.assertGreater(reward["score"], 0.9)
        self.assertEqual(reward["commit_valid"], 1.0)

    def test_grounding_is_a_hard_gate(self):
        observation = _observation(MemoryState.empty())
        context = MemoryBuilderRewardContext.from_state(
            observation, MemoryState.empty(), _probe().oracle, 0.0
        )
        edit_judge = types.SimpleNamespace(
            evaluate=lambda *args: types.SimpleNamespace(valid=False)
        )
        reward = MemoryBuilderReward(
            MemoryBuilder(),
            _AnswerAgent(),
            _Retriever(),
            _AnswerJudge(),
            edit_judge,
        ).evaluate('{"content":"Kyoto"}', context)
        self.assertEqual(reward["score"], -1.0)
        self.assertEqual(reward["commit_valid"], 0.0)


if __name__ == "__main__":
    unittest.main()
