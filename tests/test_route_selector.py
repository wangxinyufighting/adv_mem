import sys
import types
import unittest
import json
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory


# The production environment installs openai. Keep unit tests runnable in a small
# local environment because these tests inject all network-facing collaborators.
if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIError = type("APIError", (Exception,), {})
    sys.modules["openai"] = openai

from attacker.gap import GapEvaluation, GapEvaluator, GapType, novelty_values
from attacker.oracle import DeepSeekOracle
from attacker.models import (
    AttackMode,
    GraphRouteBundle,
    MemoryGraphView,
    OracleResult,
    RouteNode,
    RouteProbe,
    RouteSelectorRewardContext,
    RouterConfig,
    RouterState,
    SupportingEvidence,
)
from attacker.graph_router import GraphRouterPolicy
from attacker.probe import ProbeFactory
from attacker.reward import RouteSelectorReward
from attacker.probe_cache import ProbeCache
from attacker.reward_judge import DeepSeekRewardJudge
from attacker.selector import RouteSelector
from memory.models import MemoryNode, MemoryState
from memory.store import MemoryStore
from training.dataset_builder import RouteSelectorDatasetBuilder


class _Retriever:
    def __init__(self, nodes=()):
        self.nodes = tuple(nodes)

    def retrieve(self, question, memory, top_k=5):
        return [types.SimpleNamespace(node=node) for node in self.nodes[:top_k]]


class _AnswerAgent:
    def __init__(self, answer="INSUFFICIENT_INFORMATION"):
        self.answer = answer

    def answer_memories(self, question, memories):
        return self.answer


class _Judge:
    def __init__(self, correctness=0.0):
        self.correctness = correctness

    def evaluate(self, oracle, golden_answer, memory_answer, parametric_answer=None):
        return types.SimpleNamespace(memory_correctness=self.correctness)


def _probe() -> RouteProbe:
    evidence_node = RouteNode(
        id="fact-1",
        type="fact",
        status="activated",
        memory_type=None,
        key="destination",
        memory="The user plans to visit Kyoto.",
        background=None,
        tags=("travel",),
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
        evidence_nodes=(evidence_node,),
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


def _memory_node(provenance=True) -> MemoryNode:
    return MemoryNode(
        id="memory-1",
        content="The user plans to visit Kyoto.",
        provenance_node_ids=("fact-1",) if provenance else (),
        source_ids=("src-1",) if provenance else (),
    )


class RouteSelectorTests(unittest.TestCase):
    def test_coverage_route_anchors_the_least_visited_fact(self):
        graph = MemoryGraphView(
            0,
            "user",
            "test",
            (
                {"id": "fact-1", "type": "fact", "status": "activated", "memory": "A"},
                {"id": "fact-2", "type": "fact", "status": "activated", "memory": "B"},
            ),
            (),
        )
        state = RouterState({"fact-1": 2, "fact-2": 0})
        route = GraphRouterPolicy(RouterConfig(random_seed=0), state).coverage_route(
            graph
        )
        self.assertEqual(route.walk_node_ids, ("fact-2",))

    def test_coverage_route_includes_unvisited_temporal_history(self):
        graph = MemoryGraphView(
            0,
            "user",
            "test",
            (
                {
                    "id": "old",
                    "type": "fact",
                    "status": "archived",
                    "memory": "The old plan was Kyoto.",
                    "sources": [{"content": "Kyoto", "chat_time": "2025-01-01"}],
                },
                {
                    "id": "new",
                    "type": "fact",
                    "status": "activated",
                    "memory": "The plan changed to Osaka now.",
                    "sources": [{"content": "Osaka", "chat_time": "2025-02-01"}],
                },
            ),
            ({"source": "old", "target": "new", "type": "MERGED_TO"},),
        )
        state = RouterState({"old": 0, "new": 3})
        route = GraphRouterPolicy(RouterConfig(random_seed=0), state).coverage_route(
            graph
        )
        self.assertEqual(route.attack_mode, AttackMode.TEMPORAL_EVOLUTION)
        self.assertEqual(route.walk_node_ids, ("old", "new"))

    def test_oracle_accepts_legacy_single_rejection(self):
        class Completions:
            def create(self, **kwargs):
                message = types.SimpleNamespace(
                    content='{"id":0,"valid":false,"reason":"not_question"}'
                )
                choice = types.SimpleNamespace(
                    message=message,
                    finish_reason="stop",
                )
                return types.SimpleNamespace(choices=[choice])

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        result = DeepSeekOracle(client=client, attempts=1).evaluate(
            "Remember Kyoto",
            _probe().route,
        )
        self.assertFalse(result.valid)
        self.assertEqual(result.invalid_reason, "not_question")

    def test_choice_parser_requires_one_in_range_integer(self):
        self.assertEqual(RouteSelector.parse_choice('{"choice": 1}', 2), 1)
        for response in ('{"choice": 2}', '{"choice": true}', '{"choice": 0, "x": 1}'):
            with self.assertRaises(ValueError):
                RouteSelector.parse_choice(response, 2)

    def test_gap_types_are_separated_by_provenance_and_retrieval(self):
        probe = _probe()
        support = _memory_node()

        storage = GapEvaluator(
            _Retriever((_memory_node(False),)),
            _AnswerAgent(),
            _Judge(),
        ).evaluate(probe, MemoryState(nodes={"memory-1": _memory_node(False)}))
        self.assertEqual(storage.gap_type, GapType.STORAGE)

        retrieval = GapEvaluator(
            _Retriever(()),
            _AnswerAgent(),
            _Judge(),
        ).evaluate(probe, MemoryState(nodes={support.id: support}))
        self.assertEqual(retrieval.gap_type, GapType.RETRIEVAL)

        reasoning = GapEvaluator(
            _Retriever((support,)),
            _AnswerAgent("wrong"),
            _Judge(0.0),
        ).evaluate(probe, MemoryState(nodes={support.id: support}))
        self.assertEqual(reasoning.gap_type, GapType.REASONING)

        answered = GapEvaluator(
            _Retriever((support,)),
            _AnswerAgent("Kyoto"),
            _Judge(1.0),
        ).evaluate(probe, MemoryState(nodes={support.id: support}))
        self.assertEqual(answered.gap_type, GapType.NONE)

    def test_selector_reward_credits_route_choice(self):
        probe = _probe()
        context = RouteSelectorRewardContext.from_state(
            (probe.route,), MemoryState.empty(), cached_probes=(probe,)
        )

        class Evaluator:
            calls = 0

            def evaluate(self, selected, memory):
                self.calls += 1
                return GapEvaluation(
                    GapType.STORAGE,
                    0.0,
                    (),
                    None,
                    0.0,
                    0.0,
                )

        evaluator = Evaluator()
        reward = RouteSelectorReward(evaluator)
        self.assertGreater(reward.evaluate('{"choice": 0}', context)["score"], 0.9)
        reward.evaluate('{"choice": 0}', context)
        self.assertEqual(evaluator.calls, 1)
        self.assertEqual(reward.evaluate('{"choice": 1}', context)["score"], -1.0)

    def test_novelty_breaks_cold_start_reward_ties(self):
        first = _probe()
        second = replace(
            first,
            question_id="question-2",
            route=replace(first.route, route_id="route-2"),
            oracle=replace(first.oracle, route_id="route-2"),
        )
        context = RouteSelectorRewardContext.from_state(
            (first.route, second.route),
            MemoryState.empty(),
            (0.25, 1.0),
            (first, second),
        )

        class Evaluator:
            def evaluate(self, probe, memory):
                return GapEvaluation(GapType.STORAGE, 0.0, (), None, 0.0, 0.0)

        reward = RouteSelectorReward(Evaluator())
        low = reward.evaluate('{"choice":0}', context)["score"]
        high = reward.evaluate('{"choice":1}', context)["score"]
        self.assertGreater(high, low)

    def test_novelty_rewards_more_unseen_evidence(self):
        first = _probe()
        second = replace(
            first,
            question_id="question-2",
            route=replace(first.route, route_id="route-2"),
            oracle=replace(
                first.oracle,
                route_id="route-2",
                supporting_evidence=(
                    SupportingEvidence("src-2", "fact-2", "Osaka", None, "user"),
                    SupportingEvidence("src-3", "fact-3", "Nara", None, "user"),
                ),
            ),
        )
        low, high = novelty_values((first, second), MemoryState.empty())
        self.assertGreater(high, low)

    def test_repeat_penalty_discourages_the_same_route(self):
        probe = _probe()
        memory = MemoryState.empty()
        MemoryStore(memory).record_route_attack("route-1", GapType.STORAGE.value)
        context = RouteSelectorRewardContext.from_state(
            (probe.route,), memory, (0.0,), (probe,)
        )

        class Evaluator:
            def evaluate(self, probe, memory):
                return GapEvaluation(GapType.STORAGE, 0.0, (), None, 0.0, 0.0)

        repeated = RouteSelectorReward(Evaluator()).evaluate(
            '{"choice":0}', context
        )["score"]
        fresh = RouteSelectorReward(Evaluator()).evaluate(
            '{"choice":0}',
            RouteSelectorRewardContext.from_state(
                (probe.route,), MemoryState.empty(), (0.0,), (probe,)
            ),
        )["score"]
        self.assertLess(repeated, fresh)

    def test_semantic_judge_only_requests_supplied_answer_labels(self):
        captured = {}

        class Completions:
            def create(self, **kwargs):
                captured.update(kwargs)
                message = types.SimpleNamespace(
                    content='{"memory_correctness":"correct"}'
                )
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=message)]
                )

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(completions=Completions())
        )
        result = DeepSeekRewardJudge(client, "judge").evaluate(
            _probe().oracle,
            None,
            "Kyoto",
        )
        self.assertEqual(result.memory_correctness, 1.0)
        self.assertEqual(result.gold_correctness, 0.0)
        payload = json.loads(captured["messages"][1]["content"])
        self.assertEqual(
            payload["candidate_answers"],
            {"memory_correctness": "Kyoto"},
        )

    def test_selector_record_contains_multiple_roundtrippable_routes(self):
        first = _probe()
        second_route = replace(
            first.route,
            route_id="route-2",
            route_signature="route-2",
        )
        second = replace(
            first,
            question_id="question-2",
            route=second_route,
            oracle=replace(first.oracle, route_id="route-2"),
        )
        selector = RouteSelector()
        memory = MemoryState.empty()
        routes = (first.route, second.route)
        observation = selector.observe(routes, memory, _Retriever())
        record = selector.to_verl_record(
            observation, routes, memory, cached_probes=(first, second)
        )
        restored = RouteSelectorRewardContext.from_dict(record["extra_info"])
        self.assertEqual(len(restored.routes), 2)
        self.assertEqual(len(restored.cached_probes), 2)
        self.assertEqual(
            [item["choice"] for item in observation.to_dict()["candidates"]],
            [0, 1],
        )
        self.assertNotIn('"probe_question"', record["prompt"][1]["content"])

    def test_selector_builds_contrast_pairs_from_empty_memory(self):
        first = _probe()
        routes = tuple(
            replace(
                first.route,
                route_id=f"route-{index}",
                route_signature=f"route-{index}",
                evidence_nodes=(
                    replace(
                        first.route.evidence_nodes[0],
                        id=f"fact-{index}",
                        source_ids=(f"src-{index}",),
                    ),
                ),
            )
            for index in range(4)
        )
        memory = MemoryState(
            nodes={
                f"memory-{index}": MemoryNode(
                    f"memory-{index}",
                    "Kyoto",
                    provenance_node_ids=(f"fact-{index}",),
                    source_ids=(f"src-{index}",),
                )
                for index in range(2)
            }
        )
        records = RouteSelectorDatasetBuilder(
            _Retriever(),
        ).records(routes, memory)
        self.assertEqual(len(records), 2)
        self.assertTrue(
            all(
                record["prompt"][1]["content"].count('"choice"') == 2
                for record in records
            )
        )
        cold = RouteSelectorDatasetBuilder(_Retriever()).records(
            routes,
            MemoryState.empty(),
        )
        self.assertEqual(len(cold), 2)
        self.assertTrue(
            all(
                len(record["extra_info"]["novelty_values"]) == 2
                for record in cold
            )
        )

    def test_lazy_probe_cache_round_trip(self):
        second = replace(
            _probe(),
            question_id="question-2",
            route=replace(
                _probe().route,
                route_id="route-2",
                route_signature="route-2",
            ),
            oracle=replace(_probe().oracle, route_id="route-2"),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            first_writer = ProbeCache("test", path)
            second_writer = ProbeCache("test", path)
            first_writer.put(_probe())
            second_writer.put(second)
            restored = ProbeCache("test", path)
        self.assertEqual(restored.get(_probe().route).question_id, "question-1")
        self.assertEqual(len(restored.probes), 2)

    def test_reward_builds_only_the_selected_route_once(self):
        first = _probe()
        second = replace(
            first,
            question_id="question-2",
            route=replace(
                first.route,
                route_id="route-2",
                route_signature="route-2",
            ),
            oracle=replace(first.oracle, route_id="route-2"),
        )

        class Factory:
            calls = []

            def build(self, route):
                self.calls.append(route.route_id)
                return first if route.route_id == "route-1" else second

        class Evaluator:
            def evaluate(self, probe, memory):
                return GapEvaluation(GapType.STORAGE, 0.0, (), None, 0.0, 0.0)

        factory = Factory()
        reward = RouteSelectorReward(Evaluator(), probe_factory=factory)
        context = RouteSelectorRewardContext.from_state(
            (first.route, second.route), MemoryState.empty(), (0.0, 0.0)
        )
        reward.evaluate('{"choice":1}', context)
        reward.evaluate('{"choice":1}', context)
        self.assertEqual(factory.calls, ["route-2"])

    def test_probe_factory_logs_rejection_reasons(self):
        class Generator:
            def generate(self, route):
                return "What does the context say?"

        factory = ProbeFactory(Generator(), None, None, None, attempts=2)
        output = StringIO()
        with redirect_stdout(output):
            self.assertIsNone(factory.build(_probe().route))
        payload = json.loads(output.getvalue().removeprefix("Probe Build: "))
        self.assertFalse(payload["success"])
        self.assertEqual(
            payload["failures"], {"question_constraint/metadata_leak": 2}
        )

    def test_attack_history_is_backward_compatible_and_deduplicated(self):
        state = MemoryState.empty()
        old_payload = state.to_dict()
        old_payload.pop("attack_history")
        restored = MemoryState.from_dict(old_payload)
        self.assertEqual(restored.attack_history, {})

        store = MemoryStore(restored)
        store.record_route_attack("route-1", GapType.STORAGE.value)
        store.record_route_attack("route-1", GapType.STORAGE.value)
        self.assertEqual(store.state.attack_history["route-1"].attempts, 1)
        store.state.version += 1
        store.record_route_attack("route-1", GapType.NONE.value)
        self.assertEqual(store.state.attack_history["route-1"].attempts, 2)


if __name__ == "__main__":
    unittest.main()
