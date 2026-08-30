import sys
import types
import unittest
import json
from dataclasses import replace


# The production environment installs openai. Keep unit tests runnable in a small
# local environment because these tests inject all network-facing collaborators.
if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    openai.APIError = type("APIError", (Exception,), {})
    sys.modules["openai"] = openai

from attacker.gap import GapEvaluation, GapEvaluator, GapType
from attacker.oracle import DeepSeekOracle
from attacker.models import (
    AttackMode,
    GraphRouteBundle,
    OracleResult,
    RouteNode,
    RouteProbe,
    RouteSelectorRewardContext,
    SupportingEvidence,
)
from attacker.reward import RouteSelectorReward
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
        context = RouteSelectorRewardContext.from_state((probe,), MemoryState.empty())

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
        probes = (first, second)
        observation = selector.observe(probes, memory, _Retriever())
        record = selector.to_verl_record(observation, probes, memory)
        restored = RouteSelectorRewardContext.from_dict(record["extra_info"])
        self.assertEqual(len(restored.probes), 2)
        self.assertEqual(
            [item["choice"] for item in observation.to_dict()["candidates"]],
            [0, 1],
        )
        self.assertIn('"probe_question"', record["prompt"][1]["content"])

    def test_selector_shrinks_an_oversized_candidate_window(self):
        first = _probe()
        probes = tuple(
            replace(
                first,
                question_id=f"question-{index}",
                route=replace(
                    first.route,
                    route_id=f"route-{index}",
                    route_signature=f"route-{index}",
                ),
                oracle=replace(first.oracle, route_id=f"route-{index}"),
            )
            for index in range(4)
        )

        class Tokenizer:
            def apply_chat_template(self, prompt, **kwargs):
                choices = prompt[1]["content"].count('"choice"')
                return range(2000 if choices > 2 else 100)

        records = RouteSelectorDatasetBuilder(
            _Retriever(),
            candidates_per_prompt=4,
            tokenizer=Tokenizer(),
            max_prompt_tokens=1000,
        ).records(probes, MemoryState.empty())
        self.assertEqual(len(records), 2)

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
