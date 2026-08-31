import hashlib
import json
import os
from dataclasses import replace
from typing import Any

from openai import APIError, OpenAI

from attacker.answer_agent import is_insufficient_answer
from attacker.models import AttackMode, GraphRouteBundle, RouteProbe
from attacker.validation import (
    answer_is_leaked,
    ensure_question_mark,
    question_constraint_error,
    route_fidelity,
)
from utils.json_output import StructuredOutputError, clean_model_output


SYSTEM_PROMPT = """Write one natural, standalone question whose answer is determined
by target. Use I/my for the user's life and you for an earlier assistant response.
Do not reveal the answer, invent facts, mention the input, or join unrelated details.
Return exactly one question on one line, ending with a question mark."""

MODE_PROMPTS = {
    AttackMode.SINGLE_FACT: "Ask for one specific target detail.",
    AttackMode.SAME_TOPIC: "Ask one question that requires at least two target facts.",
    AttackMode.TEMPORAL_EVOLUTION: "Ask about the earlier and later target states.",
    AttackMode.COMPARISON: "Compare both targets on the supplied dimension.",
}


class FixedProbeQuestionGenerator:
    """Frozen generator used once per route; its output is cached as a probe."""

    def __init__(
        self,
        client: Any,
        model: str,
        max_tokens: int = 128,
    ):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    @classmethod
    def from_env(cls) -> "FixedProbeQuestionGenerator":
        return cls(
            client=OpenAI(
                api_key=os.getenv("PROBE_GENERATOR_API_KEY")
                or os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv("PROBE_GENERATOR_API_BASE")
                or os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com"),
            ),
            model=os.getenv("PROBE_GENERATOR_MODEL")
            or os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            max_tokens=int(os.getenv("PROBE_GENERATOR_MAX_TOKENS", "128")),
        )

    def generate(self, route: GraphRouteBundle) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": f"{SYSTEM_PROMPT}\n\n{MODE_PROMPTS[route.attack_mode]}",
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        route.to_attacker_context(), ensure_ascii=False
                    )
                    + "\n\n/no_think",
                },
            ],
            temperature=0.7,
            max_tokens=self.max_tokens,
        )
        question = clean_model_output(response.choices[0].message.content or "")
        if not question or "\n" in question:
            raise ValueError("Invalid question output")
        return question


class ProbeFactory:
    """Create and validate one reusable question for each proposed route."""

    def __init__(
        self,
        generator: Any,
        oracle: Any,
        answer_agent: Any,
        judge: Any,
        attempts: int = 3,
        route_threshold: float = 0.8,
        correctness_threshold: float = 0.8,
    ):
        self.generator = generator
        self.oracle = oracle
        self.answer_agent = answer_agent
        self.judge = judge
        self.attempts = attempts
        self.route_threshold = route_threshold
        self.correctness_threshold = correctness_threshold

    def build_many(
        self,
        routes: tuple[GraphRouteBundle, ...],
        cached: tuple[RouteProbe, ...] = (),
    ) -> tuple[RouteProbe, ...]:
        by_route = {probe.route.route_id: probe for probe in cached}
        probes = []
        for route in routes:
            probe = by_route.get(route.route_id) or self.build(route)
            if probe is not None:
                probes.append(probe)
        return tuple(probes)

    def build(self, route: GraphRouteBundle) -> RouteProbe | None:
        for _ in range(self.attempts):
            try:
                question = self.generator.generate(route)
                if question_constraint_error(question, route):
                    continue
                oracle = self.oracle.evaluate(question, route)
                if not oracle.valid or answer_is_leaked(question, oracle.answer):
                    continue
                if route_fidelity(route, oracle) < self.route_threshold:
                    continue
                golden_answer = self.answer_agent.answer_sources(
                    question,
                    route.source_records,
                )
                if is_insufficient_answer(golden_answer):
                    continue
                parametric_answer = self.answer_agent.answer_question(question)
                judged = self.judge.evaluate(
                    oracle,
                    golden_answer,
                    None,
                    (
                        None
                        if is_insufficient_answer(parametric_answer)
                        else parametric_answer
                    ),
                )
            except (
                APIError,
                KeyError,
                TypeError,
                ValueError,
                StructuredOutputError,
            ):
                continue

            if judged.gold_correctness < self.correctness_threshold:
                continue
            if (
                not is_insufficient_answer(parametric_answer)
                and judged.parametric_correctness >= self.correctness_threshold
            ):
                continue

            question = ensure_question_mark(question)
            oracle = replace(oracle, question=question)
            question_id = hashlib.sha256(
                f"{route.route_id}\n{question}".encode("utf-8")
            ).hexdigest()[:20]
            return RouteProbe(question_id, route, oracle, golden_answer)
        return None
