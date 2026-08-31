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
        failures: dict[str, int] = {}
        last_error = ""
        for attempt in range(1, self.attempts + 1):
            stage = "generate"
            try:
                question = self.generator.generate(route)
                if error := question_constraint_error(question, route):
                    last_error = self._reject(
                        failures, f"question_constraint/{error}"
                    )
                    continue
                stage = "oracle"
                oracle = self.oracle.evaluate(question, route)
                if not oracle.valid:
                    last_error = self._reject(
                        failures,
                        f"oracle_invalid/{oracle.invalid_reason or 'unknown'}",
                    )
                    continue
                if answer_is_leaked(question, oracle.answer):
                    last_error = self._reject(failures, "answer_leak")
                    continue
                fidelity = route_fidelity(route, oracle)
                if fidelity < self.route_threshold:
                    last_error = self._reject(
                        failures,
                        "route_fidelity",
                        f"{fidelity:.3f}<{self.route_threshold:.3f}",
                    )
                    continue
                stage = "gold_answer"
                golden_answer = self.answer_agent.answer_sources(
                    question,
                    route.source_records,
                )
                if is_insufficient_answer(golden_answer):
                    last_error = self._reject(failures, "gold_insufficient")
                    continue
                stage = "parametric_answer"
                parametric_answer = self.answer_agent.answer_question(question)
                stage = "judge"
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
            ) as error:
                last_error = self._reject(
                    failures,
                    f"{stage}_error",
                    f"{type(error).__name__}: {error}",
                )
                continue

            if judged.gold_correctness < self.correctness_threshold:
                last_error = self._reject(
                    failures,
                    "gold_incorrect",
                    f"{judged.gold_correctness:.3f}<{self.correctness_threshold:.3f}",
                )
                continue
            if (
                not is_insufficient_answer(parametric_answer)
                and judged.parametric_correctness >= self.correctness_threshold
            ):
                last_error = self._reject(
                    failures,
                    "parametric_leak",
                    f"{judged.parametric_correctness:.3f}",
                )
                continue

            question = ensure_question_mark(question)
            oracle = replace(oracle, question=question)
            question_id = hashlib.sha256(
                f"{route.route_id}\n{question}".encode("utf-8")
            ).hexdigest()[:20]
            self._log(route, True, attempt, failures, last_error)
            return RouteProbe(question_id, route, oracle, golden_answer)
        self._log(route, False, self.attempts, failures, last_error)
        return None

    @staticmethod
    def _reject(
        failures: dict[str, int],
        stage: str,
        detail: str = "",
    ) -> str:
        failures[stage] = failures.get(stage, 0) + 1
        return f"{stage}: {detail}" if detail else stage

    @staticmethod
    def _log(
        route: GraphRouteBundle,
        success: bool,
        attempts: int,
        failures: dict[str, int],
        last_error: str,
    ) -> None:
        print(
            "Probe Build: "
            + json.dumps(
                {
                    "route_id": route.route_id,
                    "success": success,
                    "attempts": attempts,
                    "failures": failures,
                    "last_error": last_error[:300],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
