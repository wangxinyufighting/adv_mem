import re

from attacker.models import AttackMode, GraphRouteBundle, OracleResult


_META_TERMS = re.compile(
    r"\b(route|evidence|facts?|nodes?|memory|context|ids?|instructions?)\b",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\W+", re.UNICODE)


def question_constraint_error(
    question: str,
    route: GraphRouteBundle,
) -> str | None:
    """Return the first deterministic question-format violation."""
    if not question or not question.endswith(("?", "？")):
        return "question_invalid"
    if question.count("?") + question.count("？") != 1:
        return "multiple_questions"
    if _META_TERMS.search(question):
        return "metadata_leak"

    lowered = question.casefold()
    identifiers = (
        *route.walk_node_ids,
        *(item.source_id for item in route.source_records),
    )
    if any(identifier.casefold() in lowered for identifier in identifiers):
        return "id_leak"
    return None


def answer_is_leaked(question: str, answer: str) -> bool:
    """Detect literal answer disclosure after the Oracle supplies the answer."""
    normalized_question = f" {_normalize(question)} "
    normalized_answer = _normalize(answer)
    return bool(normalized_answer) and f" {normalized_answer} " in normalized_question


def route_fidelity(route: GraphRouteBundle, oracle: OracleResult) -> float:
    used = {item.node_id for item in oracle.supporting_evidence}
    intended = {node.id for node in route.evidence_nodes}
    required = 1 if route.attack_mode == AttackMode.SINGLE_FACT else 2
    return min(1.0, len(used & intended) / required)


def _normalize(text: str) -> str:
    return " ".join(_SPACES.sub(" ", text.casefold()).split())
