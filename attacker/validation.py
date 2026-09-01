import re

from attacker.models import AttackMode, GraphRouteBundle, OracleResult


_META_TERMS = re.compile(
    r"\b(route|evidence|facts?|nodes?|memory|context|ids?|instructions?)\b",
    re.IGNORECASE,
)
_SPACES = re.compile(r"\W+", re.UNICODE)
_NON_UNIQUE = re.compile(
    r"\b(?:one|a)\s+(?:specific\s+)?(?:way|tip|method|example|option|suggestion)\b"
    r"|(?:一种|一个|一项|一点)(?:方法|建议|技巧|例子|选项)",
    re.IGNORECASE,
)
_USER_AS_ASSISTANT = re.compile(
    r"\b(?:what|which|where|when|why|how)\s+"
    r"(?:(?:did|do|are|were|have|had|would|could)\s+you|"
    r"(?:is|was|are|were)\s+your)\b"
    r"|你(?:是)?(?:如何|怎么|为何|为什么|何时|在哪|向|曾|说|提到|喜欢|想|计划|计算)"
    r"|你的[^？?]{0,20}(?:是什么|有哪些|多少|何时|哪里|怎么样)",
    re.IGNORECASE,
)
_FIRST_PERSON_ASSISTANT = re.compile(
    r"\bwhat did i (?:say|suggest|recommend|tell|explain|identify)\b"
    r"|我(?:说|建议|推荐|告诉|解释|指出)(?:了)?(?:什么|哪些)",
    re.IGNORECASE,
)


def question_constraint_error(
    question: str,
    route: GraphRouteBundle,
) -> str | None:
    """Return the first deterministic violation that makes a question unusable."""
    if not question:
        return "question_invalid"
    if question.count("?") + question.count("？") > 1:
        return "multiple_questions"
    if _META_TERMS.search(question):
        return "metadata_leak"
    if _NON_UNIQUE.search(question):
        return "non_unique_answer"

    lowered = question.casefold()
    identifiers = (
        *route.walk_node_ids,
        *(item.source_id for item in route.source_records),
    )
    if any(identifier.casefold() in lowered for identifier in identifiers):
        return "id_leak"
    return None


def speaker_role_error(question: str, oracle: OracleResult) -> str | None:
    roles = {item.role for item in oracle.supporting_evidence}
    if roles == {"user"} and _USER_AS_ASSISTANT.search(question):
        return "user_as_assistant"
    if roles == {"assistant"} and _FIRST_PERSON_ASSISTANT.search(question):
        return "assistant_as_user"
    return None


def has_terminal_question_mark(question: str) -> bool:
    return question.endswith(("?", "？"))


def ensure_question_mark(question: str) -> str:
    return question if has_terminal_question_mark(question) else question.rstrip(".!。！") + "?"


def normalize_question(question: str) -> str:
    return " ".join(question.casefold().rstrip("?.!？。！").split())


def answer_is_leaked(question: str, answer: str) -> bool:
    """Detect literal answer disclosure after the Oracle supplies the answer."""
    normalized_question = f" {_normalize(question)} "
    normalized_answer = _normalize(answer)
    return bool(normalized_answer) and f" {normalized_answer} " in normalized_question


def route_fidelity(route: GraphRouteBundle, oracle: OracleResult) -> float:
    source_ids = {item.source_id for item in oracle.supporting_evidence}
    used = {
        node.id
        for node in route.evidence_nodes
        if source_ids.intersection(node.source_ids)
    }
    intended = {node.id for node in route.evidence_nodes}
    required = 1 if route.attack_mode == AttackMode.SINGLE_FACT else 2
    return min(1.0, len(used & intended) / required)


def _normalize(text: str) -> str:
    return " ".join(_SPACES.sub(" ", text.casefold()).split())
