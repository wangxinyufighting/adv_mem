from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from memory.models import MemoryNode, MemoryState, RouteAttackStats
from utils.longmemeval_graph_reader import LongMemEvalGraphCase


class AttackMode(str, Enum):
    SINGLE_FACT = "single_fact"
    SAME_TOPIC = "same_topic"
    TEMPORAL_EVOLUTION = "temporal_evolution"
    COMPARISON = "comparison"


@dataclass(frozen=True)
class RouterConfig:
    min_evidence_nodes: int = 1
    max_evidence_nodes: int = 3
    max_sampling_attempts: int = 10
    random_seed: int | None = 0
    fallback_to_single_fact: bool = True
    enabled_modes: tuple[AttackMode, ...] = (
        AttackMode.SINGLE_FACT,
        AttackMode.SAME_TOPIC,
        AttackMode.TEMPORAL_EVOLUTION,
        AttackMode.COMPARISON,
    )


@dataclass
class RouterState:
    node_visit_counts: dict[str, int] = field(default_factory=dict)
    recent_route_signatures: set[str] = field(default_factory=set)

    def record(self, route: "GraphRouteBundle") -> None:
        for node in route.evidence_nodes:
            self.node_visit_counts[node.id] = self.node_visit_counts.get(node.id, 0) + 1
        self.recent_route_signatures.add(route.route_signature)


@dataclass(frozen=True)
class MemoryGraphView:
    """Graph-only view of a case; benchmark question and answer are excluded."""

    case_index: int
    user_name: str
    graph_version: str
    nodes: tuple[dict[str, Any], ...]
    edges: tuple[dict[str, Any], ...]

    @classmethod
    def from_case(
        cls,
        case: LongMemEvalGraphCase,
        graph_version: str,
    ) -> "MemoryGraphView":
        return cls(
            case_index=case.case_index,
            user_name=case.user_name,
            graph_version=graph_version,
            nodes=tuple(case.nodes),
            edges=tuple(case.edges),
        )


@dataclass(frozen=True)
class RouteStep:
    from_node_id: str
    edge_type: str
    to_node_id: str
    traversal_direction: str


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    node_id: str
    role: str | None
    chat_time: str | None
    content: str


@dataclass(frozen=True)
class RouteNode:
    id: str
    type: str | None
    status: str | None
    memory_type: str | None
    key: str | None
    memory: str
    background: str | None
    tags: tuple[str, ...]
    confidence: float | None
    version: int | None
    created_at: str | None
    updated_at: str | None
    source_ids: tuple[str, ...] = ()


"""
一次游走产生的完整“路径数据包”:

路径节点 ID
路径经过的边
证据 fact 节点
连接用的 topic 节点
原始 sources
attack mode
采样信息
"""


@dataclass(frozen=True)
class GraphRouteBundle:
    route_id: str
    graph_version: str
    case_index: int
    user_name: str
    attack_mode: AttackMode
    walk_node_ids: tuple[str, ...]
    walk_steps: tuple[RouteStep, ...]
    evidence_nodes: tuple[RouteNode, ...]
    connector_nodes: tuple[RouteNode, ...]
    source_records: tuple[SourceRecord, ...]
    route_signature: str
    sampling_seed: int | None
    sampling_attempt: int
    mode_dimension: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route_id": self.route_id,
            "graph_version": self.graph_version,
            "case_index": self.case_index,
            "user_name": self.user_name,
            "attack_mode": self.attack_mode.value,
            "walk_node_ids": list(self.walk_node_ids),
            "walk_steps": [asdict(step) for step in self.walk_steps],
            "evidence_nodes": [asdict(node) for node in self.evidence_nodes],
            "connector_nodes": [asdict(node) for node in self.connector_nodes],
            "source_records": [asdict(source) for source in self.source_records],
            "route_signature": self.route_signature,
            "sampling_seed": self.sampling_seed,
            "sampling_attempt": self.sampling_attempt,
            "mode_dimension": self.mode_dimension,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GraphRouteBundle":
        def route_node(item: dict[str, Any]) -> RouteNode:
            return RouteNode(
                **{
                    **item,
                    "tags": tuple(item["tags"]),
                    "source_ids": tuple(item["source_ids"]),
                }
            )

        return cls(
            route_id=payload["route_id"],
            graph_version=payload["graph_version"],
            case_index=payload["case_index"],
            user_name=payload["user_name"],
            attack_mode=AttackMode(payload["attack_mode"]),
            walk_node_ids=tuple(payload["walk_node_ids"]),
            walk_steps=tuple(RouteStep(**item) for item in payload["walk_steps"]),
            evidence_nodes=tuple(
                route_node(item) for item in payload["evidence_nodes"]
            ),
            connector_nodes=tuple(
                route_node(item) for item in payload["connector_nodes"]
            ),
            source_records=tuple(
                SourceRecord(**item) for item in payload["source_records"]
            ),
            route_signature=payload["route_signature"],
            sampling_seed=payload["sampling_seed"],
            sampling_attempt=payload["sampling_attempt"],
            mode_dimension=payload.get("mode_dimension"),
        )

    def to_attacker_context(self) -> dict[str, Any]:
        """Expose only route fields needed to generate the question."""
        context = {"target": [node.memory for node in self.evidence_nodes]}
        if self.attack_mode == AttackMode.COMPARISON and self.mode_dimension:
            context["dimension"] = self.mode_dimension
        return context

    def to_oracle_context(self) -> dict[str, Any]:
        """Expose only raw evidence used by the frozen Answer Agent."""
        return {
            "attack_mode": self.attack_mode.value,
            "mode_dimension": self.mode_dimension,
            "source_records": [asdict(source) for source in self.source_records],
        }


@dataclass(frozen=True)
class SupportingEvidence:
    source_id: str
    node_id: str
    quote: str
    chat_time: str | None
    role: str | None


@dataclass(frozen=True)
class OracleResult:
    route_id: str
    question: str
    valid: bool
    answer: str | None
    supporting_evidence: tuple[SupportingEvidence, ...]
    invalid_reason: str | None
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "supporting_evidence": [
                asdict(item) for item in self.supporting_evidence
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "OracleResult":
        return cls(
            **{
                **payload,
                "supporting_evidence": tuple(
                    SupportingEvidence(**item)
                    for item in payload["supporting_evidence"]
                ),
            }
        )


@dataclass(frozen=True)
class RouteProbe:
    """A route with one frozen, Oracle-validated diagnostic question."""

    question_id: str
    route: GraphRouteBundle
    oracle: OracleResult
    golden_answer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "route": self.route.to_dict(),
            "oracle": self.oracle.to_dict(),
            "golden_answer": self.golden_answer,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteProbe":
        return cls(
            question_id=payload["question_id"],
            route=GraphRouteBundle.from_dict(payload["route"]),
            oracle=OracleResult.from_dict(payload["oracle"]),
            golden_answer=payload["golden_answer"],
        )


@dataclass(frozen=True)
class AttackerObservation:
    route: GraphRouteBundle
    memory_neighborhood: tuple[MemoryNode, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.route.to_attacker_context(),
            "known": [node.content for node in self.memory_neighborhood],
        }


@dataclass(frozen=True)
class RouteCandidateObservation:
    choice: int
    probe: RouteProbe
    memory_neighborhood: tuple[MemoryNode, ...]
    history: RouteAttackStats

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "choice": self.choice,
            "relation": self.probe.route.attack_mode.value,
            "dimension": self.probe.route.mode_dimension,
            "target": [node.memory for node in self.probe.route.evidence_nodes],
            "probe_question": self.probe.oracle.question,
            "known": [node.content for node in self.memory_neighborhood],
            "history": self.history.to_dict(),
        }


@dataclass(frozen=True)
class RouteSelectorObservation:
    memory_version: int
    candidates: tuple[RouteCandidateObservation, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "candidates": [item.to_prompt_dict() for item in self.candidates],
        }


@dataclass(frozen=True)
class RouteSelectorRewardContext:
    memory_version: int
    memory_nodes: tuple[MemoryNode, ...]
    attack_history: tuple[RouteAttackStats, ...]
    probes: tuple[RouteProbe, ...]

    @classmethod
    def from_state(
        cls,
        probes: tuple[RouteProbe, ...],
        memory: MemoryState,
    ) -> "RouteSelectorRewardContext":
        return cls(
            memory_version=memory.version,
            memory_nodes=tuple(memory.nodes.values()),
            attack_history=tuple(memory.attack_history.values()),
            probes=probes,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_version": self.memory_version,
            "memory_nodes": [node.to_dict() for node in self.memory_nodes],
            "attack_history": [item.to_dict() for item in self.attack_history],
            "probes": [item.to_dict() for item in self.probes],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RouteSelectorRewardContext":
        return cls(
            memory_version=payload["memory_version"],
            memory_nodes=tuple(
                MemoryNode.from_dict(item) for item in payload["memory_nodes"]
            ),
            attack_history=tuple(
                RouteAttackStats.from_dict(item)
                for item in payload.get("attack_history", [])
            ),
            probes=tuple(RouteProbe.from_dict(item) for item in payload["probes"]),
        )

    def memory_state(self) -> MemoryState:
        history = {item.route_id: item for item in self.attack_history}
        return MemoryState(
            version=self.memory_version,
            nodes={node.id: node for node in self.memory_nodes},
            attack_history=history,
        )
