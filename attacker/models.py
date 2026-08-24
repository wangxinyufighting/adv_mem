from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from memory.models import MemoryNode, MemoryState
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
        )

    def to_attacker_context(self) -> dict[str, Any]:
        """Expose structured memories but not raw source text."""
        return {
            "route_id": self.route_id,
            "attack_mode": self.attack_mode.value,
            "walk_node_ids": list(self.walk_node_ids),
            "walk_steps": [asdict(step) for step in self.walk_steps],
            "evidence_nodes": [asdict(node) for node in self.evidence_nodes],
            "connector_nodes": [asdict(node) for node in self.connector_nodes],
        }

    def to_oracle_context(self) -> dict[str, Any]:
        """Expose only raw evidence used by the frozen Answer Agent."""
        return {
            "route_id": self.route_id,
            "attack_mode": self.attack_mode.value,
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
class AttackerObservation:
    route: GraphRouteBundle
    memory_neighborhood: tuple[MemoryNode, ...]
    prior_questions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_attacker_context(),
            "memory_neighborhood": [
                {
                    "id": node.id,
                    "content": node.content,
                    "tags": list(node.tags),
                    "time_span": node.time_span,
                }
                for node in self.memory_neighborhood
            ],
            "prior_questions": list(self.prior_questions),
        }


@dataclass(frozen=True)
class PriorQuestion:
    question: str
    answer: str


@dataclass(frozen=True)
class AttackerRewardContext:
    route: GraphRouteBundle
    memory_version: int
    memory_nodes: tuple[MemoryNode, ...]
    prior_questions: tuple[PriorQuestion, ...]

    @classmethod
    def from_state(
        cls,
        route: GraphRouteBundle,
        memory: MemoryState,
    ) -> "AttackerRewardContext":
        return cls(
            route=route,
            memory_version=memory.version,
            memory_nodes=tuple(memory.nodes.values()),
            prior_questions=tuple(
                PriorQuestion(record.question, record.oracle_answer)
                for record in memory.capability_ledger.values()
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route.to_dict(),
            "memory_version": self.memory_version,
            "memory_nodes": [node.to_dict() for node in self.memory_nodes],
            "prior_questions": [asdict(item) for item in self.prior_questions],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "AttackerRewardContext":
        return cls(
            route=GraphRouteBundle.from_dict(payload["route"]),
            memory_version=payload["memory_version"],
            memory_nodes=tuple(
                MemoryNode.from_dict(item) for item in payload["memory_nodes"]
            ),
            prior_questions=tuple(
                PriorQuestion(**item) for item in payload["prior_questions"]
            ),
        )

    def memory_state(self) -> MemoryState:
        return MemoryState(
            version=self.memory_version,
            nodes={node.id: node for node in self.memory_nodes},
        )
