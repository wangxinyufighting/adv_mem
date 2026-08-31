import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations

from attacker.models import (
    AttackMode,
    GraphRouteBundle,
    MemoryGraphView,
    RouteNode,
    RouteStep,
    RouterConfig,
    RouterState,
    SourceRecord,
)


class NoRouteFoundError(ValueError):
    pass


_COMPARISON_DIMENSIONS = {
    "cost": ("price", "cost", "budget", "paid", "purchase", "$", "价格", "费用", "预算"),
    "time": (
        "time",
        "date",
        "duration",
        "schedule",
        "frequency",
        "routine",
        "时间",
        "日期",
        "时长",
        "频率",
    ),
    "quantity": (
        "count",
        "number",
        "amount",
        "quantity",
        "distance",
        "数量",
        "次数",
        "距离",
    ),
    "preference": (
        "prefer",
        "favorite",
        "choice",
        "satisfaction",
        "like",
        "偏好",
        "喜欢",
        "选择",
    ),
    "location": ("location", "place", "venue", "city", "travel", "地点", "场所", "城市"),
    "physical_attribute": (
        "age",
        "size",
        "weight",
        "height",
        "length",
        "年龄",
        "尺寸",
        "重量",
        "长度",
    ),
}
_STATE_TERMS = (
    " plan ",
    " goal ",
    " decision ",
    " status ",
    " habit ",
    " job ",
    " relationship ",
    "计划",
    "目标",
    "决定",
    "状态",
    "习惯",
)
_TRANSITION_TERMS = (
    " no longer ",
    " used to ",
    " changed ",
    " switched ",
    " replaced ",
    " instead ",
    " started ",
    " stopped ",
    " increased ",
    " decreased ",
    " moved ",
    " became ",
    " finished ",
    " completed ",
    " graduated ",
    " quit ",
    " now ",
    " currently ",
    "不再",
    "改为",
    "换成",
    "开始",
    "停止",
    "增加",
    "减少",
    "完成",
    "毕业",
    "离职",
    "现在",
    "目前",
)


def _source_payload(raw_source):
    while isinstance(raw_source, str):
        raw_source = json.loads(raw_source)
    return raw_source


@dataclass(frozen=True)
class _RawRoute:
    mode: AttackMode
    walk_node_ids: tuple[str, ...]
    evidence_node_ids: tuple[str, ...]
    connector_node_ids: tuple[str, ...]
    steps: tuple[RouteStep, ...]
    mode_dimension: str | None = None


class _GraphIndex:
    """Small in-memory index used by the heuristic walker."""

    def __init__(self, graph: MemoryGraphView):
        self.node_by_id = {node["id"]: node for node in graph.nodes}
        self.edge_direction: dict[tuple[str, str, str], str] = {}
        self.topic_fact_ids: dict[str, list[str]] = {}
        self.merge_pairs: list[tuple[str, str]] = []

        for edge in graph.edges:
            source = edge["source"]
            target = edge["target"]
            edge_type = edge["type"]
            self.edge_direction[(source, target, edge_type)] = "forward"
            self.edge_direction[(target, source, edge_type)] = "reverse"

            if edge_type == "PARENT":
                self._index_parent_edge(source, target)
            elif edge_type == "MERGED_TO":
                self.merge_pairs.append((source, target))

        self.activated_fact_ids = [
            node_id
            for node_id, node in self.node_by_id.items()
            if node.get("type") == "fact" and node.get("status") == "activated"
        ]

    def _index_parent_edge(self, source: str, target: str) -> None:
        source_node = self.node_by_id[source]
        target_node = self.node_by_id[target]

        if source_node.get("type") == "topic" and target_node.get("type") == "fact":
            topic_id, fact_id = source, target
        elif source_node.get("type") == "fact" and target_node.get("type") == "topic":
            topic_id, fact_id = target, source
        else:
            return

        self.topic_fact_ids.setdefault(topic_id, []).append(fact_id)

    def step(self, from_id: str, to_id: str, edge_type: str) -> RouteStep:
        return RouteStep(
            from_node_id=from_id,
            edge_type=edge_type,
            to_node_id=to_id,
            traversal_direction=self.edge_direction[(from_id, to_id, edge_type)],
        )


class GraphRouterPolicy:
    """Coverage-oriented heuristic walker over a Full Memory Graph."""

    def __init__(
        self,
        config: RouterConfig | None = None,
        state: RouterState | None = None,
    ):
        self.config = config or RouterConfig()
        self.state = state or RouterState()
        self._random = random.Random(self.config.random_seed)

    def route(
        self,
        graph: MemoryGraphView,
        mode: AttackMode | None = None,
    ) -> GraphRouteBundle:
        index = _GraphIndex(graph)
        selected_mode = mode or self._random.choice(self.config.enabled_modes)

        try:
            for attempt in range(1, self.config.max_sampling_attempts + 1):
                route = self._sample_route(index, selected_mode)
                signature = self._signature(graph, route)
                if signature not in self.state.recent_route_signatures:
                    bundle = self._build_bundle(graph, index, route, signature, attempt)
                    self.state.record(bundle)
                    return bundle
        except NoRouteFoundError:
            if selected_mode != AttackMode.SINGLE_FACT and self.config.fallback_to_single_fact:
                return self.route(graph, AttackMode.SINGLE_FACT)
            raise

        raise NoRouteFoundError(f"No unseen {selected_mode.value} route is available")

    def coverage_route(self, graph: MemoryGraphView) -> GraphRouteBundle:
        """Anchor one route at least-visited eligible evidence."""
        index = _GraphIndex(graph)
        routes = [
            _RawRoute(
                mode=AttackMode.SINGLE_FACT,
                walk_node_ids=(fact_id,),
                evidence_node_ids=(fact_id,),
                connector_node_ids=(),
                steps=(),
            )
            for fact_id in index.activated_fact_ids
        ]
        routes.extend(
            _RawRoute(
                mode=AttackMode.TEMPORAL_EVOLUTION,
                walk_node_ids=(source_id, target_id),
                evidence_node_ids=(source_id, target_id),
                connector_node_ids=(),
                steps=(index.step(source_id, target_id, "MERGED_TO"),),
                mode_dimension=dimension,
            )
            for source_id, target_id, dimension in self._temporal_pairs(index)
        )
        if not routes:
            raise NoRouteFoundError("No eligible evidence route is available")
        def score(item: _RawRoute) -> tuple[int, int]:
            visits = [self._visit_count(node_id) for node_id in item.evidence_node_ids]
            return min(visits), sum(visits)

        best = min(map(score, routes))
        route = self._random.choice([item for item in routes if score(item) == best])
        signature = self._signature(graph, route)
        bundle = self._build_bundle(graph, index, route, signature, 1)
        self.state.record(bundle)
        return bundle

    def _sample_route(self, index: _GraphIndex, mode: AttackMode) -> _RawRoute:
        handlers = {
            AttackMode.SINGLE_FACT: self._route_single_fact,
            AttackMode.SAME_TOPIC: self._route_same_topic,
            AttackMode.TEMPORAL_EVOLUTION: self._route_temporal_evolution,
            AttackMode.COMPARISON: self._route_comparison,
        }
        return handlers[mode](index)

    def _route_single_fact(self, index: _GraphIndex) -> _RawRoute:
        if not index.activated_fact_ids:
            raise NoRouteFoundError("No activated fact is available")

        fact_id = self._weighted_choice(index.activated_fact_ids)
        return _RawRoute(
            mode=AttackMode.SINGLE_FACT,
            walk_node_ids=(fact_id,),
            evidence_node_ids=(fact_id,),
            connector_node_ids=(),
            steps=(),
        )

    def _route_same_topic(self, index: _GraphIndex) -> _RawRoute:
        minimum = max(2, self.config.min_evidence_nodes)
        candidates = {
            topic_id: self._activated_facts(index, fact_ids)
            for topic_id, fact_ids in index.topic_fact_ids.items()
        }
        candidates = {
            topic_id: fact_ids
            for topic_id, fact_ids in candidates.items()
            if len(fact_ids) >= minimum
        }
        if not candidates:
            raise NoRouteFoundError("No topic has enough activated facts")

        topic_id = self._random.choice(list(candidates))
        maximum = min(self.config.max_evidence_nodes, len(candidates[topic_id]))
        count = self._random.randint(minimum, maximum)
        fact_ids = self._weighted_sample(candidates[topic_id], count)
        return self._topic_route(index, AttackMode.SAME_TOPIC, topic_id, fact_ids)

    def _route_temporal_evolution(self, index: _GraphIndex) -> _RawRoute:
        pairs = self._temporal_pairs(index)
        if not pairs:
            raise NoRouteFoundError("No archived-to-activated MERGED_TO path is available")

        source_id, target_id, dimension = self._weighted_choice(
            pairs,
            key=lambda pair: self._visit_count(pair[0]) + self._visit_count(pair[1]),
        )
        return _RawRoute(
            mode=AttackMode.TEMPORAL_EVOLUTION,
            walk_node_ids=(source_id, target_id),
            evidence_node_ids=(source_id, target_id),
            connector_node_ids=(),
            steps=(index.step(source_id, target_id, "MERGED_TO"),),
            mode_dimension=dimension,
        )

    def _temporal_pairs(self, index: _GraphIndex) -> list[tuple[str, str, str]]:
        pairs = []
        for source_id, target_id in index.merge_pairs:
            source = index.node_by_id[source_id]
            target = index.node_by_id[target_id]
            dimension = self._temporal_dimension(source, target)
            if (
                source.get("type") == "fact"
                and source.get("status") == "archived"
                and target.get("type") == "fact"
                and target.get("status") == "activated"
                and self._has_later_source(source, target)
                and dimension
            ):
                pairs.append((source_id, target_id, dimension))
        return pairs

    @classmethod
    def _temporal_dimension(cls, source: dict, target: dict) -> str | None:
        target_text = " " + re.sub(r"\W+", " ", cls._node_text(target)) + " "
        if not any(term in target_text for term in _TRANSITION_TERMS):
            return None
        shared_tags = {
            str(tag).casefold()
            for tag in source.get("tags") or []
        } & {
            str(tag).casefold()
            for tag in target.get("tags") or []
        }
        if shared_tags:
            return min(shared_tags, key=lambda tag: (len(tag), tag))
        return "state" if any(term in target_text for term in _STATE_TERMS) else None

    @staticmethod
    def _has_later_source(source: dict, target: dict) -> bool:
        source_time = GraphRouterPolicy._latest_source_time(source)
        target_time = GraphRouterPolicy._latest_source_time(target)
        return bool(source_time and target_time and target_time > source_time)

    @staticmethod
    def _latest_source_time(node: dict) -> datetime | None:
        times = [
            datetime.fromisoformat(source["chat_time"].replace("Z", "+00:00"))
            for raw_source in node.get("sources") or []
            if (source := _source_payload(raw_source)).get("chat_time")
        ]
        return max(times, default=None)

    def _route_comparison(self, index: _GraphIndex) -> _RawRoute:
        candidates: list[tuple[str, str, str, str]] = []
        for topic_id, fact_ids in index.topic_fact_ids.items():
            active_fact_ids = self._activated_facts(index, fact_ids)
            for left_id, right_id in combinations(active_fact_ids, 2):
                dimensions = self._dimensions(
                    index.node_by_id[left_id]
                ) & self._dimensions(index.node_by_id[right_id])
                candidates.extend(
                    (topic_id, left_id, right_id, dimension)
                    for dimension in dimensions
                )

        if not candidates:
            raise NoRouteFoundError("No comparable facts are available")

        topic_id, left_id, right_id, dimension = self._weighted_choice(
            candidates,
            key=lambda item: self._visit_count(item[1]) + self._visit_count(item[2]),
        )
        return self._topic_route(
            index,
            AttackMode.COMPARISON,
            topic_id,
            [left_id, right_id],
            dimension,
        )

    @staticmethod
    def _node_text(node: dict) -> str:
        return " ".join(
            str(value)
            for value in (node.get("key"), node.get("memory"), *(node.get("tags") or []))
            if value
        ).casefold()

    @classmethod
    def _dimensions(cls, node: dict) -> set[str]:
        text = cls._node_text(node)
        return {
            dimension
            for dimension, terms in _COMPARISON_DIMENSIONS.items()
            if any(cls._contains_term(text, term) for term in terms)
        }

    @staticmethod
    def _contains_term(text: str, term: str) -> bool:
        if term.isascii() and term != "$":
            return bool(re.search(rf"\b{re.escape(term)}\b", text))
        return term in text

    def _topic_route(
        self,
        index: _GraphIndex,
        mode: AttackMode,
        topic_id: str,
        fact_ids: list[str],
        mode_dimension: str | None = None,
    ) -> _RawRoute:
        walk = [fact_ids[0]]
        steps = []
        for fact_id in fact_ids[1:]:
            steps.append(index.step(walk[-1], topic_id, "PARENT"))
            walk.append(topic_id)
            steps.append(index.step(topic_id, fact_id, "PARENT"))
            walk.append(fact_id)

        return _RawRoute(
            mode=mode,
            walk_node_ids=tuple(walk),
            evidence_node_ids=tuple(fact_ids),
            connector_node_ids=(topic_id,),
            steps=tuple(steps),
            mode_dimension=mode_dimension,
        )

    def _activated_facts(self, index: _GraphIndex, fact_ids: list[str]) -> list[str]:
        return [
            fact_id
            for fact_id in fact_ids
            if index.node_by_id[fact_id].get("status") == "activated"
        ]

    def _weighted_choice(self, items, key=None):
        score = key or self._visit_count
        weights = [1.0 / (1 + score(item)) for item in items]
        return self._random.choices(items, weights=weights, k=1)[0]

    def _weighted_sample(self, items: list[str], count: int) -> list[str]:
        pool = list(items)
        selected = []
        for _ in range(count):
            item = self._weighted_choice(pool)
            selected.append(item)
            pool.remove(item)
        return selected

    def _visit_count(self, node_id: str) -> int:
        return self.state.node_visit_counts.get(node_id, 0)

    def _signature(self, graph: MemoryGraphView, route: _RawRoute) -> str:
        value = "|".join(
            [
                graph.user_name,
                route.mode.value,
                route.mode_dimension or "",
                *route.walk_node_ids,
            ]
        )
        return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]

    def _build_bundle(
        self,
        graph: MemoryGraphView,
        index: _GraphIndex,
        route: _RawRoute,
        signature: str,
        attempt: int,
    ) -> GraphRouteBundle:
        unique_sources = {}
        sources_by_node = {}
        for node_id in route.evidence_node_ids:
            node_sources = {}
            for source in self._parse_sources(index.node_by_id[node_id]):
                identity = (source.role, source.chat_time, source.content)
                canonical = unique_sources.setdefault(identity, source)
                node_sources[canonical.source_id] = canonical
            sources_by_node[node_id] = tuple(node_sources.values())

        evidence_nodes = tuple(
            self._route_node(index.node_by_id[node_id], sources_by_node[node_id])
            for node_id in route.evidence_node_ids
        )
        connector_nodes = tuple(
            self._route_node(index.node_by_id[node_id], ())
            for node_id in route.connector_node_ids
        )
        source_records = tuple(unique_sources.values())

        return GraphRouteBundle(
            route_id=signature,
            graph_version=graph.graph_version,
            case_index=graph.case_index,
            user_name=graph.user_name,
            attack_mode=route.mode,
            walk_node_ids=route.walk_node_ids,
            walk_steps=route.steps,
            evidence_nodes=evidence_nodes,
            connector_nodes=connector_nodes,
            source_records=source_records,
            route_signature=signature,
            sampling_seed=self.config.random_seed,
            sampling_attempt=attempt,
            mode_dimension=route.mode_dimension,
        )

    @staticmethod
    def _route_node(
        node: dict,
        sources: tuple[SourceRecord, ...],
    ) -> RouteNode:
        return RouteNode(
            id=node["id"],
            type=node.get("type"),
            status=node.get("status"),
            memory_type=node.get("memory_type"),
            key=node.get("key"),
            memory=node.get("memory", ""),
            background=node.get("background"),
            tags=tuple(node.get("tags") or []),
            confidence=node.get("confidence"),
            version=node.get("version"),
            created_at=node.get("created_at"),
            updated_at=node.get("updated_at"),
            source_ids=tuple(source.source_id for source in sources),
        )

    @staticmethod
    def _parse_sources(node: dict) -> tuple[SourceRecord, ...]:
        return tuple(
            SourceRecord(
                source_id=f"{node['id']}:{index}",
                node_id=node["id"],
                role=source.get("role"),
                chat_time=source.get("chat_time"),
                content=source.get("content", ""),
            )
            for index, raw_source in enumerate(node.get("sources") or [])
            for source in (_source_payload(raw_source),)
        )
