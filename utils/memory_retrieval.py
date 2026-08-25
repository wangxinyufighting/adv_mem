import math
import re

from collections import Counter, deque
from dataclasses import dataclass
from typing import Protocol

from memory.models import MemoryNode, MemoryState


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]")


@dataclass(frozen=True)
class QueryPlan:
    """Structured retrieval query produced by the query parser."""

    rewritten_query: str
    retrieval_queries: tuple[str, ...]
    keywords: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()

    @property
    def embedding_queries(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((self.rewritten_query, *self.retrieval_queries)))


class QueryParser(Protocol):
    def parse(self, query: str) -> QueryPlan: ...


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class TextReranker(Protocol):
    def rerank(self, query: str, documents: list[str]) -> list[float]: ...


@dataclass(frozen=True)
class RetrievalConfig:
    recall_k: int = 20
    graph_depth: int = 1
    graph_decay: float = 0.8
    max_candidates: int = 40
    bm25_k1: float = 1.5
    bm25_b: float = 0.75


@dataclass(frozen=True)
class RetrievalResult:
    node: MemoryNode
    score: float
    rank: int
    recall_score: float = 0.0
    retrieval_sources: tuple[str, ...] = ()
    graph_seed_id: str | None = None
    hop_distance: int = 0


@dataclass(frozen=True)
class RetrievalBundle:
    query: str
    query_plan: QueryPlan
    results: tuple[RetrievalResult, ...]


class BM25MemoryRetriever:
    """BM25 retrieval over the active nodes of M_t."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        memory: MemoryState,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        nodes = memory.active_nodes
        scores = self.score(query, nodes)
        ranked = sorted(
            (
                (score, node)
                for node, score in zip(nodes, scores, strict=True)
                if score > 0
            ),
            key=lambda item: (-item[0], item[1].id),
        )
        return [
            RetrievalResult(
                node=node,
                score=score,
                rank=rank,
                recall_score=score,
                retrieval_sources=("bm25",),
            )
            for rank, (score, node) in enumerate(ranked[:top_k], start=1)
        ]

    def score(self, query: str, nodes: tuple[MemoryNode, ...]) -> list[float]:
        if not nodes:
            return []

        query_tokens = set(self._tokenize(query))
        documents = [self._tokenize(self.document_text(node)) for node in nodes]
        if not query_tokens:
            return [0.0] * len(nodes)

        document_frequency = Counter(
            token for document in documents for token in set(document)
        )
        average_length = sum(map(len, documents)) / len(documents)
        if average_length == 0:
            return [0.0] * len(nodes)
        return [
            self._score_document(
                query_tokens,
                document,
                document_frequency,
                len(documents),
                average_length,
            )
            for document in documents
        ]

    def _score_document(
        self,
        query_tokens: set[str],
        document: list[str],
        document_frequency: Counter,
        document_count: int,
        average_length: float,
    ) -> float:
        term_frequency = Counter(document)
        length_ratio = len(document) / average_length
        score = 0.0

        for token in query_tokens:
            frequency = term_frequency[token]
            if not frequency:
                continue

            frequency_in_documents = document_frequency[token]
            idf = math.log(
                1
                + (document_count - frequency_in_documents + 0.5)
                / (frequency_in_documents + 0.5)
            )
            denominator = frequency + self.k1 * (1 - self.b + self.b * length_ratio)
            score += idf * frequency * (self.k1 + 1) / denominator

        return score

    @staticmethod
    def document_text(node: MemoryNode) -> str:
        return " ".join((node.content, *node.tags))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return TOKEN_PATTERN.findall(text.lower())


class HybridMemoryRetriever:
    """MemOS-style recall adapted to the lightweight in-memory M_t."""

    def __init__(
        self,
        query_parser: QueryParser,
        embedder: TextEmbedder,
        reranker: TextReranker,
        config: RetrievalConfig | None = None,
    ):
        self.query_parser = query_parser
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()
        self.bm25 = BM25MemoryRetriever(
            k1=self.config.bm25_k1,
            b=self.config.bm25_b,
        )
        self._embedding_cache: dict[str, tuple[str, list[float]]] = {}

    @classmethod
    def from_env(cls, config: RetrievalConfig | None = None) -> "HybridMemoryRetriever":
        from utils.retrieval_clients import BGEReranker, LLMQueryParser, OpenAIEmbedder

        return cls(
            query_parser=LLMQueryParser.from_env(),
            embedder=OpenAIEmbedder.from_env(),
            reranker=BGEReranker.from_env(),
            config=config,
        )

    def retrieve(
        self,
        query: str,
        memory: MemoryState,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        return list(self.retrieve_bundle(query, memory, top_k).results)

    def retrieve_bundle(
        self,
        query: str,
        memory: MemoryState,
        top_k: int = 5,
    ) -> RetrievalBundle:
        nodes = memory.active_nodes
        if not nodes:
            empty_plan = QueryPlan(query, (query,))
            return RetrievalBundle(query=query, query_plan=empty_plan, results=())

        plan = self.query_parser.parse(query)
        node_by_id = {node.id: node for node in nodes}
        recall_scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}

        bm25_query = " ".join((query, plan.rewritten_query, *plan.keywords, *plan.tags))
        bm25_scores = self.bm25.score(bm25_query, nodes)
        self._add_ranked_channel(
            nodes,
            self._normalize_positive(bm25_scores),
            "bm25",
            recall_scores,
            sources,
        )

        query_vectors = self.embedder.embed(list(plan.embedding_queries))
        node_vectors = self._node_embeddings(nodes)
        cosine_scores = [
            max(
                self._cosine(query_vector, node_vector)
                for query_vector in query_vectors
            )
            for node_vector in node_vectors
        ]
        self._add_ranked_channel(
            nodes,
            [(score + 1.0) / 2.0 for score in cosine_scores],
            "embedding",
            recall_scores,
            sources,
        )

        self._add_metadata_matches(plan, nodes, recall_scores, sources)
        graph_info = self._expand_graph(memory, set(recall_scores), recall_scores)
        for node_id, (_, _, score) in graph_info.items():
            if node_id not in node_by_id:
                continue
            recall_scores[node_id] = max(recall_scores.get(node_id, 0.0), score)
            sources.setdefault(node_id, set()).add("graph")

        candidate_ids = sorted(
            recall_scores,
            key=lambda node_id: (-recall_scores[node_id], node_id),
        )[: self.config.max_candidates]
        candidates = [node_by_id[node_id] for node_id in candidate_ids]
        rerank_scores = self.reranker.rerank(
            plan.rewritten_query,
            [node.content for node in candidates],
        )

        ranked = sorted(
            zip(candidates, rerank_scores, strict=True),
            key=lambda item: (-item[1], item[0].id),
        )[:top_k]
        results = tuple(
            RetrievalResult(
                node=node,
                score=score,
                rank=rank,
                recall_score=recall_scores[node.id],
                retrieval_sources=tuple(sorted(sources[node.id])),
                graph_seed_id=graph_info.get(node.id, (None, 0, 0.0))[0],
                hop_distance=graph_info.get(node.id, (None, 0, 0.0))[1],
            )
            for rank, (node, score) in enumerate(ranked, start=1)
        )
        return RetrievalBundle(query=query, query_plan=plan, results=results)

    def _node_embeddings(self, nodes: tuple[MemoryNode, ...]) -> list[list[float]]:
        missing = []
        for node in nodes:
            text = self.bm25.document_text(node)
            cached = self._embedding_cache.get(node.id)
            if cached is None or cached[0] != text:
                missing.append((node.id, text))

        if missing:
            vectors = self.embedder.embed([text for _, text in missing])
            for (node_id, text), vector in zip(missing, vectors, strict=True):
                self._embedding_cache[node_id] = (text, vector)

        return [self._embedding_cache[node.id][1] for node in nodes]

    def _add_ranked_channel(
        self,
        nodes: tuple[MemoryNode, ...],
        scores: list[float],
        source: str,
        recall_scores: dict[str, float],
        sources: dict[str, set[str]],
    ) -> None:
        ranked = sorted(
            zip(nodes, scores, strict=True),
            key=lambda item: (-item[1], item[0].id),
        )[: self.config.recall_k]
        for node, score in ranked:
            if score <= 0:
                continue
            recall_scores[node.id] = max(recall_scores.get(node.id, 0.0), score)
            sources.setdefault(node.id, set()).add(source)

    @staticmethod
    def _add_metadata_matches(
        plan: QueryPlan,
        nodes: tuple[MemoryNode, ...],
        recall_scores: dict[str, float],
        sources: dict[str, set[str]],
    ) -> None:
        wanted_tags = {tag.lower() for tag in plan.tags}
        keywords = {keyword.lower() for keyword in plan.keywords}

        for node in nodes:
            node_tags = {tag.lower() for tag in node.tags}
            content = node.content.lower()
            if wanted_tags & node_tags or any(
                keyword in content for keyword in keywords
            ):
                recall_scores[node.id] = max(recall_scores.get(node.id, 0.0), 1.0)
                sources.setdefault(node.id, set()).add("metadata")

    def _expand_graph(
        self,
        memory: MemoryState,
        seed_ids: set[str],
        recall_scores: dict[str, float],
    ) -> dict[str, tuple[str, int, float]]:
        adjacency = {node_id: set() for node_id in memory.nodes}
        nodes_by_provenance: dict[str, list[str]] = {}

        for node in memory.nodes.values():
            for provenance_id in node.provenance_node_ids:
                nodes_by_provenance.setdefault(provenance_id, []).append(node.id)
                if provenance_id in memory.nodes:
                    adjacency[node.id].add(provenance_id)
                    adjacency[provenance_id].add(node.id)

        # M_t has no graph database: memories sharing source graph nodes are neighbors.
        for node_ids in nodes_by_provenance.values():
            for node_id in node_ids:
                adjacency[node_id].update(
                    other_id for other_id in node_ids if other_id != node_id
                )

        queue = deque(
            (seed_id, seed_id, 0)
            for seed_id in sorted(
                seed_ids, key=lambda node_id: (-recall_scores[node_id], node_id)
            )
        )
        visited = set(seed_ids)
        expanded: dict[str, tuple[str, int, float]] = {}

        while queue:
            node_id, seed_id, distance = queue.popleft()
            if distance == self.config.graph_depth:
                continue
            for neighbor_id in sorted(adjacency[node_id]):
                if neighbor_id in visited:
                    continue
                visited.add(neighbor_id)
                hop = distance + 1
                score = recall_scores[seed_id] * self.config.graph_decay**hop
                expanded[neighbor_id] = (seed_id, hop, score)
                queue.append((neighbor_id, seed_id, hop))

        return expanded

    @staticmethod
    def _normalize_positive(scores: list[float]) -> list[float]:
        maximum = max(scores, default=0.0)
        if maximum <= 0:
            return [0.0] * len(scores)
        return [score / maximum for score in scores]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0
