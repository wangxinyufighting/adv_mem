import json
import os

import requests

from openai import OpenAI

from utils.memory_retrieval import QueryPlan


QUERY_PARSER_PROMPT = """Analyze a question for memory retrieval.
Return JSON only with this schema:
{
  "rewritten_query": "a self-contained version of the question",
  "retrieval_queries": ["1 to 3 short semantic search queries"],
  "keywords": ["important entities, dates, and exact terms"],
  "tags": ["broad topic tags"]
}
Do not answer the question. Preserve all constraints and temporal references.
"""


class LLMQueryParser:
    """LLM query parsing through an OpenAI-compatible chat endpoint."""

    def __init__(self, client: OpenAI, model: str):
        self.client = client
        self.model = model
        self._cache: dict[str, QueryPlan] = {}

    @classmethod
    def from_env(cls) -> "LLMQueryParser":
        return cls(
            client=OpenAI(
                api_key=os.getenv("RETRIEVAL_LLM_API_KEY")
                or os.environ["DEEPSEEK_API_KEY"],
                base_url=os.getenv(
                    "RETRIEVAL_LLM_BASE_URL", "https://api.deepseek.com"
                ),
            ),
            model=os.getenv(
                "RETRIEVAL_LLM_MODEL",
                os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
            ),
        )

    def parse(self, query: str) -> QueryPlan:
        if query in self._cache:
            return self._cache[query]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": QUERY_PARSER_PROMPT},
                {"role": "user", "content": query},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        payload = json.loads(response.choices[0].message.content)
        rewritten = payload["rewritten_query"].strip()
        retrieval_queries = tuple(
            item.strip() for item in payload["retrieval_queries"] if item.strip()
        )
        plan = QueryPlan(
            rewritten_query=rewritten,
            retrieval_queries=retrieval_queries,
            keywords=tuple(
                item.strip() for item in payload.get("keywords", []) if item.strip()
            ),
            tags=tuple(
                item.strip() for item in payload.get("tags", []) if item.strip()
            ),
        )
        self._cache[query] = plan
        return plan


class OpenAIEmbedder:
    """Batch embeddings through an OpenAI-compatible endpoint."""

    def __init__(self, client: OpenAI, model: str, dimensions: int | None = None):
        self.client = client
        self.model = model
        self.dimensions = dimensions

    @classmethod
    def from_env(cls) -> "OpenAIEmbedder":
        return cls(
            client=OpenAI(
                api_key=os.environ["MOS_EMBEDDER_API_KEY"],
                base_url=os.environ["MOS_EMBEDDER_API_BASE"],
            ),
            model=os.getenv("MOS_EMBEDDER_MODEL", "text-embedding-v4"),
            dimensions=int(os.getenv("EMBEDDING_DIMENSION", "1024")),
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimensions,
        )
        return [item.embedding for item in response.data]


class BGEReranker:
    """BGE reranking through the common /v1/rerank HTTP API."""

    def __init__(self, url: str, model: str, timeout: int = 30):
        self.url = url
        self.model = model
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "BGEReranker":
        return cls(
            url=os.getenv("BGE_RERANKER_URL", "http://localhost:8000/v1/rerank"),
            model=os.getenv(
                "BGE_RERANKER_MODEL",
                os.getenv("MOS_RERANKER_MODEL", "bge-reranker-v2-m3"),
            ),
            timeout=int(os.getenv("BGE_RERANKER_TIMEOUT", "30")),
        )

    def rerank(self, query: str, documents: list[str]) -> list[float]:
        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": len(documents),
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()

        if "results" in payload:
            scores = [0.0] * len(documents)
            for item in payload["results"]:
                scores[item["index"]] = float(
                    item.get("relevance_score", item.get("score", 0.0))
                )
            return scores

        return [float(item["score"]) for item in payload["data"]]
