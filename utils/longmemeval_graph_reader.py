import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class LongMemEvalGraphCase:
    """
    一个 LongMemEval case = 一个问题 + 一个答案 + 一个 MemOS memory graph.
    """
    case_index: int
    user_name: str

    question_id: str | None
    question_type: str | None
    question: str
    answer: str
    question_date: str | None

    nodes: list[dict[str, Any]]
    edges: list[dict[str, Any]]

    @property
    def activated_nodes(self) -> list[dict[str, Any]]:
        return [
            node for node in self.nodes
            if node.get("status") == "activated"
        ]

    @property
    def archived_nodes(self) -> list[dict[str, Any]]:
        return [
            node for node in self.nodes
            if node.get("status") == "archived"
        ]

    @property
    def parent_edges(self) -> list[dict[str, Any]]:
        return [
            edge for edge in self.edges
            if edge.get("type") == "PARENT"
        ]

    @property
    def merged_to_edges(self) -> list[dict[str, Any]]:
        return [
            edge for edge in self.edges
            if edge.get("type") == "MERGED_TO"
        ]

    @property
    def fact_nodes(self) -> list[dict[str, Any]]:
        return [
            node for node in self.nodes
            if node.get("type") == "fact"
        ]

    @property
    def topic_nodes(self) -> list[dict[str, Any]]:
        return [
            node for node in self.nodes
            if node.get("type") == "topic"
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "case_index": self.case_index,
            "user_name": self.user_name,
            "question_id": self.question_id,
            "question_type": self.question_type,
            "question": self.question,
            "answer": self.answer,
            "num_nodes": len(self.nodes),
            "num_activated_nodes": len(self.activated_nodes),
            "num_archived_nodes": len(self.archived_nodes),
            "num_fact_nodes": len(self.fact_nodes),
            "num_topic_nodes": len(self.topic_nodes),
            "num_edges": len(self.edges),
            "num_parent_edges": len(self.parent_edges),
            "num_merged_to_edges": len(self.merged_to_edges),
        }


class LongMemEvalGraphReader:
    """
    从：
      1. MemOS 导出的 memory_graph_<version>.json
      2. LongMemEval 原始数据文件

    中读取逐个 LongMemEval case。

    约定：
        lme_exper_user_<version>_<case_index>

    例如：
        lme_exper_user_fullgraph5_0
        -> LongMemEval 原始数据中的第 0 个 case
    """

    def __init__(
        self,
        memory_graph_path: str | Path,
        longmemeval_path: str | Path,
        version: str = "fullgraph5",
    ):
        self.memory_graph_path = Path(memory_graph_path)
        self.longmemeval_path = Path(longmemeval_path)
        self.version = version

        self.graph_data = self._load_json(self.memory_graph_path)
        self.longmemeval_data = self._load_longmemeval(self.longmemeval_path)

        self.nodes = self.graph_data["nodes"]
        self.edges = self.graph_data["edges"]

        # 建立 id -> node，后续会经常用到
        self.node_by_id = {
            node["id"]: node
            for node in self.nodes
        }

        # 按 user_name 提前建立索引，避免每次 get_case 都扫描全部节点
        self.nodes_by_user: dict[str, list[dict[str, Any]]] = {}

        for node in self.nodes:
            user_name = node.get("user_name")
            if not user_name:
                continue

            self.nodes_by_user.setdefault(
                user_name, []
            ).append(node)

        # 当前 graph 文件中实际存在的 case index
        self.case_indices = self._collect_case_indices()

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _load_longmemeval(path: Path) -> list[dict[str, Any]]:
        """
        默认支持 LongMemEval 官方 JSON array 格式。
        同时兼容 JSONL。
        """
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if isinstance(data, list):
                return data

            if isinstance(data, dict) and "data" in data:
                return data["data"]

            raise ValueError(
                f"Unsupported LongMemEval JSON structure: {type(data)}"
            )

        except json.JSONDecodeError:
            # fallback: JSONL
            records = []

            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))

            return records

    def _collect_case_indices(self) -> list[int]:
        prefix = f"lme_exper_user_{self.version}_"
        indices = []

        for user_name in self.nodes_by_user:
            if not user_name.startswith(prefix):
                continue

            suffix = user_name[len(prefix):]

            if suffix.isdigit():
                indices.append(int(suffix))

        return sorted(indices)

    def __len__(self) -> int:
        """
        返回当前 memory graph 文件中实际存在多少个 case。
        """
        return len(self.case_indices)

    def get_case(self, case_index: int) -> LongMemEvalGraphCase:
        """
        按 LongMemEval 原始 case index 获取一个 case。
        """
        user_name = (
            f"lme_exper_user_{self.version}_{case_index}"
        )

        if user_name not in self.nodes_by_user:
            raise KeyError(
                f"Case {case_index} does not exist in "
                f"memory graph version={self.version}. "
                f"Available case indices: {self.case_indices}"
            )

        if case_index >= len(self.longmemeval_data):
            raise IndexError(
                f"case_index={case_index} exceeds LongMemEval "
                f"dataset size={len(self.longmemeval_data)}"
            )

        record = self.longmemeval_data[case_index]

        case_nodes = self.nodes_by_user[user_name]
        case_node_ids = {
            node["id"]
            for node in case_nodes
        }

        # 只保留两个端点都属于当前 case 的 edge
        case_edges = [
            edge
            for edge in self.edges
            if (
                edge.get("source") in case_node_ids
                and edge.get("target") in case_node_ids
            )
        ]

        return LongMemEvalGraphCase(
            case_index=case_index,
            user_name=user_name,

            question_id=record.get("question_id"),
            question_type=record.get("question_type"),
            question=record["question"],
            answer=record["answer"],
            question_date=record.get("question_date"),

            nodes=case_nodes,
            edges=case_edges,
        )

    def __getitem__(
        self,
        case_index: int,
    ) -> LongMemEvalGraphCase:
        return self.get_case(case_index)

    def available_cases(self) -> list[int]:
        return self.case_indices.copy()
