import hashlib
import json

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import re

MAX_SOURCE_LENGTH = 8000
_SOURCE_HEADER = re.compile(
    r"(?m)^[ \t]*"
    r"(?P<role>user|assistant|system|tool): "
    r"\[(?P<time>[^\]\n]+)\]:[ \t]*"
)


@dataclass(frozen=True)
class SourceLocation:
    message_id: str
    session_id: str
    session_index: int
    turn_index: int


@dataclass(frozen=True)
class SourceRecord:
    id: str
    type: str | None
    role: str | None
    chat_time: str | None
    content: str
    locations: tuple[SourceLocation, ...]


@dataclass(frozen=True)
class _RawMessage:
    role: str | None
    chat_time: str
    content: str
    location: SourceLocation


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

    source_records: tuple[SourceRecord, ...]
    sources_by_node: dict[str, tuple[SourceRecord, ...]]

    @property
    def activated_nodes(self) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes
            if node.get("status") == "activated"
        ]

    @property
    def archived_nodes(self) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes
            if node.get("status") == "archived"
        ]

    @property
    def parent_edges(self) -> list[dict[str, Any]]:
        return [
            edge
            for edge in self.edges
            if edge.get("type") == "PARENT"
        ]

    @property
    def merged_to_edges(self) -> list[dict[str, Any]]:
        return [
            edge
            for edge in self.edges
            if edge.get("type") == "MERGED_TO"
        ]

    @property
    def fact_nodes(self) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes
            if node.get("type") == "fact"
        ]

    @property
    def topic_nodes(self) -> list[dict[str, Any]]:
        return [
            node
            for node in self.nodes
            if node.get("type") == "topic"
        ]

    def sources_for(
        self,
        node_ids: Iterable[str],
    ) -> tuple[SourceRecord, ...]:
        """
        返回一组节点对应的去重原始证据。
        """
        records = []
        seen = set()

        for node_id in node_ids:
            for source in self.sources_by_node.get(node_id, ()):
                if source.id in seen:
                    continue

                seen.add(source.id)
                records.append(source)

        return tuple(records)

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
            "num_source_records": len(self.source_records),
        }


class LongMemEvalGraphReader:
    """
    读取 MemOS graph，并使用原始 LongMemEval 数据补全 sources。

    约定：
        lme_exper_user_<version>_<case_index>
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

        if self.version not in self.memory_graph_path.name:
            raise ValueError(
                f"Memory graph file name "
                f"{self.memory_graph_path.name} does not match "
                f"version={self.version}"
            )

        self.graph_data = self._load_json(
            self.memory_graph_path
        )
        self.longmemeval_data = self._load_longmemeval(
            self.longmemeval_path
        )

        self.nodes = self.graph_data["nodes"]
        self.edges = self.graph_data["edges"]

        self.node_by_id = {
            node["id"]: node
            for node in self.nodes
        }

        self.nodes_by_user: dict[
            str,
            list[dict[str, Any]],
        ] = {}

        for node in self.nodes:
            user_name = node.get("user_name")

            if user_name:
                self.nodes_by_user.setdefault(
                    user_name, []
                ).append(node)

        self.case_indices = self._collect_case_indices()

    @staticmethod
    def _load_json(path: Path) -> Any:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    @staticmethod
    def _load_longmemeval(
        path: Path,
    ) -> list[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)

            if isinstance(data, list):
                return data

            if isinstance(data, dict) and "data" in data:
                return data["data"]

            raise ValueError(
                "Unsupported LongMemEval JSON structure: "
                f"{type(data)}"
            )

        except json.JSONDecodeError:
            records = []

            with path.open("r", encoding="utf-8") as file:
                for line in file:
                    line = line.strip()

                    if line:
                        records.append(json.loads(line))

            return records

    @staticmethod
    def _normalize_content(value: Any) -> str:
        return (
            str(value or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
        )

    @staticmethod
    def _normalize_time(value: Any) -> str:
        if not value:
            return ""

        text = str(value).strip()

        if "/" in text:
            if not text.endswith(" UTC"):
                text += " UTC"

            dt = datetime.strptime(
                text,
                "%Y/%m/%d (%a) %H:%M UTC",
            ).replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromisoformat(
                text.replace("Z", "+00:00")
            )

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

        return dt.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _decode_source(value: Any) -> dict[str, Any]:
        for _ in range(3):
            if not isinstance(value, str):
                break

            value = json.loads(value)

        if not isinstance(value, dict):
            raise TypeError(
                "Source must be a JSON object, got "
                f"{type(value).__name__}"
            )

        return value

    @staticmethod
    def _source_items(
        node: dict[str, Any],
    ) -> list[Any]:
        sources = node.get("sources") or []

        if isinstance(sources, list):
            return sources

        if isinstance(sources, dict):
            return [sources]

        if isinstance(sources, str):
            try:
                parsed = json.loads(sources)
            except json.JSONDecodeError:
                return [sources]

            return (
                parsed
                if isinstance(parsed, list)
                else [parsed]
            )

        return [sources]

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

    def _raw_messages(
        self,
        record: dict[str, Any],
        case_index: int,
    ) -> list[_RawMessage]:
        sessions = record.get("haystack_sessions") or []
        dates = record.get("haystack_dates") or []
        session_ids = (
            record.get("haystack_session_ids") or []
        )

        if len(sessions) != len(dates):
            raise ValueError(
                f"Case {case_index} has "
                f"{len(sessions)} sessions but "
                f"{len(dates)} dates"
            )

        messages = []

        for session_index, session in enumerate(sessions):
            chat_time = self._normalize_time(
                dates[session_index]
            )

            session_id = (
                session_ids[session_index]
                if session_index < len(session_ids)
                else f"session_{session_index}"
            )

            for turn_index, message in enumerate(session):
                location = SourceLocation(
                    message_id=(
                        f"case_{case_index}:"
                        f"session_{session_index}:"
                        f"turn_{turn_index}"
                    ),
                    session_id=str(session_id),
                    session_index=session_index,
                    turn_index=turn_index,
                )

                messages.append(
                    _RawMessage(
                        role=message.get("role"),
                        chat_time=chat_time,
                        content=self._normalize_content(
                            message.get("content")
                        ),
                        location=location,
                    )
                )

        return messages

    @staticmethod
    def _make_source_id(
        case_index: int,
        role: str | None,
        chat_time: str | None,
        content: str,
    ) -> str:
        payload = json.dumps(
            [
                case_index,
                role,
                chat_time,
                content,
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )

        digest = hashlib.sha256(
            payload.encode("utf-8")
        ).hexdigest()[:20]

        return f"source_{digest}"
    
    @classmethod
    def _embedded_sources(
        cls,
        value: Any,
    ) -> list[dict[str, Any]]:
        """
        解析 source.content 中被重复 JSON 序列化的 source。
        """
        for _ in range(6):
            if not isinstance(value, str):
                break

            try:
                value = json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return []

        if isinstance(value, dict):
            return [value]

        if isinstance(value, list):
            records = []

            for item in value:
                if isinstance(item, dict):
                    records.append(item)
                elif isinstance(item, str):
                    records.extend(
                        cls._embedded_sources(item)
                    )

            return records

        return []

    def _expand_source(
        self,
        source: dict[str, Any],
        depth: int = 0,
    ) -> list[dict[str, Any]]:
        """
        将 source 递归还原为原子消息。

        支持：
        1. 普通原子 source
        2. content 中嵌套的 JSON source
        3. 拼接的多轮对话 source
        """
        if depth >= 8:
            return [source]

        if source.get("role") and source.get("chat_time"):
            return [source]

        embedded = self._embedded_sources(
            source.get("content")
        )

        if embedded:
            records = []

            for item in embedded:
                child = {
                    **source,
                    **item,
                }

                records.extend(
                    self._expand_source(
                        child,
                        depth + 1,
                    )
                )

            if records:
                return records

        content = self._normalize_content(
            source.get("content")
        )
        matches = list(
            _SOURCE_HEADER.finditer(content)
        )

        if not matches:
            return [source]

        if content[:matches[0].start()].strip():
            return [source]

        records = []

        for index, match in enumerate(matches):
            start = match.end()
            end = (
                matches[index + 1].start()
                if index + 1 < len(matches)
                else len(content)
            )

            message_content = content[start:end].strip()

            if not message_content:
                continue

            records.append(
                {
                    "type": "chat",
                    "role": match.group("role"),
                    "chat_time": match.group("time"),
                    "content": message_content,
                }
            )

        return records or [source]

    def _build_sources(
        self,
        case_index: int,
        nodes: list[dict[str, Any]],
        record: dict[str, Any],
    ) -> tuple[
        tuple[SourceRecord, ...],
        dict[str, tuple[SourceRecord, ...]],
    ]:
        raw_messages = self._raw_messages(
            record,
            case_index,
        )

        exact_index: dict[
            tuple[str | None, str, str],
            list[_RawMessage],
        ] = {}

        loose_index: dict[
            tuple[str | None, str],
            list[_RawMessage],
        ] = {}

        for message in raw_messages:
            contents = {
                message.content,
                message.content[:MAX_SOURCE_LENGTH],
            }

            for content in contents:
                exact_index.setdefault(
                    (
                        message.role,
                        message.chat_time,
                        content,
                    ),
                    [],
                ).append(message)

                loose_index.setdefault(
                    (
                        message.role,
                        content.strip(),
                    ),
                    [],
                ).append(message)

        source_pool: dict[
            tuple[str | None, str | None, str],
            SourceRecord,
        ] = {}

        sources_by_node: dict[
            str,
            tuple[SourceRecord, ...],
        ] = {}

        for node in nodes:
            node_sources = []
            seen = set()

            for raw_source in self._source_items(node):
                decoded = self._decode_source(raw_source)

                for source in self._expand_source(decoded):
                    role = source.get("role")
                    graph_time = self._normalize_time(
                        source.get("chat_time")
                    )
                    graph_content = self._normalize_content(
                        source.get("content")
                    )

                    if not graph_content:
                        continue

                    matches = exact_index.get(
                        (
                            role,
                            graph_time,
                            graph_content,
                        )
                    )

                    if not matches:
                        matches = loose_index.get(
                            (
                                role,
                                graph_content.strip(),
                            )
                        )

                    if matches:
                        full_contents = {
                            match.content
                            for match in matches
                        }

                        # 唯一时恢复完整原文。
                        # 多个不同全文拥有相同前缀时保留图中内容。
                        full_content = (
                            next(iter(full_contents))
                            if len(full_contents) == 1
                            else graph_content
                        )

                        raw_times = {
                            match.chat_time
                            for match in matches
                        }

                        resolved_time = (
                            next(iter(raw_times))
                            if len(raw_times) == 1
                            else graph_time or None
                        )

                        locations_by_id = {
                            match.location.message_id:
                            match.location
                            for match in matches
                        }

                        locations = tuple(
                            locations_by_id.values()
                        )
                    else:
                        # MemOS 生成但无法映射到单条原始消息的
                        # source。保留它，而不是中断读取。
                        full_content = graph_content
                        resolved_time = graph_time or None
                        locations = ()

                    pool_key = (
                        role,
                        resolved_time,
                        full_content,
                    )

                    if pool_key not in source_pool:
                        source_pool[pool_key] = SourceRecord(
                            id=self._make_source_id(
                                case_index,
                                role,
                                resolved_time,
                                full_content,
                            ),
                            type=source.get("type"),
                            role=role,
                            chat_time=resolved_time,
                            content=full_content,
                            locations=locations,
                        )

                    source_record = source_pool[pool_key]

                    if source_record.id in seen:
                        continue

                    seen.add(source_record.id)
                    node_sources.append(source_record)

            sources_by_node[node["id"]] = tuple(
                node_sources
            )

        return (
            tuple(source_pool.values()),
            sources_by_node,
        )
    
    def __len__(self) -> int:
        return len(self.case_indices)

    def get_case(
        self,
        case_index: int,
    ) -> LongMemEvalGraphCase:
        user_name = (
            f"lme_exper_user_{self.version}_{case_index}"
        )

        if user_name not in self.nodes_by_user:
            raise KeyError(
                f"Case {case_index} does not exist in "
                f"memory graph version={self.version}. "
                f"Available case indices: "
                f"{self.case_indices}"
            )

        if not 0 <= case_index < len(
            self.longmemeval_data
        ):
            raise IndexError(
                f"case_index={case_index} exceeds "
                "LongMemEval dataset size="
                f"{len(self.longmemeval_data)}"
            )

        record = self.longmemeval_data[case_index]
        case_nodes = self.nodes_by_user[user_name]

        case_node_ids = {
            node["id"]
            for node in case_nodes
        }

        case_edges = [
            edge
            for edge in self.edges
            if (
                edge.get("source") in case_node_ids
                and edge.get("target") in case_node_ids
            )
        ]

        source_records, sources_by_node = (
            self._build_sources(
                case_index,
                case_nodes,
                record,
            )
        )

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
            source_records=source_records,
            sources_by_node=sources_by_node,
        )

    def __getitem__(
        self,
        case_index: int,
    ) -> LongMemEvalGraphCase:
        return self.get_case(case_index)

    def available_cases(self) -> list[int]:
        return self.case_indices.copy()