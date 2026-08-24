"""Export the first N LongMemEval cases from Neo4j as one full graph."""

import argparse
import json
import os

from collections import Counter
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.time import Date, DateTime, Duration, Time


EXCLUDED_FIELDS = {"embedding", "embedding_768", "embedding_1024", "embedding_3072"}


def case_user_names(version: str, count: int) -> list[str]:
    """Map case indices [0, count) to the user names used during ingestion."""
    return [f"lme_exper_user_{version}_{index}" for index in range(count)]


def json_value(value):
    """Convert Neo4j temporal values to JSON values."""
    if isinstance(value, (Date, DateTime, Duration, Time)):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    return value


def build_graph(session, version: str, count: int) -> dict:
    user_names = case_user_names(version, count)

    node_records = session.run(
        """
        MATCH (n:Memory)
        WHERE n.user_name IN $user_names
        RETURN n
        ORDER BY n.user_name, n.created_at, n.id
        """,
        user_names=user_names,
    )
    nodes = []
    for record in node_records:
        node = record["n"]
        properties = {
            key: json_value(value)
            for key, value in dict(node).items()
            if key not in EXCLUDED_FIELDS
        }
        properties.setdefault("id", node.element_id)
        nodes.append(properties)

    exported_users = {node["user_name"] for node in nodes}
    missing_users = set(user_names) - exported_users
    if missing_users:
        raise ValueError(f"Cases not found in Neo4j: {sorted(missing_users)}")

    edge_records = session.run(
        """
        MATCH (a:Memory)-[r]->(b:Memory)
        WHERE a.user_name IN $user_names AND b.user_name IN $user_names
        RETURN a.id AS source,
               b.id AS target,
               type(r) AS type,
               properties(r) AS properties
        ORDER BY a.user_name, a.id, b.id, type(r)
        """,
        user_names=user_names,
    )
    edges = [
        {
            "source": record["source"],
            "target": record["target"],
            "type": record["type"],
            "properties": json_value(record["properties"]),
        }
        for record in edge_records
    ]

    relation_counts = dict(sorted(Counter(edge["type"] for edge in edges).items()))
    return {
        "version": version,
        "nodes": nodes,
        "edges": edges,
        "stats": {
            "num_users": len(exported_users),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "relation_counts": relation_counts,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export LongMemEval memory graphs from Neo4j.")
    parser.add_argument("--version", required=True, help="Graph version, for example fullgraph5.")
    parser.add_argument("--num-cases", "-n", type=int, required=True, help="Number of cases.")
    parser.add_argument("--output", type=Path, help="Output JSON path.")
    args = parser.parse_args()
    if args.num_cases < 1:
        parser.error("--num-cases must be greater than 0")
    return args


def main() -> None:
    args = parse_args()
    output = args.output or Path(f"data/longmemeval/memory_graph_{args.version}.json")

    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "12345678")),
    )
    try:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j")) as session:
            graph = build_graph(session, args.version, args.num_cases)
    finally:
        driver.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")

    stats = graph["stats"]
    print(f"Exported {stats['num_users']} cases, {stats['num_nodes']} nodes, "
          f"{stats['num_edges']} edges to {output}")


if __name__ == "__main__":
    main()
