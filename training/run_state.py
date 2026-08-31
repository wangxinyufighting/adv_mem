import json
from dataclasses import dataclass
from pathlib import Path

from attacker.models import RouterState
from memory.models import MemoryState


PIPELINE_VERSION = "dynamic_graph_route_loop_v2"


@dataclass
class CaseRunState:
    memory: MemoryState
    node_visit_counts: dict[str, int]

    @classmethod
    def new(cls) -> "CaseRunState":
        return cls(MemoryState.empty(), {})

    def to_dict(self) -> dict:
        return {
            "memory": self.memory.to_dict(),
            "node_visit_counts": self.node_visit_counts,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CaseRunState":
        return cls(
            MemoryState.from_dict(payload["memory"]),
            dict(payload.get("node_visit_counts", {})),
        )

    def router_state(self) -> RouterState:
        return RouterState(dict(self.node_visit_counts))


@dataclass
class RunState:
    next_round: int
    attacker_model: str
    builder_model: str
    graph_version: str
    cases: dict[int, CaseRunState]

    @classmethod
    def new(
        cls,
        model: str,
        case_indices: list[int],
        graph_version: str,
    ) -> "RunState":
        return cls(
            0,
            model,
            model,
            graph_version,
            {case: CaseRunState.new() for case in case_indices},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: str,
        case_indices: list[int],
        graph_version: str,
    ) -> "RunState":
        state_path = Path(path)
        if not state_path.exists():
            return cls.new(model, case_indices, graph_version)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("Incompatible run_state.json; use a new --work-dir")
        if payload.get("graph_version") != graph_version:
            raise ValueError("Graph version does not match run_state.json")
        cases = {
            int(case): CaseRunState.from_dict(state)
            for case, state in payload["cases"].items()
        }
        if set(cases) != set(case_indices):
            raise ValueError("Graph cases do not match run_state.json")
        return cls(
            payload["next_round"],
            payload["attacker_model"],
            payload["builder_model"],
            payload["graph_version"],
            cases,
        )

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "pipeline_version": PIPELINE_VERSION,
                    "next_round": self.next_round,
                    "attacker_model": self.attacker_model,
                    "builder_model": self.builder_model,
                    "graph_version": self.graph_version,
                    "cases": {
                        str(case): state.to_dict()
                        for case, state in self.cases.items()
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
