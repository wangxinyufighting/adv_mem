import json
from dataclasses import dataclass
from pathlib import Path

from memory.models import MemoryState


PIPELINE_VERSION = "minimal_memory_loop_v1"


@dataclass
class CaseRunState:
    memory: MemoryState

    @classmethod
    def new(cls) -> "CaseRunState":
        return cls(MemoryState.empty())

    def to_dict(self) -> dict:
        return {"memory": self.memory.to_dict()}

    @classmethod
    def from_dict(cls, payload: dict) -> "CaseRunState":
        return cls(MemoryState.from_dict(payload["memory"]))


@dataclass
class RunState:
    next_round: int
    attacker_model: str
    builder_model: str
    cases: dict[int, CaseRunState]

    @classmethod
    def new(cls, model: str, case_indices: list[int]) -> "RunState":
        return cls(
            0,
            model,
            model,
            {case: CaseRunState.new() for case in case_indices},
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        model: str,
        case_indices: list[int],
    ) -> "RunState":
        state_path = Path(path)
        if not state_path.exists():
            return cls.new(model, case_indices)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        if payload.get("pipeline_version") != PIPELINE_VERSION:
            raise ValueError("Incompatible run_state.json; use a new --work-dir")
        cases = {
            int(case): CaseRunState.from_dict(state)
            for case, state in payload["cases"].items()
        }
        if set(cases) != set(case_indices):
            raise ValueError("Probe Bank cases do not match run_state.json")
        return cls(
            payload["next_round"],
            payload["attacker_model"],
            payload["builder_model"],
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
