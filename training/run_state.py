import json
from dataclasses import dataclass
from pathlib import Path

from memory.models import MemoryState
from training.alternating import QuestionCandidate
from training.stop_condition import StopState


@dataclass
class CaseRunState:
    memory: MemoryState
    questions: dict[str, QuestionCandidate]
    stop_state: StopState
    stopped: bool = False

    @classmethod
    def new(cls) -> "CaseRunState":
        return cls(MemoryState.empty(), {}, StopState())

    def to_dict(self) -> dict:
        return {
            "memory": self.memory.to_dict(),
            "questions": [item.to_dict() for item in self.questions.values()],
            "stop_state": self.stop_state.to_dict(),
            "stopped": self.stopped,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "CaseRunState":
        return cls(
            memory=MemoryState.from_dict(payload["memory"]),
            questions={
                item["question_id"]: QuestionCandidate.from_dict(item)
                for item in payload["questions"]
            },
            stop_state=StopState.from_dict(payload["stop_state"]),
            stopped=payload["stopped"],
        )


@dataclass
class RunState:
    next_round: int
    attacker_model: str
    builder_model: str
    cases: dict[int, CaseRunState]
    attacker_role: str = "route_selector_v1"

    @classmethod
    def new(cls, model: str, case_indices: list[int]) -> "RunState":
        return cls(
            next_round=0,
            attacker_model=model,
            builder_model=model,
            cases={case_index: CaseRunState.new() for case_index in case_indices},
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
        role = payload.get("attacker_role")
        return cls(
            next_round=payload["next_round"],
            # Old checkpoints generated questions rather than selecting routes.
            attacker_model=(
                payload["attacker_model"]
                if role == "route_selector_v1"
                else model
            ),
            builder_model=payload["builder_model"],
            cases={
                int(case_index): CaseRunState.from_dict(case_state)
                for case_index, case_state in payload["cases"].items()
            },
            attacker_role="route_selector_v1",
        )

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_round": self.next_round,
            "attacker_model": self.attacker_model,
            "attacker_role": self.attacker_role,
            "builder_model": self.builder_model,
            "cases": {
                str(case_index): case_state.to_dict()
                for case_index, case_state in self.cases.items()
            },
        }
        state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
