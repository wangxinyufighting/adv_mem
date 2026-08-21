import json
from dataclasses import dataclass
from pathlib import Path

from memory.models import MemoryState
from training.alternating import QuestionCandidate
from training.stop_condition import StopState


@dataclass
class RunState:
    next_round: int
    attacker_model: str
    builder_model: str
    memory: MemoryState
    questions: dict[str, QuestionCandidate]
    stop_state: StopState

    @classmethod
    def new(cls, model: str) -> "RunState":
        return cls(0, model, model, MemoryState.empty(), {}, StopState())

    @classmethod
    def load(cls, path: str | Path, model: str) -> "RunState":
        state_path = Path(path)
        if not state_path.exists():
            return cls.new(model)
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        return cls(
            next_round=payload["next_round"],
            attacker_model=payload["attacker_model"],
            builder_model=payload["builder_model"],
            memory=MemoryState.from_dict(payload["memory"]),
            questions={
                item["question_id"]: QuestionCandidate.from_dict(item)
                for item in payload["questions"]
            },
            stop_state=StopState.from_dict(payload.get("stop_state", {})),
        )

    def save(self, path: str | Path) -> None:
        state_path = Path(path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "next_round": self.next_round,
            "attacker_model": self.attacker_model,
            "builder_model": self.builder_model,
            "memory": self.memory.to_dict(),
            "questions": [item.to_dict() for item in self.questions.values()],
            "stop_state": self.stop_state.to_dict(),
        }
        state_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
