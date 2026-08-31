import json
from dataclasses import dataclass
from pathlib import Path

from attacker.models import RouteProbe


SCHEMA_VERSION = "probe_bank_v1"


@dataclass(frozen=True)
class ProbeBank:
    graph_version: str
    cases: dict[int, tuple[RouteProbe, ...]]

    @property
    def case_indices(self) -> list[int]:
        return sorted(self.cases)

    def probes(self, case_index: int) -> tuple[RouteProbe, ...]:
        return self.cases[case_index]

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "graph_version": self.graph_version,
            "cases": {
                str(case): [probe.to_dict() for probe in probes]
                for case, probes in self.cases.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "ProbeBank":
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported Probe Bank schema")
        cases = {
            int(case): tuple(RouteProbe.from_dict(item) for item in probes)
            for case, probes in payload["cases"].items()
        }
        if not cases or any(len(probes) < 2 for probes in cases.values()):
            raise ValueError("Each Probe Bank case needs at least two probes")
        return cls(payload["graph_version"], cases)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeBank":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
