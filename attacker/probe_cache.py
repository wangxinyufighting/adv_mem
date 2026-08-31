import fcntl
import json
import os
from pathlib import Path

from attacker.models import GraphRouteBundle, RouteProbe


SCHEMA_VERSION = "probe_cache_v1"


class ProbeCache:
    """Process-safe cache for questions lazily built after route selection."""

    def __init__(self, graph_version: str, path: str | Path | None = None):
        self.graph_version = graph_version
        self.path = Path(path) if path else None
        self.probes: dict[str, RouteProbe] = {}
        self.refresh()

    def get(self, route: GraphRouteBundle) -> RouteProbe | None:
        probe = self.probes.get(route.route_id)
        if probe is None and self.path:
            self.refresh()
            probe = self.probes.get(route.route_id)
        return probe

    def put(self, probe: RouteProbe) -> RouteProbe:
        if probe.route.graph_version != self.graph_version:
            raise ValueError("Probe graph version does not match cache")
        self.probes[probe.route.route_id] = probe
        self._save()
        return probe

    def by_question(self) -> dict[str, RouteProbe]:
        return {probe.question_id: probe for probe in self.probes.values()}

    def refresh(self) -> None:
        if not self.path or not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if (
            payload.get("schema_version") != SCHEMA_VERSION
            or payload.get("graph_version") != self.graph_version
        ):
            raise ValueError("Probe cache does not match the current graph")
        self.probes = {
            route_id: RouteProbe.from_dict(item)
            for route_id, item in payload.get("probes", {}).items()
        }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        with lock_path.open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if self.path.exists():
                current = ProbeCache(self.graph_version, self.path)
                current.probes.update(self.probes)
                self.probes = current.probes
            payload = {
                "schema_version": SCHEMA_VERSION,
                "graph_version": self.graph_version,
                "probes": {
                    route_id: probe.to_dict()
                    for route_id, probe in self.probes.items()
                },
            }
            temporary = self.path.with_suffix(f"{self.path.suffix}.{os.getpid()}.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            os.replace(temporary, self.path)
