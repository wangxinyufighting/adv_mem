from attacker.models import RouteProbe
from defender.models import RepairPlan
from memory.models import MemoryOperation, MemoryState


class RepairController:
    """Choose ADD/MERGE and targets only from trusted provenance."""

    @staticmethod
    def plan(probe: RouteProbe, memory: MemoryState) -> RepairPlan:
        node_ids = {item.node_id for item in probe.oracle.supporting_evidence}
        source_ids = {item.source_id for item in probe.oracle.supporting_evidence}
        targets = tuple(
            node.id
            for node in memory.active_nodes
            if node_ids.intersection(node.provenance_node_ids)
            or source_ids.intersection(node.source_ids)
        )
        operation = MemoryOperation.MERGE if targets else MemoryOperation.ADD
        return RepairPlan(operation, targets)
