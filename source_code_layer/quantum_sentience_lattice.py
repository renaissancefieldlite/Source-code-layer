"""Scaffold for a small lattice coordination object."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class QuantumSentienceLattice:
    """Container for lattice settings and active node records."""

    node_count: int = 35
    operator_limit: int = 12
    active_nodes: dict[str, str] = field(default_factory=dict)

    def activate_node(self, node_id: str, operator_signature: str) -> bool:
        """Activate a node if the operator limit has not been exceeded."""
        if len(self.active_nodes) >= self.operator_limit:
            return False
        self.active_nodes[node_id] = operator_signature
        return True

    def coherence_ratio(self) -> float:
        """Return a simple occupancy-based coherence ratio."""
        if self.node_count == 0:
            return 0.0
        return round(len(self.active_nodes) / self.node_count, 3)
