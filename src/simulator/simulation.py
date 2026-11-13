from __future__ import annotations

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense
from src.risk.network_risk import NetworkRiskReport, assess_network_risk


def simulate_defense(graph: nx.DiGraph, actions: list[DefenseAction]) -> tuple[NetworkRiskReport, NetworkRiskReport]:
    """Replan attack paths against a copy of the defended graph."""
    before = assess_network_risk(graph)
    after = assess_network_risk(apply_defense(graph, actions))
    return before, after
