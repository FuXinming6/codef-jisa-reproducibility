from __future__ import annotations

from collections.abc import Iterable
from math import fsum

import networkx as nx

from src.defense.action import DefenseAction, DefenseType


def apply_defense(graph: nx.DiGraph, actions: Iterable[DefenseAction]) -> nx.DiGraph:
    """Return a copied graph after safely simulating a defense portfolio."""
    defended = graph.copy()
    for action in actions:
        if action.cost < 0 or not 0 <= action.effect <= 1:
            raise ValueError(f"Invalid action: {action.id}")
        if action.action_type == DefenseType.EDGE_RESTRICTION:
            source, target = action.target  # type: ignore[misc]
            if defended.has_edge(source, target):
                defended[source][target]["risk"] *= 1 - action.effect
        elif action.action_type == DefenseType.NODE_HARDENING:
            node = str(action.target)
            if node in defended:
                for source, target in list(defended.in_edges(node)) + list(defended.out_edges(node)):
                    defended[source][target]["risk"] *= 1 - action.effect
        elif action.action_type == DefenseType.NODE_ISOLATION:
            node = str(action.target)
            if node in defended:
                defended.remove_edges_from(list(defended.in_edges(node)) + list(defended.out_edges(node)))
        else:
            raise ValueError(f"Unsupported defense type: {action.action_type}")
    return defended


def portfolio_cost(actions: Iterable[DefenseAction]) -> float:
    # fsum avoids order-dependent rounding when callers pass set/frozenset
    # portfolios and keeps budget-feasibility checks stable across processes.
    return fsum(sorted(action.cost for action in actions))
