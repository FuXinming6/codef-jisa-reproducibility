from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense
from src.risk.network_risk import assess_network_risk


def calculate_synergy(graph: nx.DiGraph, action_a: DefenseAction, action_b: DefenseAction) -> float:
    baseline = assess_network_risk(graph).score
    delta_a = baseline - assess_network_risk(apply_defense(graph, [action_a])).score
    delta_b = baseline - assess_network_risk(apply_defense(graph, [action_b])).score
    delta_pair = baseline - assess_network_risk(apply_defense(graph, [action_a, action_b])).score
    return delta_pair - delta_a - delta_b


def portfolio_synergy(graph: nx.DiGraph, actions: Iterable[DefenseAction]) -> float:
    return sum(calculate_synergy(graph, first, second) for first, second in combinations(actions, 2))


class SynergyEvaluator:
    """Memoize pair synergies for one fixed graph during a CoDef search."""

    def __init__(self, graph: nx.DiGraph, actions: Iterable[DefenseAction]) -> None:
        self.graph = graph
        self.baseline = assess_network_risk(graph).score
        self.single_reduction = {
            action.id: self.baseline - assess_network_risk(apply_defense(graph, [action])).score
            for action in actions
        }
        self._pair_cache: dict[tuple[str, str], float] = {}

    def pair_synergy(self, first: DefenseAction, second: DefenseAction) -> float:
        key = tuple(sorted((first.id, second.id)))
        if key not in self._pair_cache:
            pair_reduction = self.baseline - assess_network_risk(apply_defense(self.graph, [first, second])).score
            self._pair_cache[key] = pair_reduction - self.single_reduction[first.id] - self.single_reduction[second.id]
        return self._pair_cache[key]

    def portfolio_synergy(self, actions: Iterable[DefenseAction]) -> float:
        return sum(self.pair_synergy(first, second) for first, second in combinations(actions, 2))
