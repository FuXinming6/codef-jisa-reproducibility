from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterable, Mapping

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.graph.graph_builder import ingress_nodes
from src.risk.network_risk import assess_network_risk, robust_risk


@dataclass(frozen=True)
class OptimizationResult:
    method: str
    actions: tuple[DefenseAction, ...]
    cost: float
    before_risk: float
    after_risk: float
    worst_case_risk: float
    objective: float
    diagnostics: Mapping[str, object] | None = None


@dataclass(frozen=True)
class TwoStageResult:
    """Outcome of the offline two-stage defensive replanning simulation."""

    method: str
    initial_actions: tuple[DefenseAction, ...]
    response_actions: tuple[DefenseAction, ...]
    observed_path: tuple[str, ...]
    replanned_path: tuple[str, ...]
    foothold: str | None
    initial_cost: float
    response_cost: float
    terminal_risk: float
    objective: float
    diagnostics: Mapping[str, object] | None = None


def evaluate_portfolio(
    graph: nx.DiGraph,
    actions: Iterable[DefenseAction],
    method: str,
    objective: float | None = None,
    diagnostics: Mapping[str, object] | None = None,
) -> OptimizationResult:
    selected = tuple(actions)
    defended = apply_defense(graph, selected)
    before = assess_network_risk(graph).score
    after = assess_network_risk(defended).score
    worst = robust_risk(defended, ingress_nodes(graph))
    return OptimizationResult(
        method, selected, portfolio_cost(selected), before, after, worst,
        worst if objective is None else objective, diagnostics,
    )
