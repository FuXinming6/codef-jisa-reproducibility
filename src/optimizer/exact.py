"""Exact small-scale benchmark for validating heuristic optimizers."""
from __future__ import annotations

from itertools import combinations

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.graph.graph_builder import ingress_nodes
from src.optimizer.common import OptimizationResult, evaluate_portfolio
from src.risk.network_risk import robust_risk


def exact_enumeration_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    budget: float,
    constraints: PortfolioConstraints | None = None,
    method_name: str = "Exact Enumeration",
) -> OptimizationResult:
    """Enumerate every feasible subset; intended only for small action sets."""
    if len(actions) > 20:
        raise ValueError("Exact enumeration is limited to 20 actions")

    evaluated = 0
    feasible = 0
    best_actions: tuple[DefenseAction, ...] = ()
    best_key = (float("inf"), float("inf"), float("inf"), ())
    sources = tuple(ingress_nodes(graph))
    for size in range(len(actions) + 1):
        for portfolio in combinations(actions, size):
            evaluated += 1
            action_ids = {action.id for action in portfolio}
            cost = portfolio_cost(portfolio)
            if cost > budget or (constraints is not None and not constraints.is_satisfied(action_ids)):
                continue
            feasible += 1
            defended = apply_defense(graph, portfolio)
            key = (
                robust_risk(defended, sources),
                portfolio_cost(portfolio),
                len(portfolio),
                tuple(action.id for action in portfolio),
            )
            if key < best_key:
                best_key = key
                best_actions = portfolio
    return evaluate_portfolio(
        graph,
        best_actions,
        method_name,
        objective=best_key[0],
        diagnostics={"enumerated_portfolios": evaluated, "feasible_portfolios": feasible},
    )
