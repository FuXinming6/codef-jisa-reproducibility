"""v3.6 calibrated risk-uncertainty search.

The base simulator receives one risk value per edge.  In practice that value
is an estimate.  v3.6 evaluates the same initial policy on a deterministic
low/base/high risk envelope and adds a dispersion premium to the robust
objective.  It is deliberately transparent and does not pretend to learn
confidence from a small synthetic dataset.
"""
from __future__ import annotations

from dataclasses import replace
from math import sqrt
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    evaluate_attacker_uncertainty_portfolio,
)
from src.optimizer.dynamic_budget import (
    _delayed_actions,
    dynamic_budget_codef_search,
    dynamic_budget_greedy_baseline,
)


DEFAULT_UNCERTAINTY_LEVELS = (-0.15, 0.0, 0.15)


def risk_envelope_graph(graph: nx.DiGraph, relative_shift: float) -> nx.DiGraph:
    """Return a copied graph with every edge risk shifted and clipped."""
    shifted = graph.copy()
    for source, target, data in shifted.edges(data=True):
        risk = float(data.get("risk", 0.0))
        data["risk"] = min(1.0, max(0.0, risk * (1.0 + float(relative_shift))))
    return shifted


def _policy_ids(result: AttackerUncertaintyResult) -> frozenset[str]:
    return frozenset(action.id for action in result.initial_actions)


def _reserve(result: AttackerUncertaintyResult, fallback: float) -> float:
    return float(result.diagnostics.get("selected_reserve_budget", fallback))


def _evaluate_envelope(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    levels: Sequence[float],
    *,
    profiles: Sequence[str],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    uncertainty_weight: float,
    baseline: bool,
) -> AttackerUncertaintyResult:
    delayed = _delayed_actions(actions, deployment_window=1)
    scenario_results: list[AttackerUncertaintyResult] = []
    for level in levels:
        scenario_results.append(evaluate_attacker_uncertainty_portfolio(
            risk_envelope_graph(graph, level),
            profiles,
            delayed,
            initial_ids,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=1,
            response_pool_mode=response_pool_mode,
            method_name="v3.6-UncertaintyCalibration-Scenario",
        ))
    objectives = tuple(float(item.objective) for item in scenario_results)
    mean_objective = sum(objectives) / len(objectives)
    dispersion = sqrt(sum((value - mean_objective) ** 2 for value in objectives) / len(objectives))
    worst = max(scenario_results, key=lambda item: (item.objective, item.worst_terminal_risk))
    objective = max(objectives) + float(uncertainty_weight) * dispersion
    diagnostics = dict(worst.diagnostics)
    diagnostics.update({
        "uncertainty_calibration": True,
        "uncertainty_calibration_baseline": baseline,
        "uncertainty_levels": tuple(float(level) for level in levels),
        "uncertainty_scenario_objectives": tuple(round(value, 8) for value in objectives),
        "uncertainty_dispersion": round(dispersion, 8),
        "uncertainty_weight": float(uncertainty_weight),
        "uncertainty_objective": objective,
    })
    return AttackerUncertaintyResult(
        "v3.6-UncertaintyCalibration-Greedy" if baseline else "v3.6-UncertaintyCalibration-CoDef",
        worst.initial_actions,
        worst.profile_results,
        worst.mean_terminal_risk,
        worst.worst_terminal_risk,
        worst.risk_stddev,
        objective,
        diagnostics,
    )


def uncertainty_calibrated_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    uncertainty_levels: Sequence[float] = DEFAULT_UNCERTAINTY_LEVELS,
    uncertainty_weight: float = 0.25,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
    baseline: bool = False,
) -> AttackerUncertaintyResult:
    """Select one policy using a low/base/high edge-risk envelope."""
    if not uncertainty_levels:
        raise ValueError("uncertainty_levels must not be empty")
    seed_results: list[AttackerUncertaintyResult] = []
    for level in uncertainty_levels:
        scenario_graph = risk_envelope_graph(graph, level)
        if baseline:
            seed_results.append(dynamic_budget_greedy_baseline(
                scenario_graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget, deployment_window=1,
                response_window=1, constraints=constraints,
                response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
                response_pool_mode=response_pool_mode,
            ))
        else:
            seed_results.append(dynamic_budget_codef_search(
                scenario_graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget, deployment_window=1,
                response_window=1, constraints=constraints,
                response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
                response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
                candidate_action_ids=candidate_action_ids,
            ))
    candidates: dict[tuple[frozenset[str], float], None] = {}
    for seed in seed_results:
        candidates.setdefault((_policy_ids(seed), _reserve(seed, total_budget)), None)
    scored = [
        _evaluate_envelope(
            graph, actions, initial_ids, reserve_budget, uncertainty_levels,
            profiles=profiles, constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            uncertainty_weight=uncertainty_weight, baseline=baseline,
        )
        for (initial_ids, reserve_budget) in candidates
    ]
    selected = min(scored, key=lambda item: (item.objective, item.worst_terminal_risk, tuple(a.id for a in item.initial_actions)))
    diagnostics = dict(selected.diagnostics)
    diagnostics["uncertainty_candidate_count"] = len(scored)
    return replace(selected, diagnostics=diagnostics)


def uncertainty_calibrated_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    return uncertainty_calibrated_search(graph, actions, total_budget, baseline=True, **kwargs)
