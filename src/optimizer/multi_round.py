"""v3.1 multi-round attacker replanning.

v3.0 already chooses a delayed initial/response policy.  This module adds one
bounded emergency-response window after that first response.  The attacker is
allowed to replan again, and the defender may spend a small remaining budget on
an action touching the newly exposed path.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.graph.path_search import find_attack_paths
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerProfileResult,
    AttackerUncertaintyResult,
)
from src.optimizer.dynamic_budget import (
    _annotate,
    _delay_for,
    dynamic_budget_codef_search,
    dynamic_budget_greedy_baseline,
)
from src.optimizer.robust_two_stage import _aggregate_risk
from src.optimizer.two_stage import (
    _attacker_path_key,
    _path_probability,
    response_actions_for_region,
)


def _replan_path(graph: nx.DiGraph, foothold: str | None, profile: str) -> tuple[tuple[str, ...], float]:
    if foothold is None or foothold not in graph:
        return (), 0.0
    paths = find_attack_paths(graph, sources=[foothold])
    if not paths:
        return (), 0.0
    path = max(paths, key=lambda item: _attacker_path_key(graph, item, profile))
    risk = _path_probability(graph, path) * float(graph.nodes[path[-1]].get("asset_value", 1.0))
    return path, risk


def _emergency_action(
    graph: nx.DiGraph,
    actions: Sequence[DefenseAction],
    initial_ids: frozenset[str],
    first_response: Sequence[DefenseAction],
    observed_path: tuple[str, ...],
    replanned_path: tuple[str, ...],
    emergency_budget: float,
    response_window: int,
    constraints: PortfolioConstraints | None,
    profile: str,
    baseline_terminal: float,
) -> tuple[DefenseAction | None, tuple[str, ...], float]:
    """Select one immediate action against the newly exposed path."""
    if not replanned_path or emergency_budget <= 0:
        return None, replanned_path, 0.0
    used = set(initial_ids) | {action.id for action in first_response}
    current_graph = apply_defense(graph, [*actions_for_ids(actions, initial_ids), *first_response])
    candidates = response_actions_for_region(current_graph, list(actions), replanned_path)
    candidates = [
        action for action in candidates
        if action.id not in used
        and action.cost <= emergency_budget
        and _delay_for(action) <= response_window
    ]
    if constraints is not None:
        candidates = [
            action for action in candidates
            if constraints.is_satisfied(frozenset(used | {action.id}))
        ]
    prefix_probability = _path_probability(
        apply_defense(graph, actions_for_ids(actions, initial_ids)), observed_path
    ) if observed_path else 1.0
    base_path, base_raw = _replan_path(current_graph, replanned_path[-1] if replanned_path else None, profile)
    # The third round is optional: if its approximation cannot improve the
    # already validated two-stage result, retain that result exactly.
    best_key = (baseline_terminal, float("inf"), ())
    best: tuple[DefenseAction | None, tuple[str, ...], float] = (None, replanned_path, baseline_terminal)
    for action in candidates:
        defended = apply_defense(current_graph, [action])
        path, raw = _replan_path(defended, replanned_path[-1], profile)
        terminal = min(baseline_terminal, prefix_probability * raw)
        key = (terminal, float(action.cost), (action.id,))
        if key < best_key:
            best_key = key
            best = (action, path, terminal)
    return best


def actions_for_ids(actions: Sequence[DefenseAction], ids: frozenset[str]) -> list[DefenseAction]:
    return [action for action in actions if action.id in ids]


def _apply_round(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    seed: AttackerUncertaintyResult,
    emergency_budget: float,
    response_window: int,
    constraints: PortfolioConstraints | None,
    method_name: str,
) -> AttackerUncertaintyResult:
    profile_results: list[AttackerProfileResult] = []
    initial_ids = frozenset(action.id for action in seed.initial_actions)
    for item in seed.profile_results:
        row = item.result
        emergency, final_path, terminal = _emergency_action(
            graph,
            actions,
            initial_ids,
            row.response_actions,
            row.observed_path,
            row.replanned_path,
            emergency_budget,
            response_window,
            constraints,
            item.profile,
            row.terminal_risk,
        )
        response = tuple(row.response_actions) + ((emergency,) if emergency is not None else ())
        diagnostics = dict(row.diagnostics or {})
        diagnostics.update({
            "multi_round": True,
            "emergency_action": emergency.id if emergency is not None else None,
            "emergency_budget": emergency_budget,
            "round_count": 3,
        })
        profile_results.append(AttackerProfileResult(
            item.profile,
            replace(
                row,
                method=method_name,
                response_actions=response,
                response_cost=portfolio_cost(response),
                replanned_path=final_path,
                terminal_risk=terminal,
                objective=terminal,
                diagnostics=diagnostics,
            ),
        ))
    risks = tuple(item.result.terminal_risk for item in profile_results)
    mean_risk, worst, variation, objective = _aggregate_risk(risks, 0.5, 0.2)
    diagnostics = dict(seed.diagnostics)
    diagnostics.update({
        "multi_round": True,
        "round_count": 3,
        "emergency_budget": emergency_budget,
        "emergency_response_window": response_window,
        "emergency_actions": tuple(
            item.result.diagnostics.get("emergency_action")
            for item in profile_results
        ),
    })
    return AttackerUncertaintyResult(
        method_name,
        seed.initial_actions,
        tuple(profile_results),
        mean_risk,
        worst,
        variation,
        objective,
        diagnostics,
    )


def multi_round_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    deployment_window: int = 1,
    response_window: int = 1,
    emergency_budget: float = 1.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    seed = dynamic_budget_codef_search(
        graph, actions, total_budget,
        profiles=profiles, min_initial_budget=min_initial_budget,
        deployment_window=deployment_window, response_window=response_window,
        constraints=constraints, response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
        max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_action_ids,
    )
    result = _apply_round(
        graph,
        [_action_with_delay(action, deployment_window) for action in actions],
        seed,
        emergency_budget,
        response_window,
        constraints,
        "v3.1-MultiRound-CoDef",
    )
    # Preserve the project's established guard: an experimental extra round
    # must never make the returned policy worse than the same-condition
    # dynamic Greedy fallback.
    baseline = multi_round_greedy_baseline(
        graph, actions, total_budget,
        profiles=profiles, min_initial_budget=min_initial_budget,
        deployment_window=deployment_window, response_window=response_window,
        emergency_budget=emergency_budget, constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode,
    )
    if baseline.objective < result.objective:
        diagnostics = dict(result.diagnostics)
        diagnostics.update({
            "multi_round_fallback": True,
            "fallback_method": baseline.method,
            "pre_guard_objective": result.objective,
        })
        result = AttackerUncertaintyResult(
            baseline.method,
            baseline.initial_actions,
            baseline.profile_results,
            baseline.mean_terminal_risk,
            baseline.worst_terminal_risk,
            baseline.risk_stddev,
            baseline.objective,
            diagnostics,
        )
    return _annotate(
        result,
        total_budget=float(total_budget),
        initial_budget=float(seed.diagnostics["selected_initial_budget"]),
        reserve_budget=float(seed.diagnostics["selected_reserve_budget"]),
        deployment_window=deployment_window,
        response_window=response_window,
        split_count=int(seed.diagnostics["budget_split_count"]),
        baseline=False,
    )


def _action_with_delay(action: DefenseAction, deployment_window: int) -> DefenseAction:
    delay = _delay_for(action)
    factor = min(1.0, (deployment_window + 1.0) / (deployment_window + delay + 1.0))
    return replace(action, effect=round(float(action.effect) * factor, 8), response_delay=delay)


def multi_round_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    deployment_window: int = 1,
    response_window: int = 1,
    emergency_budget: float = 1.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
) -> AttackerUncertaintyResult:
    seed = dynamic_budget_greedy_baseline(
        graph, actions, total_budget,
        profiles=profiles, min_initial_budget=min_initial_budget,
        deployment_window=deployment_window, response_window=response_window,
        constraints=constraints, response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
    )
    result = _apply_round(
        graph,
        [_action_with_delay(action, deployment_window) for action in actions],
        seed,
        emergency_budget,
        response_window,
        constraints,
        "v3.1-MultiRound-Greedy",
    )
    return _annotate(
        result,
        total_budget=float(total_budget),
        initial_budget=float(seed.diagnostics["selected_initial_budget"]),
        reserve_budget=float(seed.diagnostics["selected_reserve_budget"]),
        deployment_window=deployment_window,
        response_window=response_window,
        split_count=int(seed.diagnostics["budget_split_count"]),
        baseline=True,
    )
