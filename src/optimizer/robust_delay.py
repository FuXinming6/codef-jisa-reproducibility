"""v3.3 robust policy selection across deployment-delay scenarios."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    attacker_uncertainty_codef_search,
    attacker_uncertainty_greedy_baseline,
    evaluate_attacker_uncertainty_portfolio,
)
from src.optimizer.dynamic_budget import _annotate, _delayed_actions, dynamic_budget_codef_search, dynamic_budget_greedy_baseline
from src.optimizer.robust_two_stage import _aggregate_risk


DEFAULT_DELAY_SCENARIOS = ((0, 1), (1, 1), (1, 2))


def _policy_key(result: AttackerUncertaintyResult) -> frozenset[str]:
    return frozenset(action.id for action in result.initial_actions)


def _evaluate_policy(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    deployment_window: int,
    response_window: int,
    profiles: Sequence[str],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    method_name: str,
) -> AttackerUncertaintyResult:
    delayed = _delayed_actions(actions, deployment_window)
    return evaluate_attacker_uncertainty_portfolio(
        graph,
        profiles,
        delayed,
        initial_ids,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=response_window,
        response_pool_mode=response_pool_mode,
        method_name=method_name,
    )


def robust_delay_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    delay_scenarios: Sequence[tuple[int, int]] = DEFAULT_DELAY_SCENARIOS,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
    baseline: bool = False,
) -> AttackerUncertaintyResult:
    """Select one policy by its worst objective across delay conditions."""
    if not delay_scenarios:
        raise ValueError("delay_scenarios must not be empty")
    seeds: list[AttackerUncertaintyResult] = []
    for deployment_window, response_window in delay_scenarios:
        if baseline:
            seed = dynamic_budget_greedy_baseline(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget,
                deployment_window=deployment_window, response_window=response_window,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            )
        else:
            seed = dynamic_budget_codef_search(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget,
                deployment_window=deployment_window, response_window=response_window,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
                max_action_candidates=max_action_candidates,
                candidate_action_ids=candidate_action_ids,
            )
        seeds.append(seed)
    candidates: list[tuple[frozenset[str], float, int, int]] = []
    seen: set[frozenset[str]] = set()
    for seed in seeds:
        ids = _policy_key(seed)
        if ids not in seen:
            seen.add(ids)
            candidates.append((ids, float(seed.diagnostics["selected_reserve_budget"]), int(seed.diagnostics["deployment_window"]), int(seed.diagnostics["response_window"])))
    scored: list[tuple[float, float, frozenset[str], AttackerUncertaintyResult]] = []
    for ids, reserve_budget, _, _ in candidates:
        scenario_results = []
        for deployment_window, response_window in delay_scenarios:
            scenario_results.append(_evaluate_policy(
                graph, actions, ids, reserve_budget, deployment_window, response_window,
                profiles, constraints, response_pool_size, max_phase_one_paths,
                response_pool_mode, "v3.3-RobustDelay-Scenario",
            ))
        worst = max(scenario_results, key=lambda item: (item.objective, item.worst_terminal_risk))
        scored.append((max(item.objective for item in scenario_results), max(item.worst_terminal_risk for item in scenario_results), ids, worst))
    robust_objective, robust_worst, selected_ids, selected = min(scored, key=lambda item: (item[0], item[1], tuple(sorted(item[2]))))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "robust_delay": True,
        "robust_delay_baseline": baseline,
        "delay_scenarios": tuple(tuple(item) for item in delay_scenarios),
        "robust_policy_count": len(scored),
        "robust_objective": robust_objective,
        "robust_worst_terminal_risk": robust_worst,
        "robust_initial_ids": tuple(sorted(selected_ids)),
    })
    return AttackerUncertaintyResult(
        "v3.3-RobustDelay-Greedy" if baseline else "v3.3-RobustDelay-CoDef",
        selected.initial_actions,
        selected.profile_results,
        selected.mean_terminal_risk,
        selected.worst_terminal_risk,
        selected.risk_stddev,
        robust_objective,
        diagnostics,
    )
