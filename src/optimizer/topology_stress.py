"""v3.9 adversarial topology stress testing for policy selection."""
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
from src.optimizer.dynamic_budget import _delayed_actions, dynamic_budget_greedy_baseline
from src.optimizer.pareto_policy import pareto_policy_search
from src.optimizer.synergy_pruning import synergy_pruning_search


def stress_graphs(graph: nx.DiGraph) -> tuple[tuple[str, nx.DiGraph], ...]:
    """Create deterministic stress variants without changing node semantics."""
    scenarios: list[tuple[str, nx.DiGraph]] = [("base", graph.copy())]

    global_high = graph.copy()
    for _, _, data in global_high.edges(data=True):
        data["risk"] = min(1.0, float(data.get("risk", 0.0)) * 1.15)
    scenarios.append(("global-risk-plus-15", global_high))

    ingress_high = graph.copy()
    ingress = {node for node, data in ingress_high.nodes(data=True) if data.get("type") == "internet"}
    for source, target, data in ingress_high.edges(data=True):
        if source in ingress:
            data["risk"] = min(1.0, float(data.get("risk", 0.0)) * 1.30)
    scenarios.append(("ingress-pressure", ingress_high))

    critical_high = graph.copy()
    critical = {node for node, data in critical_high.nodes(data=True) if data.get("critical")}
    for source, target, data in critical_high.edges(data=True):
        if target in critical:
            data["risk"] = min(1.0, float(data.get("risk", 0.0)) * 1.25)
    scenarios.append(("critical-asset-pressure", critical_high))

    alternate = graph.copy()
    # Increase every non-critical route slightly to make bypass paths matter.
    for source, target, data in alternate.edges(data=True):
        if not bool(alternate.nodes[target].get("critical", False)):
            data["risk"] = min(1.0, float(data.get("risk", 0.0)) * 1.10)
    scenarios.append(("lateral-bypass-pressure", alternate))
    return tuple(scenarios)


def _evaluate_stress_policy(
    scenarios: Sequence[tuple[str, nx.DiGraph]],
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    *,
    profiles: Sequence[str],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    baseline: bool,
) -> AttackerUncertaintyResult:
    delayed = _delayed_actions(actions, 1)
    results: list[AttackerUncertaintyResult] = []
    for name, graph in scenarios:
        results.append(evaluate_attacker_uncertainty_portfolio(
            graph, profiles, delayed, initial_ids, reserve_budget,
            constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
            response_pool_mode=response_pool_mode,
            method_name="v3.9-TopologyStress-Scenario",
        ))
    objectives = tuple(float(result.objective) for result in results)
    average = sum(objectives) / len(objectives)
    dispersion = sqrt(sum((value - average) ** 2 for value in objectives) / len(objectives))
    worst = max(results, key=lambda item: (item.objective, item.worst_terminal_risk))
    objective = max(objectives) + 0.2 * dispersion
    diagnostics = dict(worst.diagnostics)
    diagnostics.update({
        "topology_stress": True,
        "topology_stress_baseline": baseline,
        "stress_scenario_names": tuple(name for name, _ in scenarios),
        "stress_scenario_objectives": tuple(round(value, 8) for value in objectives),
        "stress_dispersion": round(dispersion, 8),
        "stress_objective": objective,
    })
    return AttackerUncertaintyResult(
        "v3.9-TopologyStress-Greedy" if baseline else "v3.9-TopologyStress-CoDef",
        worst.initial_actions,
        worst.profile_results,
        worst.mean_terminal_risk,
        worst.worst_terminal_risk,
        worst.risk_stddev,
        objective,
        diagnostics,
    )


def topology_stress_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    max_action_candidates: int | None = 6,
    baseline: bool = False,
) -> AttackerUncertaintyResult:
    scenarios = stress_graphs(graph)
    if baseline:
        seeds = [("greedy", dynamic_budget_greedy_baseline(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
        ))]
    else:
        seeds = [
            ("pareto", pareto_policy_search(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget, constraints=constraints,
                response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
                response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            )),
            ("synergy", synergy_pruning_search(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget, constraints=constraints,
                response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
                response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            )),
        ]
    candidates: dict[tuple[frozenset[str], float], str] = {}
    for source, seed in seeds:
        candidates.setdefault((frozenset(action.id for action in seed.initial_actions), float(seed.diagnostics.get("selected_reserve_budget", total_budget))), source)
    scored = [
        (source, _evaluate_stress_policy(
            scenarios, actions, initial_ids, reserve, profiles=profiles,
            constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            baseline=baseline,
        ))
        for (initial_ids, reserve), source in candidates.items()
    ]
    source, selected = min(scored, key=lambda item: (item[1].objective, item[1].worst_terminal_risk, tuple(a.id for a in item[1].initial_actions)))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "stress_candidate_count": len(scored),
        "stress_selected_source": source,
    })
    return replace(selected, diagnostics=diagnostics)


def topology_stress_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    return topology_stress_search(graph, actions, total_budget, baseline=True, **kwargs)
