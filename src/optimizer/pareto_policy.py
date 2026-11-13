"""v3.8 multi-objective Pareto policy selection."""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction, DefenseType
from src.defense.apply_defense import portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    evaluate_attacker_uncertainty_portfolio,
)
from src.optimizer.capacity_schedule import schedule_initial_actions
from src.optimizer.dynamic_budget import _delayed_actions, dynamic_budget_greedy_baseline
from src.optimizer.synergy_pruning import synergy_pruning_search


def _dimensions(
    result: AttackerUncertaintyResult,
    actions: Sequence[DefenseAction],
    reserve_budget: float,
) -> dict[str, float]:
    initial = tuple(result.initial_actions)
    schedule = schedule_initial_actions(initial, window_capacity=2, deadline=2)
    disruption = sum(
        1.0 if action.action_type == DefenseType.NODE_ISOLATION else
        0.5 if action.action_type == DefenseType.EDGE_RESTRICTION else 0.2
        for action in initial
    )
    committed_cost = portfolio_cost(initial)
    return {
        "risk": float(result.objective),
        "makespan": float(schedule["makespan"]),
        "cost": float(committed_cost + 0.25 * reserve_budget),
        "disruption": float(disruption),
    }


def pareto_front(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    """Return rows not dominated on every minimization dimension."""
    dimensions = ("risk", "makespan", "cost", "disruption")
    front: list[dict[str, object]] = []
    for row in rows:
        values = {key: float(row[key]) for key in dimensions}
        dominated = False
        for other in rows:
            if other is row:
                continue
            other_values = {key: float(other[key]) for key in dimensions}
            no_worse = all(other_values[key] <= values[key] + 1e-12 for key in dimensions)
            strictly_better = any(other_values[key] < values[key] - 1e-12 for key in dimensions)
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(dict(row))
    return front


def _normalize(value: float, values: Sequence[float]) -> float:
    low, high = min(values), max(values)
    return 0.0 if high - low <= 1e-12 else (value - low) / (high - low)


def pareto_policy_search(
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
    risk_weight: float = 0.65,
    time_weight: float = 0.20,
    cost_weight: float = 0.15,
    baseline: bool = False,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    """Choose a policy from a Pareto front under explicit preferences."""
    if risk_weight < 0 or time_weight < 0 or cost_weight < 0 or risk_weight + time_weight + cost_weight <= 0:
        raise ValueError("objective weights must be non-negative and not all zero")
    if baseline:
        seeds = [("greedy", dynamic_budget_greedy_baseline(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode,
        ))]
    else:
        seeds = [("synergy", synergy_pruning_search(
            graph, actions, total_budget, profiles=profiles,
            min_initial_budget=min_initial_budget, constraints=constraints,
            response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
            response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids,
        ))]

    candidate_rows: list[dict[str, object]] = []
    for source, seed in seeds:
        reserve = float(seed.diagnostics.get("selected_reserve_budget", total_budget))
        initial_ids = frozenset(action.id for action in seed.initial_actions)
        evaluated = evaluate_attacker_uncertainty_portfolio(
            graph,
            profiles,
            _delayed_actions(actions, 1),
            initial_ids,
            reserve,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=1,
            response_pool_mode=response_pool_mode,
            method_name="v3.8-Pareto-Scenario",
        )
        dimensions = _dimensions(evaluated, actions, reserve)
        candidate_rows.append({
            "source": source,
            "result": evaluated,
            "initial_ids": tuple(action.id for action in evaluated.initial_actions),
            "reserve_budget": reserve,
            **dimensions,
        })

    # Add one deliberately lighter policy when the search found a costly one.
    if not baseline and candidate_rows:
        empty = evaluate_attacker_uncertainty_portfolio(
            graph, profiles, _delayed_actions(actions, 1), frozenset(), total_budget,
            constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
            response_pool_mode=response_pool_mode, method_name="v3.8-Pareto-Scenario",
        )
        candidate_rows.append({
            "source": "empty-baseline", "result": empty,
            "initial_ids": (), "reserve_budget": total_budget,
            **_dimensions(empty, actions, total_budget),
        })

    front = pareto_front(candidate_rows)
    risk_values = [float(row["risk"]) for row in front]
    time_values = [float(row["makespan"]) for row in front]
    cost_values = [float(row["cost"]) for row in front]
    for row in front:
        row["preference_score"] = (
            risk_weight * _normalize(float(row["risk"]), risk_values)
            + time_weight * _normalize(float(row["makespan"]), time_values)
            + cost_weight * _normalize(float(row["cost"]), cost_values)
        )
    selected_row = min(front, key=lambda row: (float(row["preference_score"]), tuple(row["initial_ids"])))
    selected_result = selected_row["result"]
    diagnostics = dict(selected_result.diagnostics)
    diagnostics.update({
        "pareto_policy": True,
        "pareto_policy_baseline": baseline,
        "pareto_front_size": len(front),
        "pareto_preference_weights": (risk_weight, time_weight, cost_weight),
        "pareto_front_json": [
            {key: value for key, value in row.items() if key != "result"}
            for row in front
        ],
        "pareto_selected_source": selected_row["source"],
    })
    return replace(
        selected_result,
        method="v3.8-Pareto-Greedy" if baseline else "v3.8-Pareto-CoDef",
        objective=float(selected_row["preference_score"]),
        diagnostics=diagnostics,
    )


def pareto_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    return pareto_policy_search(graph, actions, total_budget, baseline=True, **kwargs)
