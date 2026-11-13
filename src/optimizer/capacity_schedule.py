"""v3.4 maintenance-window capacity aware policy selection.

v3.0 made deployment delay explicit and v3.3 selected a policy that is robust
to several delay conditions.  This module adds the operational constraint that
actions must share a finite maintenance window.  A portfolio is therefore
scored by both its simulated terminal risk and the amount by which its initial
deployment exceeds a deadline.

The schedule is deliberately deterministic and offline: it is not a live
change-management system.  Its purpose is to expose a realistic feasibility
signal to the optimizer without hiding the original two-stage evaluator.
"""
from __future__ import annotations

from dataclasses import replace
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
    _budget_splits,
    _delay_for,
    _delayed_actions,
    dynamic_budget_codef_search,
    dynamic_budget_greedy_baseline,
)
from src.optimizer.robust_delay import DEFAULT_DELAY_SCENARIOS


def schedule_initial_actions(
    actions: Sequence[DefenseAction],
    *,
    window_capacity: int = 2,
    deadline: int = 2,
) -> dict[str, object]:
    """Place actions in the earliest feasible discrete maintenance slots.

    Each action occupies ``delay`` slots.  The returned record is stable across
    runs and is useful in reports because it explains why a policy received a
    scheduling penalty.
    """
    capacity = max(1, int(window_capacity))
    due = max(0, int(deadline))
    occupancy: list[int] = []
    placements: list[dict[str, object]] = []

    # Fast, high-impact actions get an earlier chance, while IDs break ties.
    ordered = sorted(actions, key=lambda item: (_delay_for(item), -float(item.effect), item.id))
    for action in ordered:
        duration = max(1, _delay_for(action))
        start = 0
        while True:
            end = start + duration
            if len(occupancy) < end:
                occupancy.extend([0] * (end - len(occupancy)))
            if all(occupancy[slot] < capacity for slot in range(start, end)):
                for slot in range(start, end):
                    occupancy[slot] += 1
                placements.append({
                    "id": action.id,
                    "start": start,
                    "finish": end,
                    "delay": duration,
                })
                break
            start += 1

    makespan = max((int(item["finish"]) for item in placements), default=0)
    late_slots = max(0, makespan - due)
    return {
        "window_capacity": capacity,
        "deadline": due,
        "makespan": makespan,
        "late_slots": late_slots,
        "placements": tuple(placements),
    }


def _policy_key(result: AttackerUncertaintyResult) -> frozenset[str]:
    return frozenset(action.id for action in result.initial_actions)


def _evaluate_policy(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    deployment_window: int,
    response_window: int,
    *,
    profiles: Sequence[str],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    method_name: str,
) -> AttackerUncertaintyResult:
    return evaluate_attacker_uncertainty_portfolio(
        graph,
        profiles,
        _delayed_actions(actions, deployment_window),
        initial_ids,
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=response_window,
        response_pool_mode=response_pool_mode,
        method_name=method_name,
    )


def _score_candidate(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    delay_scenarios: Sequence[tuple[int, int]],
    *,
    profiles: Sequence[str],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    window_capacity: int,
    deadline: int,
    late_penalty: float,
    baseline: bool,
) -> AttackerUncertaintyResult:
    scenario_results = [
        _evaluate_policy(
            graph, actions, initial_ids, reserve_budget, deployment_window, response_window,
            profiles=profiles, constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            method_name="v3.4-CapacitySchedule-Scenario",
        )
        for deployment_window, response_window in delay_scenarios
    ]
    worst = max(scenario_results, key=lambda item: (item.objective, item.worst_terminal_risk))
    schedule = schedule_initial_actions(
        tuple(action for action in actions if action.id in initial_ids),
        window_capacity=window_capacity,
        deadline=deadline,
    )
    penalty = round(float(schedule["late_slots"]) * float(late_penalty), 8)
    objective = float(max(item.objective for item in scenario_results)) + penalty
    diagnostics = dict(worst.diagnostics)
    diagnostics.update({
        "capacity_schedule": True,
        "capacity_schedule_baseline": baseline,
        "delay_scenarios": tuple(tuple(item) for item in delay_scenarios),
        "capacity_objective": objective,
        "risk_objective": max(item.objective for item in scenario_results),
        "schedule_penalty": penalty,
        "schedule_json": schedule,
    })
    return AttackerUncertaintyResult(
        "v3.4-CapacitySchedule-Greedy" if baseline else "v3.4-CapacitySchedule-CoDef",
        worst.initial_actions,
        worst.profile_results,
        worst.mean_terminal_risk,
        worst.worst_terminal_risk,
        worst.risk_stddev,
        objective,
        diagnostics,
    )


def _candidate_seeds(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str],
    min_initial_budget: float,
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_phase_one_paths: int | None,
    response_pool_mode: str,
    max_action_candidates: int | None,
    candidate_action_ids: Sequence[str] | None,
    baseline: bool,
) -> list[AttackerUncertaintyResult]:
    seeds: list[AttackerUncertaintyResult] = []
    for deployment_window, response_window in DEFAULT_DELAY_SCENARIOS:
        if baseline:
            seeds.append(dynamic_budget_greedy_baseline(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget,
                deployment_window=deployment_window, response_window=response_window,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            ))
        else:
            seeds.append(dynamic_budget_codef_search(
                graph, actions, total_budget, profiles=profiles,
                min_initial_budget=min_initial_budget,
                deployment_window=deployment_window, response_window=response_window,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
                max_action_candidates=max_action_candidates,
                candidate_action_ids=candidate_action_ids,
            ))
    # Include the empty portfolio as a valid, explainable fallback.
    seeds.append(AttackerUncertaintyResult(
        "v3.4-empty-seed", (), (), 0.0, 0.0, 0.0, 0.0,
        {"selected_reserve_budget": float(total_budget)},
    ))
    return seeds


def capacity_schedule_search(
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
    window_capacity: int = 2,
    deadline: int = 2,
    late_penalty: float = 0.01,
    baseline: bool = False,
) -> AttackerUncertaintyResult:
    """Choose a policy that is robust to delay and feasible in a work window."""
    seeds = _candidate_seeds(
        graph, actions, total_budget, profiles=profiles,
        min_initial_budget=min_initial_budget, constraints=constraints,
        response_pool_size=response_pool_size, max_phase_one_paths=max_phase_one_paths,
        response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
        candidate_action_ids=candidate_action_ids, baseline=baseline,
    )
    candidates: dict[frozenset[str], float] = {}
    for seed in seeds:
        candidates.setdefault(_policy_key(seed), float(seed.diagnostics.get("selected_reserve_budget", total_budget)))
    scored = [
        _score_candidate(
            graph, actions, ids, reserve, delay_scenarios,
            profiles=profiles, constraints=constraints, response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths, response_pool_mode=response_pool_mode,
            window_capacity=window_capacity, deadline=deadline, late_penalty=late_penalty,
            baseline=baseline,
        )
        for ids, reserve in candidates.items()
    ]
    selected = min(scored, key=lambda item: (
        item.objective, item.worst_terminal_risk, tuple(action.id for action in item.initial_actions),
    ))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "capacity_candidate_count": len(scored),
        "capacity_window_capacity": int(window_capacity),
        "capacity_deadline": int(deadline),
        "capacity_late_penalty": float(late_penalty),
    })
    return replace(selected, diagnostics=diagnostics)


def capacity_schedule_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    """Fair comparator using the same capacity-aware objective."""
    return capacity_schedule_search(graph, actions, total_budget, baseline=True, **kwargs)
