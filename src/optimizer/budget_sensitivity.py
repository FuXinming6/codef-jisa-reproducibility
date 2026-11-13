"""v4.3 budget split sensitivity and diminishing-return selection."""
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
)
from src.optimizer.dynamic_budget import _budget_splits, _delayed_actions


def budget_sensitivity_search(
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
    candidate_action_ids: Sequence[str] | None = None,
    baseline: bool = False,
) -> AttackerUncertaintyResult:
    """Evaluate every feasible initial/response budget split explicitly."""
    splits = _budget_splits(total_budget, min_initial_budget)
    if not splits:
        raise ValueError("no feasible budget split")
    delayed = _delayed_actions(actions, deployment_window=1)
    rows: list[tuple[tuple[float, float], AttackerUncertaintyResult]] = []
    for initial_budget, reserve_budget in splits:
        if baseline:
            result = attacker_uncertainty_greedy_baseline(
                graph, profiles, delayed, initial_budget, reserve_budget,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
                response_pool_mode=response_pool_mode,
            )
        else:
            result = attacker_uncertainty_codef_search(
                graph, profiles, delayed, initial_budget, reserve_budget,
                constraints=constraints, response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths, max_response_delay=1,
                response_pool_mode=response_pool_mode, max_action_candidates=max_action_candidates,
                candidate_action_ids=candidate_action_ids,
                method_name="v4.3-BudgetSensitivity-Scenario",
            )
        rows.append(((initial_budget, reserve_budget), result))
    selected_split, selected = min(rows, key=lambda item: (
        item[1].objective, item[1].worst_terminal_risk, tuple(action.id for action in item[1].initial_actions),
    ))
    frontier = [
        {
            "initial_budget": split[0],
            "reserve_budget": split[1],
            "objective": round(result.objective, 8),
            "worst_terminal_risk": round(result.worst_terminal_risk, 8),
            "initial_ids": tuple(action.id for action in result.initial_actions),
        }
        for split, result in rows
    ]
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "budget_sensitivity": True,
        "budget_sensitivity_baseline": baseline,
        "budget_split_frontier": frontier,
        "budget_split_count": len(rows),
        "selected_initial_budget": selected_split[0],
        "selected_reserve_budget": selected_split[1],
    })
    return replace(
        selected,
        method="v4.3-BudgetSensitivity-Greedy" if baseline else "v4.3-BudgetSensitivity-CoDef",
        diagnostics=diagnostics,
    )


def budget_sensitivity_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    **kwargs: object,
) -> AttackerUncertaintyResult:
    return budget_sensitivity_search(graph, actions, total_budget, baseline=True, **kwargs)
