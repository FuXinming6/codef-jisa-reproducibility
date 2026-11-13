"""v3.0 rolling budget search with explicit deployment latency.

This module is intentionally separate from the v2.x evaluator.  It keeps the
existing two-stage search as the inner solver, but makes two previously static
assumptions explicit:

* the total defense budget may be split between the initial and response
  stages instead of being fixed to one split; and
* a defense action has a deployment delay, so its effective protection is
  discounted during the current response window and slow actions may be
  unavailable to an immediate response.

The model remains an offline graph simulation.  It does not apply changes to
live infrastructure.  v3.0 is a bounded research variant, not a replacement
for the validated v2.x protocol.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction, DefenseType
from src.defense.constraints import PortfolioConstraints
from src.optimizer.attacker_uncertainty import (
    ATTACKER_PROFILES,
    AttackerUncertaintyResult,
    attacker_uncertainty_codef_search,
    attacker_uncertainty_greedy_baseline,
)


DEFAULT_DEPLOYMENT_DELAY = {
    DefenseType.NODE_HARDENING: 2,
    DefenseType.EDGE_RESTRICTION: 1,
    DefenseType.NODE_ISOLATION: 3,
}


def _delay_for(action: DefenseAction) -> int:
    """Use configured latency when present, otherwise a v3.0 type default."""
    configured = int(action.response_delay)
    if configured > 0:
        return configured
    return int(DEFAULT_DEPLOYMENT_DELAY[action.action_type])


def _effective_action(action: DefenseAction, deployment_window: int) -> DefenseAction:
    """Return a delayed action with a bounded current-window effect.

    An action that takes longer to deploy still has value, but only part of
    its advertised effect is available in the current rolling window.  The
    factor is deterministic and monotone in delay, which makes experiments
    reproducible and easy to explain in a report.
    """
    delay = _delay_for(action)
    window = max(0, int(deployment_window))
    factor = min(1.0, (window + 1.0) / (window + delay + 1.0))
    return replace(
        action,
        effect=round(float(action.effect) * factor, 8),
        response_delay=delay,
    )


def _delayed_actions(actions: Sequence[DefenseAction], deployment_window: int) -> list[DefenseAction]:
    return [_effective_action(action, deployment_window) for action in actions]


def _budget_splits(total_budget: float, min_initial_budget: float) -> tuple[tuple[float, float], ...]:
    """Enumerate reproducible integer-cost budget splits."""
    total = float(total_budget)
    lower = max(0.0, min(float(min_initial_budget), total))
    upper = total - lower
    if total.is_integer() and lower.is_integer():
        values = range(int(lower), int(upper) + 1)
        return tuple((float(initial), round(total - initial, 6)) for initial in values)
    # Non-integer budgets are supported with a small deterministic grid.
    step = 0.5
    count = int(round((upper - lower) / step))
    return tuple(
        (round(lower + index * step, 6), round(total - lower - index * step, 6))
        for index in range(count + 1)
    )


def _result_key(result: AttackerUncertaintyResult) -> tuple[float, float, float, tuple[str, ...]]:
    return (
        float(result.objective),
        float(result.worst_terminal_risk),
        float(result.mean_terminal_risk),
        tuple(action.id for action in result.initial_actions),
    )


def _annotate(
    result: AttackerUncertaintyResult,
    *,
    total_budget: float,
    initial_budget: float,
    reserve_budget: float,
    deployment_window: int,
    response_window: int,
    split_count: int,
    baseline: bool,
) -> AttackerUncertaintyResult:
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "dynamic_budget": True,
        "dynamic_budget_baseline": baseline,
        "total_budget": total_budget,
        "selected_initial_budget": initial_budget,
        "selected_reserve_budget": reserve_budget,
        "budget_split_count": split_count,
        "deployment_window": deployment_window,
        "response_window": response_window,
        "delay_policy": "configured_or_type_default",
        "delay_defaults": {key.value: value for key, value in DEFAULT_DEPLOYMENT_DELAY.items()},
    })
    return AttackerUncertaintyResult(
        result.method,
        result.initial_actions,
        result.profile_results,
        result.mean_terminal_risk,
        result.worst_terminal_risk,
        result.risk_stddev,
        result.objective,
        diagnostics,
    )


def dynamic_budget_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    deployment_window: int = 1,
    response_window: int = 1,
    constraints: PortfolioConstraints | None = None,
    beam_width: int = 2,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
    max_action_candidates: int | None = 6,
    candidate_action_ids: Sequence[str] | None = None,
) -> AttackerUncertaintyResult:
    """Choose both the initial budget and portfolio under delayed actions."""
    if total_budget < 0:
        raise ValueError("total_budget must be non-negative")
    splits = _budget_splits(total_budget, min_initial_budget)
    if not splits:
        raise ValueError("no feasible dynamic budget split")
    delayed = _delayed_actions(actions, deployment_window)
    results: list[tuple[tuple[float, float], AttackerUncertaintyResult]] = []
    for initial_budget, reserve_budget in splits:
        result = attacker_uncertainty_codef_search(
            graph,
            profiles,
            delayed,
            initial_budget=initial_budget,
            reserve_budget=reserve_budget,
            constraints=constraints,
            beam_width=beam_width,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=response_window,
            response_pool_mode=response_pool_mode,
            worst_weight=worst_weight,
            variation_weight=variation_weight,
            max_action_candidates=max_action_candidates,
            candidate_action_ids=candidate_action_ids,
            method_name="v3.0-DynamicBudget-CoDef",
        )
        results.append(((initial_budget, reserve_budget), result))
    split, best = min(results, key=lambda item: _result_key(item[1]))
    return _annotate(
        best,
        total_budget=float(total_budget),
        initial_budget=split[0],
        reserve_budget=split[1],
        deployment_window=deployment_window,
        response_window=response_window,
        split_count=len(splits),
        baseline=False,
    )


def dynamic_budget_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    total_budget: float,
    *,
    profiles: Sequence[str] = ATTACKER_PROFILES,
    min_initial_budget: float = 2.0,
    deployment_window: int = 1,
    response_window: int = 1,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
) -> AttackerUncertaintyResult:
    """Fair baseline: Greedy is also allowed to choose the budget split."""
    splits = _budget_splits(total_budget, min_initial_budget)
    delayed = _delayed_actions(actions, deployment_window)
    results: list[tuple[tuple[float, float], AttackerUncertaintyResult]] = []
    for initial_budget, reserve_budget in splits:
        result = attacker_uncertainty_greedy_baseline(
            graph,
            profiles,
            delayed,
            initial_budget=initial_budget,
            reserve_budget=reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=response_window,
            response_pool_mode=response_pool_mode,
            worst_weight=worst_weight,
            variation_weight=variation_weight,
        )
        results.append(((initial_budget, reserve_budget), result))
    split, best = min(results, key=lambda item: _result_key(item[1]))
    return _annotate(
        best,
        total_budget=float(total_budget),
        initial_budget=split[0],
        reserve_budget=split[1],
        deployment_window=deployment_window,
        response_window=response_window,
        split_count=len(splits),
        baseline=True,
    )
