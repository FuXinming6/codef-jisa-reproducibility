"""Multi-scenario robust search for the consolidated two-stage CoDef model."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from statistics import mean
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.optimizer.baselines import greedy_baseline
from src.optimizer.common import TwoStageResult
from src.optimizer.two_stage import _dependency_closed, _result_from_evaluation, evaluate_two_stage_strategy


@dataclass(frozen=True)
class RobustTwoStageResult:
    """One initial portfolio evaluated over a fixed scenario ensemble."""

    method: str
    initial_actions: tuple[DefenseAction, ...]
    scenario_results: tuple[TwoStageResult, ...]
    mean_terminal_risk: float
    worst_terminal_risk: float
    risk_stddev: float
    objective: float
    diagnostics: dict[str, object]


def evaluate_robust_initial_portfolio(
    scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
    objective_mode: str = "mean_tail",
    method_name: str = "TwoStage-Robust-FixedPortfolio",
) -> RobustTwoStageResult:
    """Evaluate one fixed initial portfolio on every supplied scenario."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    results: list[TwoStageResult] = []
    for index, scenario in enumerate(scenarios):
        evaluation = evaluate_two_stage_strategy(
            scenario,
            actions,
            initial_ids,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_ranking="path_coverage",
            use_response_cache=True,
            response_pool_mode=response_pool_mode,
        )
        results.append(_result_from_evaluation(
            method_name,
            actions,
            initial_ids,
            evaluation,
            {"scenario_index": index, "robust_evaluation": True},
        ))
    risks = tuple(result.terminal_risk for result in results)
    average, worst, variation, objective = _aggregate_risk(
        risks, worst_weight, variation_weight, objective_mode,
    )
    return RobustTwoStageResult(
        method=method_name,
        initial_actions=tuple(action for action in actions if action.id in initial_ids),
        scenario_results=tuple(results),
        mean_terminal_risk=average,
        worst_terminal_risk=worst,
        risk_stddev=variation,
        objective=objective,
        diagnostics={
            "scenario_count": len(scenarios),
            "worst_weight": worst_weight,
            "variation_weight": variation_weight,
            "objective_mode": objective_mode,
            "response_pool_mode": response_pool_mode,
        },
    )


def robust_two_stage_greedy_baseline(
    scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
    objective_mode: str = "mean_tail",
    method_name: str = "TwoStage-Robust-Greedy",
) -> RobustTwoStageResult:
    """Choose one initial portfolio from per-scenario Greedy candidates."""
    if not scenarios:
        raise ValueError("at least one scenario is required")
    actions_by_id = {action.id: action for action in actions}
    candidates: set[frozenset[str]] = {frozenset()}
    for scenario in scenarios:
        greedy = greedy_baseline(scenario, actions, initial_budget, constraints=constraints)
        candidates.add(frozenset(action.id for action in greedy.actions))
    evaluated = [
        evaluate_robust_initial_portfolio(
            scenarios,
            actions,
            ids,
            reserve_budget,
            constraints=constraints,
            response_pool_size=response_pool_size,
            max_phase_one_paths=max_phase_one_paths,
            max_response_delay=max_response_delay,
            response_pool_mode=response_pool_mode,
            worst_weight=worst_weight,
            variation_weight=variation_weight,
            objective_mode=objective_mode,
            method_name=method_name,
        )
        for ids in candidates
        if portfolio_cost(actions_by_id[action_id] for action_id in ids) <= initial_budget
        and (constraints is None or constraints.is_satisfied(ids))
    ]
    if not evaluated:
        raise ValueError("no feasible robust Greedy candidate")
    selected = min(evaluated, key=lambda result: (
        result.objective,
        result.worst_terminal_risk,
        result.mean_terminal_risk,
        portfolio_cost(result.initial_actions),
        tuple(action.id for action in result.initial_actions),
    ))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({"candidate_count": len(evaluated), "greedy_guard": True})
    return RobustTwoStageResult(
        selected.method,
        selected.initial_actions,
        selected.scenario_results,
        selected.mean_terminal_risk,
        selected.worst_terminal_risk,
        selected.risk_stddev,
        selected.objective,
        diagnostics,
    )


def _aggregate_risk(
    risks: Sequence[float],
    worst_weight: float,
    variation_weight: float,
    objective_mode: str = "mean_tail",
) -> tuple[float, float, float, float]:
    if not risks:
        raise ValueError("at least one scenario risk is required")
    average = mean(risks)
    worst = max(risks)
    # ``statistics.stdev`` can reject numpy-like float subclasses on some
    # Python builds (``float`` has no ``numerator`` attribute).  The explicit
    # sample standard deviation is numerically equivalent here and keeps the
    # evaluator stable for larger generated topologies.
    variation = (
        sqrt(sum((float(value) - float(average)) ** 2 for value in risks) / (len(risks) - 1))
        if len(risks) > 1 else 0.0
    )
    if objective_mode == "minimax":
        objective = worst
    elif objective_mode == "mean_tail":
        objective = average + worst_weight * (worst - average) + variation_weight * variation
    else:
        raise ValueError("objective_mode must be mean_tail or minimax")
    return average, worst, variation, objective


def _result_key(
    objective: float,
    worst: float,
    average: float,
    cost: float,
    ids: frozenset[str],
) -> tuple[float, float, float, float, tuple[str, ...]]:
    return objective, worst, average, cost, tuple(sorted(ids))


def robust_two_stage_codef_search(
    scenarios: Sequence[nx.DiGraph],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    beam_width: int = 2,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
    objective_mode: str = "mean_tail",
    method_name: str = "TwoStage-CoDef-Robust",
) -> RobustTwoStageResult:
    """Search one initial portfolio against a fixed scenario ensemble.

    The objective is ``mean + worst_weight * (worst - mean) + variation_weight
    * stddev``.  Every candidate uses the same two-stage response evaluator in
    every scenario.  A robust Greedy candidate is included as a guard: it is
    selected from the per-scenario Greedy portfolios using the same aggregate
    objective, so the comparison does not give CoDef a different budget or
    information model.
    """
    if not scenarios:
        raise ValueError("at least one scenario is required")
    if initial_budget < 0 or reserve_budget < 0:
        raise ValueError("stage budgets must be non-negative")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if not 0 <= worst_weight <= 1 or variation_weight < 0:
        raise ValueError("invalid robust objective weights")
    if objective_mode not in {"mean_tail", "minimax"}:
        raise ValueError("objective_mode must be mean_tail or minimax")
    if response_pool_mode not in {"fixed", "adaptive_diverse"}:
        raise ValueError("response_pool_mode must be fixed or adaptive_diverse")

    actions_by_id = {action.id: action for action in actions}
    cache: dict[frozenset[str], tuple[tuple[TwoStageResult, ...], tuple[float, float, float, float]]] = {}

    def evaluate(ids: frozenset[str]) -> tuple[tuple[TwoStageResult, ...], tuple[float, float, float, float]]:
        if ids not in cache:
            results: list[TwoStageResult] = []
            for scenario in scenarios:
                evaluation = evaluate_two_stage_strategy(
                    scenario,
                    actions,
                    ids,
                    reserve_budget,
                    constraints=constraints,
                    response_pool_size=response_pool_size,
                    max_phase_one_paths=max_phase_one_paths,
                    max_response_delay=max_response_delay,
                    response_ranking="path_coverage",
                    use_response_cache=True,
                    response_pool_mode=response_pool_mode,
                )
                results.append(_result_from_evaluation(
                    method_name,
                    actions,
                    ids,
                    evaluation,
                    {"scenario_index": len(results), "robust_evaluation": True},
                ))
            risks = tuple(result.terminal_risk for result in results)
            cache[ids] = (tuple(results), _aggregate_risk(
                risks, worst_weight, variation_weight, objective_mode,
            ))
        return cache[ids]

    greedy_ids_by_scenario: list[frozenset[str]] = []
    for scenario in scenarios:
        greedy = greedy_baseline(scenario, actions, initial_budget, constraints=constraints)
        greedy_ids_by_scenario.append(frozenset(action.id for action in greedy.actions))
    guard_ids = set(greedy_ids_by_scenario)
    guard_ids.add(frozenset())
    feasible_guard_ids = {
        ids for ids in guard_ids
        if portfolio_cost(actions_by_id[action_id] for action_id in ids) <= initial_budget
        and (constraints is None or constraints.is_satisfied(ids))
    }
    if not feasible_guard_ids:
        raise ValueError("no feasible robust Greedy guard portfolio")

    def ranking(ids: frozenset[str]) -> tuple[float, float, float, float, tuple[str, ...]]:
        _, aggregate = evaluate(ids)
        average, worst, _, objective = aggregate
        return _result_key(
            objective,
            worst,
            average,
            portfolio_cost(actions_by_id[action_id] for action_id in ids),
            ids,
        )

    robust_greedy_ids = min(feasible_guard_ids, key=ranking)
    beam: list[frozenset[str]] = [frozenset()]
    seen = set(beam)
    feasible_initials = set(feasible_guard_ids)
    expanded = 0
    pruned_budget = 0
    pruned_constraint = 0
    while beam:
        candidates: set[frozenset[str]] = set()
        for ids in beam:
            for action in actions:
                expanded += 1
                candidate = _dependency_closed(ids | {action.id}, constraints)
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected = [actions_by_id[action_id] for action_id in candidate]
                if portfolio_cost(selected) > initial_budget:
                    pruned_budget += 1
                    continue
                if constraints is not None and not constraints.is_satisfied(candidate):
                    pruned_constraint += 1
                    continue
                candidates.add(candidate)
                feasible_initials.add(candidate)
        if not candidates:
            break
        beam = sorted(candidates, key=ranking)[:beam_width]

    best_ids = min(feasible_initials, key=ranking)
    scenario_results, aggregate = evaluate(best_ids)
    average, worst, variation, objective = aggregate
    return RobustTwoStageResult(
        method=method_name,
        initial_actions=tuple(actions_by_id[action_id] for action_id in sorted(best_ids)),
        scenario_results=scenario_results,
        mean_terminal_risk=average,
        worst_terminal_risk=worst,
        risk_stddev=variation,
        objective=objective,
        diagnostics={
            "scenario_count": len(scenarios),
            "worst_weight": worst_weight,
            "variation_weight": variation_weight,
            "objective_mode": objective_mode,
            "robust_greedy_initial_actions": tuple(sorted(robust_greedy_ids)),
            "seeded_by_robust_greedy": True,
            "expanded_initial_candidates": expanded,
            "evaluated_initial_portfolios": len(cache),
            "pruned_initial_budget": pruned_budget,
            "pruned_initial_constraint": pruned_constraint,
            "response_pool_mode": response_pool_mode,
        },
    )
