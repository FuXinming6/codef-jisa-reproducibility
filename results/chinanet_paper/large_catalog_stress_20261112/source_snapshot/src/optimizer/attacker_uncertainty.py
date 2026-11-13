"""Robust two-stage search under multiple attacker behavior profiles."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from statistics import mean, stdev
from typing import Sequence

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.optimizer.baselines import greedy_baseline
from src.optimizer.common import TwoStageResult
from src.optimizer.robust_two_stage import _aggregate_risk
from src.optimizer.two_stage import _dependency_closed, _result_from_evaluation, evaluate_two_stage_strategy
from src.risk.network_risk import assess_network_risk


ATTACKER_PROFILES = ("risk_max", "shortest_path", "asset_targeted")


@dataclass(frozen=True)
class AttackerProfileResult:
    profile: str
    result: TwoStageResult


@dataclass(frozen=True)
class AttackerUncertaintyResult:
    method: str
    initial_actions: tuple[DefenseAction, ...]
    profile_results: tuple[AttackerProfileResult, ...]
    mean_terminal_risk: float
    worst_terminal_risk: float
    risk_stddev: float
    objective: float
    diagnostics: dict[str, object]


def _validate_profiles(profiles: Sequence[str]) -> tuple[str, ...]:
    normalized = tuple(dict.fromkeys(profiles))
    if not normalized:
        raise ValueError("at least one attacker profile is required")
    unknown = set(normalized) - set(ATTACKER_PROFILES)
    if unknown:
        raise ValueError(f"unknown attacker profiles: {sorted(unknown)}")
    return normalized


def evaluate_attacker_uncertainty_portfolio(
    graph: nx.DiGraph,
    profiles: Sequence[str],
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
    method_name: str = "TwoStage-Attacker-Uncertainty",
    response_scope: str = "local",
    replan_attacker: bool = True,
) -> AttackerUncertaintyResult:
    """Evaluate one initial portfolio for every attacker profile."""
    profiles = _validate_profiles(profiles)
    profile_results: list[AttackerProfileResult] = []
    for profile in profiles:
        evaluation = evaluate_two_stage_strategy(
            graph,
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
            attacker_profile=profile,
            response_scope=response_scope,
            replan_attacker=replan_attacker,
        )
        profile_results.append(AttackerProfileResult(
            profile,
            _result_from_evaluation(
                method_name,
                actions,
                initial_ids,
                evaluation,
                {"attacker_profile": profile, "attacker_uncertainty": True},
            ),
        ))
    risks = tuple(item.result.terminal_risk for item in profile_results)
    average, worst, variation, objective = _aggregate_risk(risks, worst_weight, variation_weight)
    return AttackerUncertaintyResult(
        method=method_name,
        initial_actions=tuple(action for action in actions if action.id in initial_ids),
        profile_results=tuple(profile_results),
        mean_terminal_risk=average,
        worst_terminal_risk=worst,
        risk_stddev=variation,
        objective=objective,
        diagnostics={
            "profiles": profiles,
            "profile_count": len(profiles),
            "worst_weight": worst_weight,
            "variation_weight": variation_weight,
            "response_pool_mode": response_pool_mode,
            "response_scope": response_scope,
            "replan_attacker": replan_attacker,
        },
    )


def attacker_uncertainty_greedy_baseline(
    graph: nx.DiGraph,
    profiles: Sequence[str],
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
) -> AttackerUncertaintyResult:
    """Score the same static Greedy initial portfolio under all profiles."""
    greedy = greedy_baseline(graph, actions, initial_budget, constraints=constraints)
    return evaluate_attacker_uncertainty_portfolio(
        graph,
        profiles,
        actions,
        frozenset(action.id for action in greedy.actions),
        reserve_budget,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_pool_mode=response_pool_mode,
        worst_weight=worst_weight,
        variation_weight=variation_weight,
        method_name="TwoStage-Attacker-Uncertainty-Greedy",
    )


def attacker_uncertainty_multistart_greedy_baseline(
    graph: nx.DiGraph,
    profiles: Sequence[str],
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
    start_size: int = 1,
    memoize_static_scores: bool = False,
) -> AttackerUncertaintyResult:
    """Evaluate static Greedy completions from empty, single, or pair starts.

    Every resulting portfolio is judged using CoDef's same multi-profile,
    two-stage terminal objective.  This is a stronger comparator than one
    fixed Greedy start, but it has no Beam expansion.
    """
    if start_size not in {1, 2}:
        raise ValueError("start_size must be one or two")
    profiles = _validate_profiles(profiles)
    actions_by_id = {action.id: action for action in actions}
    static_score_cache: dict[frozenset[str], float] = {}
    static_score_cache_hits = 0

    def static_score(ids: frozenset[str]) -> float:
        nonlocal static_score_cache_hits
        if memoize_static_scores and ids in static_score_cache:
            static_score_cache_hits += 1
            return static_score_cache[ids]
        score = assess_network_risk(
            apply_defense(graph, (actions_by_id[item] for item in ids))
        ).score
        if memoize_static_scores:
            static_score_cache[ids] = score
        return score

    def feasible(ids: frozenset[str]) -> bool:
        return portfolio_cost(actions_by_id[item] for item in ids) <= initial_budget and (
            constraints is None or constraints.is_satisfied(ids)
        )

    def complete(start_ids: frozenset[str]) -> frozenset[str] | None:
        ids = _dependency_closed(start_ids, constraints)
        if not feasible(ids):
            return None
        while True:
            current_score = static_score(ids)
            proposals: list[tuple[float, str, frozenset[str]]] = []
            for action in actions:
                if action.id in ids:
                    continue
                candidate = _dependency_closed(ids | {action.id}, constraints)
                if candidate == ids or not feasible(candidate):
                    continue
                extra_cost = portfolio_cost(actions_by_id[item] for item in candidate - ids)
                if extra_cost <= 0:
                    continue
                score = static_score(candidate)
                proposals.append(((current_score - score) / extra_cost, action.id, candidate))
            if not proposals:
                return ids
            improvement, _, next_ids = max(proposals, key=lambda item: (item[0], item[1]))
            if improvement <= 0:
                return ids
            ids = next_ids

    starts = [frozenset()]
    starts.extend(frozenset((action.id,)) for action in actions)
    if start_size == 2:
        starts.extend(frozenset((left.id, right.id)) for left, right in combinations(actions, 2))
    portfolios = {
        candidate
        for start in starts
        if (candidate := complete(start)) is not None
    }
    evaluated = [
        evaluate_attacker_uncertainty_portfolio(
            graph, profiles, actions, ids, reserve_budget, constraints,
            response_pool_size, max_phase_one_paths, max_response_delay,
            response_pool_mode, worst_weight, variation_weight,
            method_name=(
                "TwoStage-Attacker-Uncertainty-PairStart-Greedy"
                if start_size == 2
                else "TwoStage-Attacker-Uncertainty-MultiStart-Greedy"
            ),
        )
        for ids in portfolios
    ]
    selected = min(evaluated, key=lambda result: (
        result.objective, result.worst_terminal_risk, result.mean_terminal_risk,
        portfolio_cost(result.initial_actions), tuple(action.id for action in result.initial_actions),
    ))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "static_construction_starts": len(starts),
        "static_start_size": start_size,
        "distinct_static_portfolios": len(portfolios),
    })
    if memoize_static_scores:
        diagnostics.update({
            "memoized_static_score_entries": len(static_score_cache),
            "memoized_static_score_hits": static_score_cache_hits,
        })
    return AttackerUncertaintyResult(
        selected.method, selected.initial_actions, selected.profile_results,
        selected.mean_terminal_risk, selected.worst_terminal_risk,
        selected.risk_stddev, selected.objective, diagnostics,
    )


def attacker_uncertainty_static_baseline(
    graph: nx.DiGraph,
    profiles: Sequence[str],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    strategy: str,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    seed: int = 7,
) -> AttackerUncertaintyResult:
    """Construct a static portfolio, then score it by the shared objective."""
    from src.optimizer.two_stage import (
        two_stage_genetic_baseline,
        two_stage_random_baseline,
        two_stage_risk_topk_baseline,
    )

    profiles = _validate_profiles(profiles)
    if strategy == "random":
        static = two_stage_random_baseline(
            graph, actions, initial_budget, reserve_budget, constraints,
            response_pool_size, max_phase_one_paths, max_response_delay,
            response_pool_mode, trials=100, seed=seed,
        )
    elif strategy == "risk_topk":
        static = two_stage_risk_topk_baseline(
            graph, actions, initial_budget, reserve_budget, constraints,
            response_pool_size, max_phase_one_paths, max_response_delay, response_pool_mode,
        )
    elif strategy == "genetic":
        static = two_stage_genetic_baseline(
            graph, actions, initial_budget, reserve_budget, constraints,
            response_pool_size, max_phase_one_paths, max_response_delay,
            response_pool_mode, population_size=24, generations=16, seed=seed,
        )
    else:
        raise ValueError("strategy must be random, risk_topk, or genetic")
    result = evaluate_attacker_uncertainty_portfolio(
        graph, profiles, actions, frozenset(action.id for action in static.initial_actions),
        reserve_budget, constraints, response_pool_size, max_phase_one_paths,
        max_response_delay, response_pool_mode,
        method_name=f"TwoStage-Attacker-Uncertainty-{strategy}",
    )
    diagnostics = dict(result.diagnostics)
    diagnostics.update({"static_strategy": strategy, "static_seed": seed})
    return AttackerUncertaintyResult(
        result.method, result.initial_actions, result.profile_results,
        result.mean_terminal_risk, result.worst_terminal_risk,
        result.risk_stddev, result.objective, diagnostics,
    )


def attacker_uncertainty_exact_enumeration(
    graph: nx.DiGraph,
    profiles: Sequence[str],
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = None,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
) -> AttackerUncertaintyResult:
    """Enumerate all initial portfolios for an at-most-12-action reference."""
    profiles = _validate_profiles(profiles)
    if len(actions) > 12:
        raise ValueError("multi-profile exact enumeration is limited to 12 actions")
    feasible = [
        frozenset(action.id for action in portfolio)
        for count in range(len(actions) + 1)
        for portfolio in combinations(actions, count)
        if portfolio_cost(portfolio) <= initial_budget
        and (constraints is None or constraints.is_satisfied(frozenset(action.id for action in portfolio)))
    ]
    evaluated = [
        evaluate_attacker_uncertainty_portfolio(
            graph, profiles, actions, ids, reserve_budget, constraints,
            response_pool_size, max_phase_one_paths, max_response_delay,
            response_pool_mode, worst_weight, variation_weight,
            method_name="TwoStage-Attacker-Uncertainty-Exact",
        )
        for ids in feasible
    ]
    selected = min(evaluated, key=lambda result: (
        result.objective, result.worst_terminal_risk, result.mean_terminal_risk,
        portfolio_cost(result.initial_actions), tuple(action.id for action in result.initial_actions),
    ))
    diagnostics = dict(selected.diagnostics)
    diagnostics.update({
        "enumerated_initial_portfolios": 2 ** len(actions),
        "feasible_initial_portfolios": len(feasible),
    })
    return AttackerUncertaintyResult(
        selected.method, selected.initial_actions, selected.profile_results,
        selected.mean_terminal_risk, selected.worst_terminal_risk,
        selected.risk_stddev, selected.objective, diagnostics,
    )


def attacker_uncertainty_codef_search(
    graph: nx.DiGraph,
    profiles: Sequence[str],
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
    max_action_candidates: int | None = None,
    method_name: str = "TwoStage-CoDef-Attacker-Uncertainty",
    candidate_action_ids: Sequence[str] | None = None,
    response_scope: str = "local",
    replan_attacker: bool = True,
) -> AttackerUncertaintyResult:
    """Search one initial portfolio against several attacker profiles."""
    profiles = _validate_profiles(profiles)
    if initial_budget < 0 or reserve_budget < 0:
        raise ValueError("stage budgets must be non-negative")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    actions_by_id = {action.id: action for action in actions}
    candidate_actions = actions
    if candidate_action_ids is not None:
        ordered = [actions_by_id[action_id] for action_id in candidate_action_ids if action_id in actions_by_id]
        if ordered:
            candidate_actions = ordered
    if max_action_candidates is not None and candidate_action_ids is None:
        if max_action_candidates < 1:
            raise ValueError("max_action_candidates must be at least 1")

        def relevance(action: DefenseAction) -> tuple[float, str]:
            if isinstance(action.target, tuple):
                source, target = action.target
                edge_risk = float(graph.get_edge_data(source, target, {}).get("risk", 0.0))
                asset_value = float(graph.nodes[target].get("asset_value", 1.0)) if target in graph else 1.0
                score = action.effect * (edge_risk + asset_value)
            else:
                node = str(action.target)
                incident = sum(
                    float(data.get("risk", 0.0))
                    for _, _, data in graph.in_edges(node, data=True)
                ) + sum(
                    float(data.get("risk", 0.0))
                    for _, _, data in graph.out_edges(node, data=True)
                ) if node in graph else 0.0
                asset_value = float(graph.nodes[node].get("asset_value", 1.0)) if node in graph else 1.0
                critical_bonus = 1.5 if bool(graph.nodes[node].get("critical", False)) else 1.0
                score = action.effect * (incident + asset_value) * critical_bonus
            return score / max(action.cost, 1e-9), action.id

        candidate_actions = sorted(actions, key=relevance, reverse=True)[:max_action_candidates]
    elif max_action_candidates is not None:
        if max_action_candidates < 1:
            raise ValueError("max_action_candidates must be at least 1")
        candidate_actions = candidate_actions[:max_action_candidates]
    cache: dict[frozenset[str], AttackerUncertaintyResult] = {}

    def evaluate(ids: frozenset[str]) -> AttackerUncertaintyResult:
        if ids not in cache:
            cache[ids] = evaluate_attacker_uncertainty_portfolio(
                graph, profiles, actions, ids, reserve_budget,
                constraints=constraints,
                response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths,
                max_response_delay=max_response_delay,
                response_pool_mode=response_pool_mode,
                worst_weight=worst_weight,
                variation_weight=variation_weight,
                method_name=method_name,
                response_scope=response_scope,
                replan_attacker=replan_attacker,
            )
        return cache[ids]

    def key(ids: frozenset[str]):
        result = evaluate(ids)
        return (
            result.objective,
            result.worst_terminal_risk,
            result.mean_terminal_risk,
            portfolio_cost(actions_by_id[action_id] for action_id in ids),
            tuple(sorted(ids)),
        )

    greedy_ids = frozenset(action.id for action in greedy_baseline(
        graph, actions, initial_budget, constraints=constraints,
    ).actions)
    beam = [frozenset()]
    seen = set(beam)
    feasible_initials = {frozenset(), greedy_ids}
    expanded = 0
    while beam:
        candidates: set[frozenset[str]] = set()
        for ids in beam:
            for action in candidate_actions:
                expanded += 1
                candidate = _dependency_closed(ids | {action.id}, constraints)
                if candidate in seen:
                    continue
                seen.add(candidate)
                selected = [actions_by_id[action_id] for action_id in candidate]
                if portfolio_cost(selected) > initial_budget:
                    continue
                if constraints is not None and not constraints.is_satisfied(candidate):
                    continue
                candidates.add(candidate)
                feasible_initials.add(candidate)
        if not candidates:
            break
        beam = sorted(candidates, key=key)[:beam_width]

    best_ids = min(feasible_initials, key=key)
    result = evaluate(best_ids)
    diagnostics = dict(result.diagnostics)
    diagnostics.update({
        "seeded_by_greedy": True,
        "expanded_initial_candidates": expanded,
        "evaluated_initial_portfolios": len(cache),
        "candidate_action_count": len(candidate_actions),
        "max_action_candidates": max_action_candidates,
        "response_scope": response_scope,
        "replan_attacker": replan_attacker,
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


def attacker_uncertainty_local_refine(
    graph: nx.DiGraph,
    profiles: Sequence[str],
    actions: list[DefenseAction],
    seed_portfolios: Sequence[frozenset[str]],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    response_pool_mode: str = "adaptive_diverse",
    worst_weight: float = 0.5,
    variation_weight: float = 0.2,
    max_neighbors_per_seed: int = 48,
    method_name: str = "TwoStage-CoDef-v25-LocalRefine",
) -> AttackerUncertaintyResult:
    """Refine a few trusted portfolios with profile-aware one-action moves.

    The v2.4 beam search is intentionally narrow for speed. This refinement
    stage explores add/remove/replace-one-action neighbors around the public
    v2.4 result and any route-conditioned candidate, then evaluates every
    neighbor under all attacker profiles. It is bounded so it cannot turn the
    main search into an exhaustive enumeration.
    """
    profiles = _validate_profiles(profiles)
    if initial_budget < 0 or reserve_budget < 0:
        raise ValueError("stage budgets must be non-negative")
    if max_neighbors_per_seed < 1:
        raise ValueError("max_neighbors_per_seed must be at least 1")
    actions_by_id = {action.id: action for action in actions}
    action_order = sorted(actions, key=lambda action: (-action.effect / max(action.cost, 1e-9), action.id))
    cache: dict[frozenset[str], AttackerUncertaintyResult] = {}

    def feasible(ids: frozenset[str]) -> bool:
        if portfolio_cost(actions_by_id[action_id] for action_id in ids) > initial_budget:
            return False
        return constraints is None or constraints.is_satisfied(ids)

    def evaluate(ids: frozenset[str]) -> AttackerUncertaintyResult:
        if ids not in cache:
            cache[ids] = evaluate_attacker_uncertainty_portfolio(
                graph, profiles, actions, ids, reserve_budget,
                constraints=constraints,
                response_pool_size=response_pool_size,
                max_phase_one_paths=max_phase_one_paths,
                max_response_delay=max_response_delay,
                response_pool_mode=response_pool_mode,
                worst_weight=worst_weight,
                variation_weight=variation_weight,
                method_name=method_name,
            )
        return cache[ids]

    candidates: set[frozenset[str]] = set()
    for seed in seed_portfolios:
        if feasible(seed):
            candidates.add(seed)
        neighbors: list[frozenset[str]] = []
        removals: tuple[str | None, ...] = (None, *sorted(seed))
        for removed in removals:
            base = seed if removed is None else frozenset(seed - {removed})
            for action in action_order:
                candidate = _dependency_closed(base | {action.id}, constraints)
                if candidate == seed or not feasible(candidate):
                    continue
                neighbors.append(candidate)
        for candidate in sorted(set(neighbors), key=lambda ids: (
            portfolio_cost(actions_by_id[action_id] for action_id in ids), tuple(sorted(ids)),
        ))[:max_neighbors_per_seed]:
            candidates.add(candidate)
    if not candidates:
        raise ValueError("no feasible local-refinement portfolio")
    best = min(
        (evaluate(ids) for ids in candidates),
        key=lambda result: (
            result.objective,
            result.worst_terminal_risk,
            result.mean_terminal_risk,
            portfolio_cost(result.initial_actions),
            tuple(action.id for action in result.initial_actions),
        ),
    )
    diagnostics = dict(best.diagnostics)
    diagnostics.update({
        "local_refine": True,
        "seed_portfolio_count": len(seed_portfolios),
        "local_candidate_count": len(candidates),
        "max_neighbors_per_seed": max_neighbors_per_seed,
    })
    return AttackerUncertaintyResult(
        best.method,
        best.initial_actions,
        best.profile_results,
        best.mean_terminal_risk,
        best.worst_terminal_risk,
        best.risk_stddev,
        best.objective,
        diagnostics,
    )
