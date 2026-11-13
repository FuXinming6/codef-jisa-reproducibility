"""Offline two-stage attacker replanning and adaptive defense optimization.

The model deliberately stays inside the project's abstract graph simulator.
It does not inspect, scan, or alter a live network.  A defender deploys an
initial portfolio, an attacker reaches a midpoint foothold, the defender
observes the traversed prefix and spends a reserved budget in that exposed
region, and the attacker then replans from the foothold.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import prod
from random import Random

import networkx as nx

from src.defense.action import DefenseAction, DefenseType
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.graph.graph_builder import critical_assets
from src.graph.path_search import find_attack_paths
from src.optimizer.baselines import greedy_baseline
from src.optimizer.codef import get_dynamic_beam_size
from src.optimizer.common import TwoStageResult
from src.risk.network_risk import assess_network_risk


@dataclass(frozen=True)
class _StageEvaluation:
    terminal_risk: float
    observed_path: tuple[str, ...]
    replanned_path: tuple[str, ...]
    foothold: str | None
    response_ids: frozenset[str]
    eligible_response_ids: frozenset[str]


def _path_probability(graph: nx.DiGraph, path: tuple[str, ...]) -> float:
    return prod(float(graph[source][target]["risk"]) for source, target in zip(path, path[1:]))


def _committed_route_evaluation(
    initial_graph: nx.DiGraph,
    response_graph: nx.DiGraph,
    observed_path: tuple[str, ...],
    committed_path: tuple[str, ...],
    response_ids: frozenset[str],
    eligible_response_ids: frozenset[str],
) -> _StageEvaluation:
    """Score a fixed attempted suffix, retaining it in evidence if blocked."""
    split = _foothold_index(committed_path)
    suffix = committed_path[split:] if split is not None else committed_path
    survives = all(
        response_graph.has_edge(source, target)
        for source, target in zip(suffix, suffix[1:])
    )
    risk = 0.0
    if survives:
        risk = (
            _path_probability(initial_graph, observed_path)
            * _path_probability(response_graph, suffix)
            * float(response_graph.nodes[suffix[-1]].get("asset_value", 1.0))
        )
    return _StageEvaluation(
        risk, observed_path, suffix, observed_path[-1], response_ids, eligible_response_ids,
    )


def _foothold_index(path: tuple[str, ...]) -> int | None:
    """Pick a deterministic interior node for the phase-one foothold."""
    if len(path) < 3:
        return None
    return max(1, (len(path) - 1) // 2)


def _validate_attacker_profile(attacker_profile: str) -> None:
    if attacker_profile not in {"risk_max", "shortest_path", "asset_targeted"}:
        raise ValueError(
            "attacker_profile must be risk_max, shortest_path, or asset_targeted"
        )


def _attacker_path_key(
    graph: nx.DiGraph,
    path: tuple[str, ...],
    attacker_profile: str,
) -> tuple[float, float, tuple[str, ...]]:
    """Rank one continuation from the attacker's declared behavior model."""
    _validate_attacker_profile(attacker_profile)
    risk = _path_probability(graph, path) * float(graph.nodes[path[-1]].get("asset_value", 1.0))
    asset_value = float(graph.nodes[path[-1]].get("asset_value", 1.0))
    if attacker_profile == "shortest_path":
        return (-float(len(path)), risk, path)
    if attacker_profile == "asset_targeted":
        return (asset_value, risk, path)
    return (risk, float(len(path)), path)


def _stage_evaluation_key(
    graph: nx.DiGraph,
    evaluation: _StageEvaluation,
    attacker_profile: str,
) -> tuple[float, float, tuple[str, ...], tuple[str, ...]]:
    """Select which observed path the modeled attacker would prefer."""
    _validate_attacker_profile(attacker_profile)
    replanned = evaluation.replanned_path or evaluation.observed_path
    terminal_asset = float(graph.nodes[replanned[-1]].get("asset_value", 1.0)) if replanned else 0.0
    if attacker_profile == "shortest_path":
        total_hops = len(evaluation.observed_path) + len(evaluation.replanned_path)
        return (-float(total_hops), evaluation.terminal_risk, evaluation.observed_path, evaluation.replanned_path)
    if attacker_profile == "asset_targeted":
        return (terminal_asset, evaluation.terminal_risk, evaluation.observed_path, evaluation.replanned_path)
    return (evaluation.terminal_risk, float(len(replanned)), evaluation.observed_path, evaluation.replanned_path)


def exposed_region(graph: nx.DiGraph, observed_path: tuple[str, ...]) -> frozenset[str]:
    """Return exposed path nodes plus one-hop options visible at the foothold."""
    if not observed_path:
        return frozenset()
    foothold = observed_path[-1]
    return frozenset(observed_path) | frozenset(graph.successors(foothold))


def response_actions_for_region(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    observed_path: tuple[str, ...],
) -> list[DefenseAction]:
    """Restrict response actions to the observed path and foothold neighborhood."""
    region = exposed_region(graph, observed_path)
    eligible: list[DefenseAction] = []
    for action in actions:
        if action.action_type == DefenseType.EDGE_RESTRICTION:
            source, target = action.target  # type: ignore[misc]
            if source in region or target in region:
                eligible.append(action)
        elif str(action.target) in region:
            eligible.append(action)
    return eligible


def _dependency_closed(
    ids: frozenset[str],
    constraints: PortfolioConstraints | None,
) -> frozenset[str]:
    if constraints is None:
        return ids
    closed = set(ids)
    pending = list(ids)
    while pending:
        action_id = pending.pop()
        for required in constraints.requirements.get(action_id, frozenset()):
            if required not in closed:
                closed.add(required)
                pending.append(required)
    return frozenset(closed)


def _select_response(
    initial_graph: nx.DiGraph,
    initial_ids: frozenset[str],
    actions_by_id: dict[str, DefenseAction],
    reserve_budget: float,
    observed_path: tuple[str, ...],
    constraints: PortfolioConstraints | None,
    response_pool_size: int,
    max_response_delay: int | None,
    response_graph_cache: dict[frozenset[str], nx.DiGraph],
    continuation_cache: dict[tuple[object, ...], tuple[tuple[str, ...], float]],
    response_ranking: str,
    use_response_cache: bool,
    response_pool_mode: str,
    attacker_profile: str = "risk_max",
    response_scope: str = "local",
    replan_attacker: bool = True,
    committed_paths: tuple[tuple[str, ...], ...] | None = None,
) -> _StageEvaluation:
    """Choose a bounded, path-visible response against one reached foothold."""
    _validate_attacker_profile(attacker_profile)
    foothold = observed_path[-1]
    if response_scope == "local":
        scope_actions = response_actions_for_region(initial_graph, list(actions_by_id.values()), observed_path)
    elif response_scope == "global":
        scope_actions = list(actions_by_id.values())
    else:
        raise ValueError("response_scope must be local or global")
    available = [
        action for action in scope_actions
        if action.id not in initial_ids and (max_response_delay is None or action.response_delay <= max_response_delay)
    ]
    if not replan_attacker and (
        not committed_paths
        or any(path[:len(observed_path)] != observed_path for path in committed_paths)
    ):
        raise ValueError("no-replanning evaluation requires committed paths matching the observed prefix")
    baseline_paths = find_attack_paths(initial_graph, sources=[foothold])
    if not baseline_paths:
        return _StageEvaluation(0.0, observed_path, (), foothold, frozenset(), frozenset())

    ranked: list[tuple[float, str]] = []
    if response_ranking == "legacy_simulation":
        baseline_score = max(
            _path_probability(initial_graph, path) * float(initial_graph.nodes[path[-1]].get("asset_value", 1.0))
            for path in baseline_paths
        )
        for action in available:
            reduced = apply_defense(initial_graph, [action])
            next_score = max((
                _path_probability(reduced, path) * float(reduced.nodes[path[-1]].get("asset_value", 1.0))
                for path in find_attack_paths(reduced, sources=[foothold])
            ), default=0.0)
            ranked.append(((baseline_score - next_score) / action.cost, action.id))
    elif response_ranking == "path_coverage":
        # v1.1 ranking: retain actions that cover the most currently reachable
        # continuation risk. All retained portfolios are evaluated exactly.
        for action in available:
            touched_risk = 0.0
            for path in baseline_paths:
                if action.action_type == DefenseType.EDGE_RESTRICTION:
                    source, target = action.target  # type: ignore[misc]
                    touches_path = (source, target) in zip(path, path[1:])
                else:
                    touches_path = str(action.target) in path
                if touches_path:
                    touched_risk += _path_probability(initial_graph, path) * float(
                        initial_graph.nodes[path[-1]].get("asset_value", 1.0)
                    )
            ranked.append((touched_risk * action.effect / action.cost, action.id))
    else:
        raise ValueError("response_ranking must be legacy_simulation or path_coverage")
    ranked.sort(key=lambda item: (-item[0], item[1]))
    if response_pool_mode == "fixed":
        pool_ids = [action_id for _, action_id in ranked[:response_pool_size]]
    elif response_pool_mode == "adaptive_diverse":
        # Expand only when several response actions have comparable coverage;
        # then prefer actions that protect different targets or edge pairs.
        desired = response_pool_size
        if len(ranked) > response_pool_size and ranked[response_pool_size - 1][0] > 0:
            desired = min(len(ranked), response_pool_size + 2)
        pool_ids = []
        covered_targets: set[object] = set()
        for _, action_id in ranked:
            action = actions_by_id[action_id]
            target_key: object = tuple(action.target) if isinstance(action.target, tuple) else action.target
            if target_key not in covered_targets:
                pool_ids.append(action_id)
                covered_targets.add(target_key)
            if len(pool_ids) == desired:
                break
        if len(pool_ids) < desired:
            pool_ids.extend(action_id for _, action_id in ranked if action_id not in pool_ids)
            pool_ids = pool_ids[:desired]
    else:
        raise ValueError("response_pool_mode must be fixed or adaptive_diverse")

    best_key = (float("inf"), float("inf"), ())
    best_ids = frozenset()
    best_replanned = ()
    prefix_probability = _path_probability(initial_graph, observed_path)
    for size in range(len(pool_ids) + 1):
        for subset in combinations(pool_ids, size):
            response_ids = frozenset(subset)
            # Defense operations commute mathematically, but repeated floating
            # point multiplications do not. Canonicalize the action order so
            # a frozenset's hash iteration cannot perturb tie comparisons.
            response = [actions_by_id[action_id] for action_id in sorted(response_ids)]
            if portfolio_cost(response) > reserve_budget:
                continue
            all_ids = initial_ids | response_ids
            if constraints is not None and not constraints.is_satisfied(all_ids):
                continue
            continuation_key = (foothold, response_ids) if replan_attacker else (foothold, response_ids, committed_paths)
            if use_response_cache and response_ids not in response_graph_cache:
                response_graph_cache[response_ids] = apply_defense(initial_graph, response)
            if not use_response_cache or continuation_key not in continuation_cache:
                response_graph = response_graph_cache[response_ids] if use_response_cache else apply_defense(initial_graph, response)
                if replan_attacker:
                    continuations = find_attack_paths(response_graph, sources=[foothold])
                    if continuations:
                        worst_path = max(
                            continuations,
                            key=lambda continuation: _attacker_path_key(
                                response_graph, continuation, attacker_profile,
                            ),
                        )
                    else:
                        worst_path = ()
                    if worst_path:
                        raw_risk = _path_probability(response_graph, worst_path) * float(
                            response_graph.nodes[worst_path[-1]].get("asset_value", 1.0)
                        )
                    else:
                        raw_risk = 0.0
                else:
                    assert committed_paths is not None
                    attempted = [
                        _committed_route_evaluation(
                            initial_graph, response_graph, observed_path, path,
                            response_ids, frozenset(pool_ids),
                        )
                        for path in committed_paths
                    ]
                    worst_evaluation = max(
                        attempted,
                        key=lambda item: _stage_evaluation_key(initial_graph, item, attacker_profile),
                    )
                    worst_path = worst_evaluation.replanned_path
                    raw_risk = worst_evaluation.terminal_risk / max(prefix_probability, 1e-300)
                if use_response_cache:
                    continuation_cache[continuation_key] = (worst_path, raw_risk)
            worst_path, raw_risk = continuation_cache[continuation_key] if use_response_cache else (worst_path, raw_risk)
            worst_risk = prefix_probability * raw_risk
            key = (worst_risk, portfolio_cost(response), tuple(sorted(response_ids)))
            if key < best_key:
                best_key, best_ids, best_replanned = key, response_ids, worst_path
    return _StageEvaluation(
        best_key[0], observed_path, best_replanned, foothold, best_ids, frozenset(pool_ids)
    )


def evaluate_two_stage_strategy(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 8,
    max_phase_one_paths: int | None = None,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "fixed",
    attacker_profile: str = "risk_max",
    response_scope: str = "local",
    replan_attacker: bool = True,
    foothold_offset: int = 0,
) -> _StageEvaluation:
    """Evaluate the defender's worst phase-one path under adaptive response.

    foothold_offset shifts the deterministic midpoint index and clamps it to
    an interior node. The default zero preserves the benchmark model; nonzero
    values support fixed-policy observation-timing sensitivity analyses.
    """
    _validate_attacker_profile(attacker_profile)
    if isinstance(foothold_offset, bool) or not isinstance(foothold_offset, int):
        raise ValueError("foothold_offset must be an integer number of edges")
    if response_pool_size < 1:
        raise ValueError("response_pool_size must be at least 1")
    if max_phase_one_paths is not None and max_phase_one_paths < 1:
        raise ValueError("max_phase_one_paths must be positive when supplied")
    if max_response_delay is not None and max_response_delay < 0:
        raise ValueError("max_response_delay must be non-negative when supplied")
    actions_by_id = {action.id: action for action in actions}
    initial = [actions_by_id[action_id] for action_id in sorted(initial_ids)]
    initial_graph = apply_defense(graph, initial)
    evaluations: list[_StageEvaluation] = []
    response_graph_cache: dict[frozenset[str], nx.DiGraph] = {}
    continuation_cache: dict[tuple[object, ...], tuple[tuple[str, ...], float]] = {}
    phase_one_paths = find_attack_paths(initial_graph)
    if max_phase_one_paths is not None or not replan_attacker:
        if attacker_profile == "shortest_path":
            ranked_phase_one_paths = sorted(
                phase_one_paths,
                key=lambda path: (
                    len(path),
                    -_path_probability(initial_graph, path) * float(initial_graph.nodes[path[-1]].get("asset_value", 1.0)),
                    path,
                ),
            )
        elif attacker_profile == "asset_targeted":
            ranked_phase_one_paths = sorted(
                phase_one_paths,
                key=lambda path: (
                    -float(initial_graph.nodes[path[-1]].get("asset_value", 1.0)),
                    -_path_probability(initial_graph, path),
                    path,
                ),
            )
        else:
            ranked_phase_one_paths = sorted(
                phase_one_paths,
                key=lambda path: (
                    -_path_probability(initial_graph, path) * float(initial_graph.nodes[path[-1]].get("asset_value", 1.0)),
                    path,
                ),
            )
        if max_phase_one_paths is not None:
            ranked_phase_one_paths = ranked_phase_one_paths[:max_phase_one_paths]
        phase_one_paths = ranked_phase_one_paths
    path_groups: dict[tuple[str, ...], list[tuple[str, ...]]] = {}
    for path in phase_one_paths:
        split = _foothold_index(path)
        if split is not None and foothold_offset:
            split = min(len(path) - 2, max(1, split + foothold_offset))
        if split is None:
            # A direct ingress-to-asset path has no interior foothold and no
            # response opportunity, but it is still a successful attack.
            if len(path) >= 2:
                terminal_risk = _path_probability(initial_graph, path) * float(
                    initial_graph.nodes[path[-1]].get("asset_value", 1.0)
                )
                evaluations.append(
                    _StageEvaluation(
                        terminal_risk,
                        path,
                        path,
                        path[-1],
                        frozenset(),
                        frozenset(),
                    )
                )
            continue
        observed_path = path[: split + 1]
        path_groups.setdefault(observed_path, []).append(path)
    for observed_path, committed_paths in path_groups.items():
        selected_response = _select_response(
            initial_graph, initial_ids, actions_by_id, reserve_budget,
            observed_path, constraints, response_pool_size,
            max_response_delay, response_graph_cache, continuation_cache,
            response_ranking, use_response_cache, response_pool_mode,
            attacker_profile, response_scope, replan_attacker,
            None if replan_attacker else tuple(committed_paths),
        )
        if replan_attacker:
            evaluations.extend([selected_response] * len(committed_paths))
            continue
        response = [actions_by_id[action_id] for action_id in sorted(selected_response.response_ids)]
        if use_response_cache and selected_response.response_ids in response_graph_cache:
            response_graph = response_graph_cache[selected_response.response_ids]
        else:
            response_graph = apply_defense(initial_graph, response)
        evaluations.extend(
            _committed_route_evaluation(
                initial_graph, response_graph, observed_path, path,
                selected_response.response_ids, selected_response.eligible_response_ids,
            )
            for path in committed_paths
        )
    if not evaluations:
        return _StageEvaluation(0.0, (), (), None, frozenset(), frozenset())
    return max(
        evaluations,
        key=lambda item: _stage_evaluation_key(initial_graph, item, attacker_profile),
    )


def _result_from_evaluation(
    method_name: str,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    evaluation: _StageEvaluation,
    diagnostics: dict[str, object],
) -> TwoStageResult:
    actions_by_id = {action.id: action for action in actions}
    initial = tuple(actions_by_id[action_id] for action_id in sorted(initial_ids))
    response = tuple(actions_by_id[action_id] for action_id in sorted(evaluation.response_ids))
    diagnostics.update({
        "eligible_response_actions": tuple(sorted(evaluation.eligible_response_ids)),
        "response_action_count": len(response),
        "initial_action_count": len(initial),
    })
    return TwoStageResult(
        method=method_name,
        initial_actions=initial,
        response_actions=response,
        observed_path=evaluation.observed_path,
        replanned_path=evaluation.replanned_path,
        foothold=evaluation.foothold,
        initial_cost=portfolio_cost(initial),
        response_cost=portfolio_cost(response),
        terminal_risk=evaluation.terminal_risk,
        objective=evaluation.terminal_risk,
        diagnostics=diagnostics,
    )


def evaluate_two_stage_initial_plan(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_ids: frozenset[str],
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 8,
    max_phase_one_paths: int | None = None,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "fixed",
    method_name: str = "TwoStage-FixedInitialPlan",
) -> TwoStageResult:
    """Score a fixed initial portfolio under a specified response-time window.

    This supports a fair delay-blind baseline: choose initial actions while
    assuming a broad response window, then re-evaluate that unchanged initial
    portfolio when the real response window is short.
    """
    actions_by_id = {action.id: action for action in actions}
    if not initial_ids.issubset(actions_by_id):
        raise ValueError("initial_ids must belong to actions")
    if constraints is not None and not constraints.is_satisfied(initial_ids):
        raise ValueError("initial_ids must satisfy portfolio constraints")
    evaluation = evaluate_two_stage_strategy(
        graph, actions, initial_ids, reserve_budget, constraints,
        response_pool_size, max_phase_one_paths, max_response_delay, response_ranking, use_response_cache,
        response_pool_mode,
    )
    return _result_from_evaluation(
        method_name, actions, initial_ids, evaluation,
        {"reserve_budget": reserve_budget, "max_phase_one_paths": max_phase_one_paths,
         "max_response_delay": max_response_delay, "fixed_initial_plan": True},
    )


def _feasible_portfolio_from_order(
    ordered_ids: list[str],
    actions_by_id: dict[str, DefenseAction],
    budget: float,
    constraints: PortfolioConstraints | None,
) -> frozenset[str]:
    """Build a budget-feasible portfolio while closing declared dependencies."""
    selected = frozenset()
    for action_id in ordered_ids:
        candidate = _dependency_closed(selected | {action_id}, constraints)
        candidate_actions = [actions_by_id[item] for item in candidate]
        if portfolio_cost(candidate_actions) <= budget and (constraints is None or constraints.allows_partial(candidate)):
            selected = candidate
    return selected if constraints is None or constraints.is_satisfied(selected) else frozenset()


def _two_stage_result_for_static_ids(
    graph: nx.DiGraph, actions: list[DefenseAction], initial_ids: frozenset[str], reserve_budget: float,
    constraints: PortfolioConstraints | None, response_pool_size: int, max_phase_one_paths: int | None,
    max_response_delay: int | None, response_pool_mode: str, method_name: str,
) -> TwoStageResult:
    return evaluate_two_stage_initial_plan(
        graph, actions, initial_ids, reserve_budget, constraints, response_pool_size,
        max_phase_one_paths, max_response_delay, "path_coverage", True, response_pool_mode, method_name,
    )


def two_stage_random_baseline(
    graph: nx.DiGraph, actions: list[DefenseAction], initial_budget: float, reserve_budget: float,
    constraints: PortfolioConstraints | None = None, response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12, max_response_delay: int | None = None,
    response_pool_mode: str = "fixed", trials: int = 100, seed: int = 7,
    method_name: str = "TwoStage-Random",
) -> TwoStageResult:
    """Best static-risk portfolio among reproducible feasible random samples."""
    rng = Random(seed); actions_by_id = {action.id: action for action in actions}
    candidates = [frozenset()]
    for _ in range(max(1, trials)):
        order = [action.id for action in actions]; rng.shuffle(order)
        selected = _feasible_portfolio_from_order(
            [action_id for action_id in order if rng.random() < 0.5], actions_by_id, initial_budget, constraints,
        )
        candidates.append(selected)
    best_ids = min(candidates, key=lambda ids: (
        assess_network_risk(apply_defense(graph, [actions_by_id[item] for item in ids])).score,
        tuple(sorted(ids)),
    ))
    return _two_stage_result_for_static_ids(graph, actions, best_ids, reserve_budget, constraints,
                                             response_pool_size, max_phase_one_paths, max_response_delay,
                                             response_pool_mode, method_name)


def two_stage_risk_topk_baseline(
    graph: nx.DiGraph, actions: list[DefenseAction], initial_budget: float, reserve_budget: float,
    constraints: PortfolioConstraints | None = None, response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12, max_response_delay: int | None = None,
    response_pool_mode: str = "fixed", method_name: str = "TwoStage-RiskTopK",
) -> TwoStageResult:
    """Rank one-action static reductions, then use the shared response model."""
    actions_by_id = {action.id: action for action in actions}
    before = assess_network_risk(graph).score
    order = [action.id for action in sorted(
        actions,
        key=lambda action: (-(before - assess_network_risk(apply_defense(graph, [action])).score) / action.cost, action.id),
    )]
    selected = _feasible_portfolio_from_order(order, actions_by_id, initial_budget, constraints)
    return _two_stage_result_for_static_ids(graph, actions, selected, reserve_budget, constraints,
                                             response_pool_size, max_phase_one_paths, max_response_delay,
                                             response_pool_mode, method_name)


def two_stage_genetic_baseline(
    graph: nx.DiGraph, actions: list[DefenseAction], initial_budget: float, reserve_budget: float,
    constraints: PortfolioConstraints | None = None, response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12, max_response_delay: int | None = None,
    response_pool_mode: str = "fixed", population_size: int = 24, generations: int = 16,
    mutation_rate: float = 0.10, seed: int = 7, method_name: str = "TwoStage-Genetic",
) -> TwoStageResult:
    """Constrained genetic search over static initial portfolios, then response."""
    rng = Random(seed); actions_by_id = {action.id: action for action in actions}; action_ids = [action.id for action in actions]

    def random_ids() -> frozenset[str]:
        shuffled = action_ids[:]; rng.shuffle(shuffled)
        return _feasible_portfolio_from_order([item for item in shuffled if rng.random() < 0.5], actions_by_id, initial_budget, constraints)

    def static_score(ids: frozenset[str]) -> float:
        return assess_network_risk(apply_defense(graph, [actions_by_id[item] for item in ids])).score

    population = [random_ids() for _ in range(max(4, population_size))]
    population[0] = frozenset()
    for _ in range(generations):
        ranked = sorted(population, key=lambda ids: (static_score(ids), tuple(sorted(ids))))
        next_population = ranked[:2]
        while len(next_population) < len(population):
            first, second = rng.choice(ranked[: max(2, len(ranked) // 2)]), rng.choice(ranked[: max(2, len(ranked) // 2)])
            desired = [item for item in action_ids if (item in first if rng.random() < 0.5 else item in second)]
            desired = [item for item in desired if rng.random() >= mutation_rate]
            if rng.random() < mutation_rate:
                desired.append(rng.choice(action_ids))
            next_population.append(_feasible_portfolio_from_order(desired, actions_by_id, initial_budget, constraints))
        population = next_population
    best_ids = min(population, key=lambda ids: (static_score(ids), tuple(sorted(ids))))
    return _two_stage_result_for_static_ids(graph, actions, best_ids, reserve_budget, constraints,
                                             response_pool_size, max_phase_one_paths, max_response_delay,
                                             response_pool_mode, method_name)


def two_stage_greedy_baseline(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 8,
    max_phase_one_paths: int | None = None,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "fixed",
    method_name: str = "TwoStage-Greedy",
) -> TwoStageResult:
    """Static-first Greedy baseline followed by a path-visible response."""
    greedy = greedy_baseline(graph, actions, initial_budget, constraints=constraints, method_name=method_name)
    initial_ids = frozenset(action.id for action in greedy.actions)
    evaluation = evaluate_two_stage_strategy(
        graph, actions, initial_ids, reserve_budget, constraints, response_pool_size, max_phase_one_paths,
        max_response_delay, response_ranking, use_response_cache, response_pool_mode,
    )
    return _result_from_evaluation(
        method_name, actions, initial_ids, evaluation,
        {"initial_budget": initial_budget, "reserve_budget": reserve_budget,
         "max_phase_one_paths": max_phase_one_paths, "max_response_delay": max_response_delay,
         "response_ranking": response_ranking, "use_response_cache": use_response_cache,
         "response_pool_mode": response_pool_mode,
         "seeded_by_greedy": False},
    )


def two_stage_codef_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    beam_width: int = 6,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 8,
    max_phase_one_paths: int | None = None,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "fixed",
    use_dynamic_beam: bool = False,
    dynamic_beam_min: int = 2,
    use_initial_exchange: bool = False,
    exchange_pool_size: int = 4,
    exchange_max_removed: int = 2,
    exchange_eval_limit: int = 12,
    exchange_ranking: str = "static",
    method_name: str = "TwoStage-CoDef-Guarded",
) -> TwoStageResult:
    """Beam-search initial deployments against adaptive attacker replanning.

    A static Greedy plan is always included as a seed.  The returned strategy
    is therefore never worse than this two-stage Greedy baseline under the
    same budgets and response-visibility rule.
    """
    if initial_budget < 0 or reserve_budget < 0:
        raise ValueError("stage budgets must be non-negative")
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if exchange_pool_size < 2 or exchange_eval_limit < 1 or not 1 <= exchange_max_removed <= 2:
        raise ValueError("invalid bounded initial-exchange parameters")
    if use_dynamic_beam and not 2 <= dynamic_beam_min <= beam_width:
        raise ValueError("dynamic_beam_min must be between 2 and beam_width")
    if exchange_ranking not in {"static", "terminal_risk"}:
        raise ValueError("exchange_ranking must be static or terminal_risk")
    actions_by_id = {action.id: action for action in actions}
    greedy = greedy_baseline(graph, actions, initial_budget, constraints=constraints)
    greedy_ids = frozenset(action.id for action in greedy.actions)
    cache: dict[frozenset[str], _StageEvaluation] = {}

    def score(ids: frozenset[str]) -> _StageEvaluation:
        if ids not in cache:
            cache[ids] = evaluate_two_stage_strategy(
                graph, actions, ids, reserve_budget, constraints, response_pool_size, max_phase_one_paths,
                max_response_delay, response_ranking, use_response_cache, response_pool_mode,
            )
        return cache[ids]

    beam = [frozenset()]
    seen = set(beam)
    feasible_initials = {frozenset(), greedy_ids}
    expanded = 0
    pruned_budget = 0
    pruned_constraint = 0
    beam_target_history: list[int] = []
    beam_retained_history: list[int] = []
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
        ranked_candidates = sorted(
            candidates,
            key=lambda ids: (
                score(ids).terminal_risk,
                portfolio_cost(actions_by_id[action_id] for action_id in ids),
                tuple(sorted(ids)),
            ),
        )
        current_beam_width = (
            get_dynamic_beam_size(
                len(graph.nodes),
                [score(ids).terminal_risk for ids in ranked_candidates],
                min_beam_size=dynamic_beam_min,
                max_beam_size=beam_width,
            )
            if use_dynamic_beam else beam_width
        )
        beam_target_history.append(current_beam_width)
        beam = ranked_candidates[:current_beam_width]
        beam_retained_history.append(len(beam))

    best_ids = min(
        feasible_initials,
        key=lambda ids: (
            score(ids).terminal_risk,
            portfolio_cost(actions_by_id[action_id] for action_id in ids),
            tuple(sorted(ids)),
        ),
    )
    exchange_proposals = 0
    exchange_evaluated = 0
    exchange_improved = False
    if use_initial_exchange and best_ids:
        current_graph = apply_defense(graph, [actions_by_id[action_id] for action_id in best_ids])
        before_static = assess_network_risk(current_graph).score
        proposals = []
        for action in actions:
            if action.id in best_ids:
                continue
            after_static = assess_network_risk(apply_defense(current_graph, [action])).score
            proposals.append(((before_static - after_static) / action.cost, action.id))
        proposal_pool = [action_id for _, action_id in sorted(proposals, key=lambda item: (-item[0], item[1]))[:exchange_pool_size]]
        exchange_candidates: set[frozenset[str]] = set()
        for remove_count in range(1, min(exchange_max_removed, len(best_ids)) + 1):
            for removed in combinations(sorted(best_ids), remove_count):
                remaining = best_ids - set(removed)
                for add_count in (1, 2):
                    for added in combinations(proposal_pool, add_count):
                        candidate = _dependency_closed(frozenset(remaining | set(added)), constraints)
                        selected = [actions_by_id[action_id] for action_id in candidate]
                        if portfolio_cost(selected) <= initial_budget and (constraints is None or constraints.is_satisfied(candidate)):
                            exchange_candidates.add(candidate)
        exchange_proposals = len(exchange_candidates)
        # Static risk only proposes a bounded set; terminal risk remains the
        # exact deciding objective for every retained exchange candidate.
        if exchange_ranking == "terminal_risk":
            retained = sorted(
                exchange_candidates,
                key=lambda ids: (score(ids).terminal_risk, tuple(sorted(ids))),
            )[:exchange_eval_limit]
        else:
            retained = sorted(
                exchange_candidates,
                key=lambda ids: (
                    assess_network_risk(apply_defense(graph, [actions_by_id[action_id] for action_id in ids])).score,
                    tuple(sorted(ids)),
                ),
            )[:exchange_eval_limit]
        exchange_evaluated = len(retained)
        candidate_best = min(
            [best_ids, *retained],
            key=lambda ids: (
                score(ids).terminal_risk,
                portfolio_cost(actions_by_id[action_id] for action_id in ids),
                tuple(sorted(ids)),
            ),
        )
        exchange_improved = candidate_best != best_ids
        best_ids = candidate_best
    evaluation = score(best_ids)
    return _result_from_evaluation(
        method_name, actions, best_ids, evaluation,
        {
            "initial_budget": initial_budget,
            "reserve_budget": reserve_budget,
            "max_phase_one_paths": max_phase_one_paths,
            "max_response_delay": max_response_delay,
            "response_ranking": response_ranking,
            "use_response_cache": use_response_cache,
            "response_pool_mode": response_pool_mode,
            "use_dynamic_beam": use_dynamic_beam,
            "dynamic_beam_min": dynamic_beam_min,
            "beam_target_history": tuple(beam_target_history),
            "beam_retained_history": tuple(beam_retained_history),
            "seeded_by_greedy": True,
            "expanded_initial_candidates": expanded,
            "evaluated_initial_portfolios": len(cache),
            "pruned_initial_budget": pruned_budget,
            "pruned_initial_constraint": pruned_constraint,
            "exchange_proposals": exchange_proposals,
            "exchange_evaluated": exchange_evaluated,
            "exchange_improved": exchange_improved,
            "exchange_ranking": exchange_ranking,
        },
    )


def two_stage_v21_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    response_pool_size: int = 4,
    max_phase_one_paths: int | None = 12,
    max_response_delay: int | None = None,
    method_name: str = "TwoStage-CoDef-v2.1",
) -> TwoStageResult:
    """Run the consolidated v2.1 strategy.

    The five-strategy study showed that the useful, repeatable contribution is
    the Greedy guard plus bounded two-stage Beam search with a diverse response
    pool. Dynamic Beam and initial exchange remain available in the lower-level
    function for historical ablations, but are disabled in the official v2.1
    path because they did not lower risk in the held-out comparisons and
    increased runtime.
    """
    return two_stage_codef_search(
        graph,
        actions,
        initial_budget,
        reserve_budget,
        beam_width=2,
        constraints=constraints,
        response_pool_size=response_pool_size,
        max_phase_one_paths=max_phase_one_paths,
        max_response_delay=max_response_delay,
        response_ranking="path_coverage",
        use_response_cache=True,
        response_pool_mode="adaptive_diverse",
        use_dynamic_beam=False,
        dynamic_beam_min=2,
        use_initial_exchange=False,
        exchange_pool_size=6,
        exchange_max_removed=2,
        exchange_eval_limit=16,
        exchange_ranking="static",
        method_name=method_name,
    )


def two_stage_exact_enumeration(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    initial_budget: float,
    reserve_budget: float,
    constraints: PortfolioConstraints | None = None,
    max_response_delay: int | None = None,
    response_ranking: str = "path_coverage",
    use_response_cache: bool = True,
    response_pool_mode: str = "fixed",
    method_name: str = "TwoStage-Exact",
) -> TwoStageResult:
    """Enumerate every phase-one portfolio for a small two-stage benchmark.

    Response selection receives the entire visible action set, so this is an
    exact baseline for the model's current budgets, observability rule and
    abstract network graph.  It is deliberately capped at 12 actions.
    """
    if len(actions) > 12:
        raise ValueError("two-stage exact enumeration is limited to 12 actions")
    actions_by_id = {action.id: action for action in actions}
    best_ids = frozenset()
    best_evaluation: _StageEvaluation | None = None
    evaluated = 0
    feasible = 0
    for size in range(len(actions) + 1):
        for portfolio in combinations(actions, size):
            evaluated += 1
            ids = frozenset(action.id for action in portfolio)
            if portfolio_cost(portfolio) > initial_budget:
                continue
            if constraints is not None and not constraints.is_satisfied(ids):
                continue
            feasible += 1
            evaluation = evaluate_two_stage_strategy(
                graph, actions, ids, reserve_budget, constraints, response_pool_size=len(actions),
                max_response_delay=max_response_delay, response_ranking=response_ranking,
                use_response_cache=use_response_cache, response_pool_mode=response_pool_mode,
            )
            key = (evaluation.terminal_risk, portfolio_cost(portfolio), tuple(sorted(ids)))
            current_key = (
                best_evaluation.terminal_risk if best_evaluation is not None else float("inf"),
                portfolio_cost(actions_by_id[action_id] for action_id in best_ids),
                tuple(sorted(best_ids)),
            )
            if key < current_key:
                best_ids, best_evaluation = ids, evaluation
    assert best_evaluation is not None
    return _result_from_evaluation(
        method_name, actions, best_ids, best_evaluation,
        {"initial_budget": initial_budget, "reserve_budget": reserve_budget,
         "enumerated_initial_portfolios": evaluated, "feasible_initial_portfolios": feasible,
         "exact_response_pool": len(actions), "max_response_delay": max_response_delay},
    )
