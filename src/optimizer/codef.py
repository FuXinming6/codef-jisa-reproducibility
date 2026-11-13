from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations
from collections import Counter

import networkx as nx

from src.defense.action import DefenseAction
from src.defense.apply_defense import apply_defense, portfolio_cost
from src.defense.constraints import PortfolioConstraints
from src.defense.synergy import SynergyEvaluator
from src.graph.graph_builder import ingress_nodes
from src.graph.path_search import find_attack_paths
from src.optimizer.common import OptimizationResult, evaluate_portfolio
from src.risk.network_risk import (
    assess_fixed_path_risk,
    assess_network_risk,
    ingress_risk_vector,
    robust_risk,
)


def get_dynamic_beam_size(
    node_count: int,
    ranked_worst_case_risks: list[float],
    min_beam_size: int = 2,
    max_beam_size: int = 8,
) -> int:
    """Choose a beam width from topology scale and the best two candidates.

    The input risks must already be ordered from lower (better) to higher.
    Small graphs start wide, while 50-node-or-larger graphs start narrow. When
    the leading candidates are close, the beam widens to preserve alternatives;
    when the lead is clear, it narrows to avoid unnecessary evaluations.
    """
    if min_beam_size < 2:
        raise ValueError("min_beam_size must be at least 2")
    if min_beam_size > max_beam_size:
        raise ValueError("min_beam_size cannot exceed max_beam_size")
    if node_count <= 20:
        base_size = 8
    elif node_count <= 40:
        base_size = 4
    else:
        base_size = 2
    base_size = max(min_beam_size, min(base_size, max_beam_size))
    if len(ranked_worst_case_risks) < 2:
        return base_size

    best, second = ranked_worst_case_risks[:2]
    relative_gap = (second - best) / max(abs(best), 1e-6)
    if relative_gap < 0.03:
        return min(base_size + 2, max_beam_size)
    if relative_gap > 0.15:
        return max(base_size - 1, min_beam_size)
    return base_size


def codef_beam_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    budget: float,
    beam_width: int = 6,
    synergy_weight: float = 0.0,
    use_synergy: bool = True,
    use_robust: bool = True,
    robust_weight: float = 1.0,
    adaptive_replanning: bool = True,
    method_name: str = "CoDef",
) -> OptimizationResult:
    """Budgeted CoDef search with switches for controlled ablation studies."""
    if not 0 <= robust_weight <= 1:
        raise ValueError("robust_weight must be between 0 and 1")
    action_by_id = {action.id: action for action in actions}
    base_paths = find_attack_paths(graph)
    sources = ingress_nodes(graph)
    synergy_evaluator = SynergyEvaluator(graph, actions) if use_synergy else None
    objective_cache: dict[frozenset[str], float] = {}

    def ranking_risk(selected: list[DefenseAction]) -> float:
        defended = apply_defense(graph, selected)
        if not adaptive_replanning:
            return assess_fixed_path_risk(defended, base_paths).score
        if use_robust:
            worst_entry_risk = robust_risk(defended, sources)
            if robust_weight == 1:
                return worst_entry_risk
            global_risk = assess_network_risk(defended).score
            return robust_weight * worst_entry_risk + (1 - robust_weight) * global_risk
        return assess_network_risk(defended).score

    def objective_for(selected_ids: frozenset[str]) -> float:
        if selected_ids not in objective_cache:
            selected = [action_by_id[action_id] for action_id in selected_ids]
            synergy = synergy_evaluator.portfolio_synergy(selected) if synergy_evaluator else 0.0
            objective_cache[selected_ids] = ranking_risk(selected) - synergy_weight * synergy
        return objective_cache[selected_ids]

    beam: list[frozenset[str]] = [frozenset()]
    seen = set(beam)
    best_ids = frozenset()
    best_objective = float("inf")

    while beam:
        candidates: list[tuple[float, frozenset[str]]] = []
        for selected_ids in beam:
            objective = objective_for(selected_ids)
            if objective < best_objective:
                best_ids, best_objective = selected_ids, objective
            for action in actions:
                expanded = selected_ids | {action.id}
                if expanded in seen:
                    continue
                if portfolio_cost(action_by_id[action_id] for action_id in expanded) <= budget:
                    seen.add(expanded)
                    candidates.append((objective_for(expanded), expanded))
        candidates.sort(key=lambda item: (item[0], tuple(sorted(item[1]))))
        beam = [selected for _, selected in candidates[:beam_width]]

    chosen = [action_by_id[action_id] for action_id in sorted(best_ids)]
    return evaluate_portfolio(graph, chosen, method_name, objective=best_objective)


def codef_pareto_search(
    graph: nx.DiGraph,
    actions: list[DefenseAction],
    budget: float,
    beam_width: int = 8,
    use_dynamic_beam: bool = False,
    dynamic_beam_min: int = 2,
    use_dominance_pruning: bool = True,
    exchange_pool_size: int = 10,
    exchange_max_removed: int = 1,
    use_local_exchange: bool = True,
    constraints: PortfolioConstraints | None = None,
    initial_portfolios: Iterable[Iterable[DefenseAction]] | None = None,
    method_name: str = "CoDef-Pareto",
) -> OptimizationResult:
    """Robust beam search that preserves useful defenses for every ingress.

    The v0.5 beam ranks a partial portfolio only by its current maximum ingress
    risk.  Here each partial portfolio keeps an ingress-risk vector.  A portfolio
    is removed only if another one costs no more and is no worse at *every*
    ingress.  This lets the search retain a defense that improves a currently
    secondary ingress, so it can later combine with a defense for the worst one.
    A bounded one-to-one / one-to-two exchange then repairs early choices using
    the exact robust objective.
    """
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")
    if use_dynamic_beam and not 2 <= dynamic_beam_min <= beam_width:
        raise ValueError("dynamic_beam_min must be between 2 and beam_width")
    if exchange_pool_size < 2:
        raise ValueError("exchange_pool_size must be at least 2")
    if not 1 <= exchange_max_removed <= 2:
        raise ValueError("exchange_max_removed must be 1 or 2")

    action_by_id = {action.id: action for action in actions}
    sources = tuple(ingress_nodes(graph))
    vector_cache: dict[frozenset[str], tuple[float, ...]] = {}
    cost_cache: dict[frozenset[str], float] = {}
    diagnostics: Counter[str] = Counter()

    def selected_actions(ids: frozenset[str]) -> list[DefenseAction]:
        return [action_by_id[action_id] for action_id in sorted(ids)]

    def cost_for(ids: frozenset[str]) -> float:
        if ids not in cost_cache:
            cost_cache[ids] = portfolio_cost(selected_actions(ids))
        return cost_cache[ids]

    def vector_for(ids: frozenset[str]) -> tuple[float, ...]:
        if ids not in vector_cache:
            defended = apply_defense(graph, selected_actions(ids))
            vector_cache[ids] = ingress_risk_vector(defended, sources)
        return vector_cache[ids]

    def rank_key(ids: frozenset[str]) -> tuple[float, float, float, tuple[str, ...]]:
        vector = vector_for(ids)
        return (max(vector, default=0.0), sum(vector), cost_for(ids), tuple(sorted(ids)))

    def dominates(left: frozenset[str], right: frozenset[str]) -> bool:
        left_vector, right_vector = vector_for(left), vector_for(right)
        no_worse = cost_for(left) <= cost_for(right) and all(
            left_score <= right_score for left_score, right_score in zip(left_vector, right_vector)
        )
        strictly_better = cost_for(left) < cost_for(right) or any(
            left_score < right_score for left_score, right_score in zip(left_vector, right_vector)
        )
        return no_worse and strictly_better

    def pareto_frontier(candidates: set[frozenset[str]]) -> list[frozenset[str]]:
        if not use_dominance_pruning:
            return list(candidates)
        frontier = [
            candidate
            for candidate in candidates
            if not any(other != candidate and dominates(other, candidate) for other in candidates)
        ]
        diagnostics["pruned_dominated"] += len(candidates) - len(frontier)
        return frontier

    def diverse_frontier(frontier: list[frozenset[str]], current_beam_width: int) -> list[frozenset[str]]:
        if len(frontier) <= current_beam_width:
            return sorted(frontier, key=rank_key)

        # One best robust candidate plus rotating ingress-specific leaders keeps
        # the beam compact without throwing away all secondary-ingress progress.
        ranked_lists = [sorted(frontier, key=rank_key)]
        ranked_lists.extend(
            sorted(frontier, key=lambda ids, index=index: (vector_for(ids)[index], rank_key(ids)))
            for index in range(len(sources))
        )
        retained: list[frozenset[str]] = []
        for position in range(len(frontier)):
            for ranked in ranked_lists:
                if ranked[position] not in retained:
                    retained.append(ranked[position])
                    if len(retained) == current_beam_width:
                        diagnostics["pruned_beam_trim"] += len(frontier) - len(retained)
                        return retained
        return retained

    beam: list[frozenset[str]] = [frozenset()]
    seen: set[frozenset[str]] = set(beam)
    explored: set[frozenset[str]] = set(beam)
    for portfolio in initial_portfolios or ():
        ids = frozenset(action.id for action in portfolio)
        if not ids.issubset(action_by_id):
            raise ValueError("initial portfolio contains an unknown action")
        if cost_for(ids) <= budget and (constraints is None or constraints.is_satisfied(ids)):
            explored.add(ids)
    diagnostics["seed_portfolios"] = len(explored) - 1
    beam_target_history: list[int] = []
    beam_retained_history: list[int] = []
    while beam:
        expansions: set[frozenset[str]] = set()
        for ids in beam:
            for action in actions:
                diagnostics["expanded_candidates"] += 1
                expanded = ids | {action.id}
                if expanded in seen:
                    diagnostics["pruned_duplicate"] += 1
                    continue
                seen.add(expanded)
                if constraints is not None and not constraints.allows_partial(expanded):
                    diagnostics["pruned_constraint"] += 1
                    continue
                if cost_for(expanded) > budget:
                    diagnostics["pruned_budget"] += 1
                    continue
                expansions.add(expanded)
        if not expansions:
            break
        explored.update(expansions)
        frontier = pareto_frontier(expansions)
        ranked_risks = [rank_key(ids)[0] for ids in sorted(frontier, key=rank_key)]
        current_beam_width = (
            get_dynamic_beam_size(
                graph.number_of_nodes(), ranked_risks, dynamic_beam_min, beam_width
            )
            if use_dynamic_beam
            else beam_width
        )
        beam_target_history.append(current_beam_width)
        beam = diverse_frontier(frontier, current_beam_width)
        beam_retained_history.append(len(beam))

    diagnostics["explored_portfolios"] = len(explored)
    diagnostics["beam_rounds"] = len(beam_target_history)
    diagnostics["beam_target_total"] = sum(beam_target_history)
    diagnostics["beam_retained_total"] = sum(beam_retained_history)
    diagnostics["beam_target_history"] = tuple(beam_target_history)
    diagnostics["beam_retained_history"] = tuple(beam_retained_history)
    feasible_portfolios = [
        ids for ids in explored if constraints is None or constraints.is_satisfied(ids)
    ]
    diagnostics["final_feasible_portfolios"] = len(feasible_portfolios)
    best_ids = min(feasible_portfolios, key=rank_key)

    if use_local_exchange and best_ids:
        improved = True
        while improved:
            improved = False
            current = best_ids
            current_actions = selected_actions(current)
            current_graph = apply_defense(graph, current_actions)
            # Cheap global gain ranks a small proposal pool; every retained
            # exchange is still checked against the exact worst-ingress score.
            proposals = []
            before = assess_network_risk(current_graph).score
            for action in actions:
                if action.id in current:
                    continue
                after = assess_network_risk(apply_defense(current_graph, [action])).score
                proposals.append(((before - after) / action.cost, action.id))
            pool = [action_id for _, action_id in sorted(proposals, key=lambda item: (-item[0], item[1]))[:exchange_pool_size]]
            candidate_ids: set[frozenset[str]] = set()
            for removed_count in range(1, min(exchange_max_removed, len(current)) + 1):
                for removed in combinations(sorted(current), removed_count):
                    remaining = current - set(removed)
                    for added_count in (1, 2):
                        for added in combinations(pool, added_count):
                            candidate_ids.add(remaining | set(added))
            diagnostics["exchange_proposals"] += len(candidate_ids)
            feasible = []
            for ids in candidate_ids:
                if cost_for(ids) > budget:
                    diagnostics["exchange_pruned_budget"] += 1
                    continue
                if constraints is not None and not constraints.is_satisfied(ids):
                    diagnostics["exchange_pruned_constraint"] += 1
                    continue
                feasible.append(ids)
            if feasible:
                candidate = min(feasible, key=rank_key)
                if rank_key(candidate) < rank_key(current):
                    best_ids = candidate
                    improved = True

    chosen = selected_actions(best_ids)
    return evaluate_portfolio(
        graph, chosen, method_name, objective=rank_key(best_ids)[0], diagnostics=dict(diagnostics)
    )
